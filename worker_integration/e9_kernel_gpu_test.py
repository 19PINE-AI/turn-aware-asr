"""E9: direct GPU unit test of the Metronome sink Triton kernel (the decisive kernel proof).

E7 validates the sink through the full vLLM serving path; E9 validates the KERNEL ITSELF by
calling the exact `metronome_sink_kernel.unified_attention` entry point vLLM calls, with a
single prefilled sequence over a contiguous paged KV, and comparing its output to the
pure-PyTorch `reference_union_attention` (the ground truth). No engine, no KV manager, no
CUDA-graph capture — one small kernel launch, so it is immune to the init hangs / co-tenant
contention that make the full-engine path flaky on this shared box.

The kernel bakes the sink size S from METRONOME_SINK_TOKENS at import (a tl.constexpr), so
run one process per S. For each requested window W it checks:
  (a) kernel output == reference_union_attention(q,k,v, window=W, sink=S)  (correctness)
  (b) when S>0 and W<T: kernel output != window-only reference                (sink has effect)

Usage (system python3.10 / vLLM env):
    METRONOME_SINK_TOKENS=0 python3 worker_integration/e9_kernel_gpu_test.py --windows 8,40
    METRONOME_SINK_TOKENS=6 python3 worker_integration/e9_kernel_gpu_test.py --windows 8
"""
import argparse, os
import torch


def reference_union_attention(q, k, v, window, sink):
    """Ground-truth [0,sink) ∪ [t-window, t] causal attention, single head. q,k,v: [T,D]."""
    T, D = q.shape
    pos = torch.arange(T, device=q.device)
    qi = pos[:, None]; ki = pos[None, :]
    causal = ki <= qi
    in_window = (qi - ki) < window
    in_sink = ki < sink
    mask = causal & (in_window | in_sink)
    scores = (q.float() @ k.float().T) / (D ** 0.5)
    scores = scores.masked_fill(~mask, float("-inf"))
    w = torch.softmax(scores, dim=-1)
    return (w @ v.float())


def run_window(uattn, T, H, Hkv, D, block_size, W, S, dtype, device):
    """Build a single prefill sequence over a contiguous paged KV, run the kernel, compare."""
    torch.manual_seed(1234 + W + 1000 * S)
    q = torch.randn(T, H, D, dtype=dtype, device=device)
    k = torch.randn(T, Hkv, D, dtype=dtype, device=device)
    v = torch.randn(T, Hkv, D, dtype=dtype, device=device)

    num_blocks = (T + block_size - 1) // block_size
    key_cache = torch.zeros(num_blocks, block_size, Hkv, D, dtype=dtype, device=device)
    val_cache = torch.zeros(num_blocks, block_size, Hkv, D, dtype=dtype, device=device)
    for t in range(T):
        key_cache[t // block_size, t % block_size] = k[t]
        val_cache[t // block_size, t % block_size] = v[t]

    out = torch.empty(T, H, D, dtype=dtype, device=device)
    cu_seqlens_q = torch.tensor([0, T], dtype=torch.int32, device=device)
    seqused_k = torch.tensor([T], dtype=torch.int32, device=device)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device)[None, :]
    scale = 1.0 / (D ** 0.5)

    uattn(q, key_cache, val_cache, out, cu_seqlens_q, T, seqused_k, T, scale,
          True, (W - 1, 0), block_table, 0.0, None, None, None)
    torch.cuda.synchronize()

    # reference per (query head -> its kv head)
    per_kv = H // Hkv
    max_union, max_winonly = 0.0, 0.0
    for h in range(H):
        kv = h // per_kv
        ref = reference_union_attention(q[:, h, :], k[:, kv, :], v[:, kv, :], W, S).to(dtype)
        won = reference_union_attention(q[:, h, :], k[:, kv, :], v[:, kv, :], W, 0).to(dtype)
        max_union = max(max_union, (out[:, h, :].float() - ref.float()).abs().max().item())
        max_winonly = max(max_winonly, (out[:, h, :].float() - won.float()).abs().max().item())
    return max_union, max_winonly


def run_decode(uattn, ctx, H, Hkv, D, block_size, W, S, dtype, device):
    """Decode step: ONE new query at position `ctx`, attending a cached KV of length ctx+1
    (the streaming serving path, context_len>0). Compares to the union reference."""
    torch.manual_seed(4321 + W + 1000 * S)
    T = ctx + 1
    kfull = torch.randn(T, Hkv, D, dtype=dtype, device=device)
    vfull = torch.randn(T, Hkv, D, dtype=dtype, device=device)
    q = torch.randn(1, H, D, dtype=dtype, device=device)  # the single decode query
    num_blocks = (T + block_size - 1) // block_size
    kc = torch.zeros(num_blocks, block_size, Hkv, D, dtype=dtype, device=device)
    vc = torch.zeros(num_blocks, block_size, Hkv, D, dtype=dtype, device=device)
    for t in range(T):
        kc[t // block_size, t % block_size] = kfull[t]
        vc[t // block_size, t % block_size] = vfull[t]
    out = torch.empty(1, H, D, dtype=dtype, device=device)
    cu = torch.tensor([0, 1], dtype=torch.int32, device=device)
    su = torch.tensor([T], dtype=torch.int32, device=device)
    bt = torch.arange(num_blocks, dtype=torch.int32, device=device)[None, :]
    uattn(q, kc, vc, out, cu, 1, su, T, 1.0 / (D ** 0.5), True, (W - 1, 0), bt, 0.0,
          None, None, None)
    torch.cuda.synchronize()
    # reference: the single query at abs pos ctx over keys [0,T)
    per_kv = H // Hkv
    pos = torch.arange(T, device=device)
    max_u, max_w = 0.0, 0.0
    for h in range(H):
        kv = h // per_kv
        qh, kh, vh = q[0, h], kfull[:, kv], vfull[:, kv]
        for win, tag in [(W, "u"), (0, "w")]:
            m = (pos <= ctx) & (((ctx - pos) < W) | (pos < (S if win else 0)))
            sc = (qh.float() @ kh.float().T) / (D ** 0.5)
            sc = sc.masked_fill(~m, float("-inf"))
            ref = (torch.softmax(sc, -1) @ vh.float())
            d = (out[0, h].float() - ref.float()).abs().max().item()
            if tag == "u":
                max_u = max(max_u, d)
            else:
                max_w = max(max_w, d)
    return max_u, max_w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="prefill", choices=["prefill", "decode"])
    ap.add_argument("--windows", default="8,40")
    ap.add_argument("--T", type=int, default=40)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--head-size", type=int, default=64)
    ap.add_argument("--block-size", type=int, default=16)
    args = ap.parse_args()

    S = int(os.environ.get("METRONOME_SINK_TOKENS", "0") or "0")
    device = "cuda"
    dtype = torch.float16

    import metronome_sink_kernel as mk
    assert mk._SINK == S, f"kernel _SINK={mk._SINK} != env S={S} (import order?)"
    print(f"### E9 kernel GPU test: S={S}, T={args.T}, H={args.heads}, Hkv={args.kv_heads}, "
          f"D={args.head_size}, block={args.block_size}, dtype={dtype}")

    tol = 4e-3
    ok = True
    for W in [int(x) for x in args.windows.split(",")]:
        if args.mode == "decode":
            du, dw = run_decode(mk.unified_attention, args.T - 1, args.heads, args.kv_heads,
                                args.head_size, args.block_size, W, S, dtype, device)
        else:
            du, dw = run_window(mk.unified_attention, args.T, args.heads, args.kv_heads,
                                args.head_size, args.block_size, W, S, dtype, device)
        # (a) kernel must match the union reference
        match = du < tol
        # (b) if sink can matter (S>0, W<T, and there are keys in [0,S) outside the window)
        sink_can_matter = S > 0 and W < args.T
        effect = dw if sink_can_matter else float("nan")
        verdict = "PASS" if match else "FAIL"
        extra = ""
        if sink_can_matter:
            extra = f"| sink-effect vs window-only={dw:.4f} ({'sink visible' if dw > tol else 'NO effect!'})"
            if dw <= tol:
                verdict = "FAIL"  # sink was supposed to change the output but didn't
        print(f"  W={W:>3d} S={S:<4d}  |kernel-union|={du:.5f}  {verdict}  {extra}")
        ok = ok and (verdict == "PASS")

    print("E9_PASS" if ok else "E9_FAIL")


if __name__ == "__main__":
    main()
