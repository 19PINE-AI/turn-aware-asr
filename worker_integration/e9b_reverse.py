"""E9b: reverse-engineer what mask the sink kernel ACTUALLY computes, by matching its
output against a battery of candidate reference masks. Run with METRONOME_SINK_TOKENS set."""
import os
import torch


def ref_mask(q, k, v, mask):
    T, D = q.shape
    s = (q.float() @ k.float().T) / (D ** 0.5)
    s = s.masked_fill(~mask, float("-inf"))
    return torch.softmax(s, -1) @ v.float()


def main():
    S = int(os.environ.get("METRONOME_SINK_TOKENS", "0") or "0")
    W = int(os.environ.get("E9B_W", "4"))
    T, H, Hkv, D, bs = 40, 4, 2, 64, 16
    device, dtype = "cuda", torch.float16
    torch.manual_seed(1234 + W + 1000 * S)
    q = torch.randn(T, H, D, dtype=dtype, device=device)
    k = torch.randn(T, Hkv, D, dtype=dtype, device=device)
    v = torch.randn(T, Hkv, D, dtype=dtype, device=device)
    nb = (T + bs - 1) // bs
    kc = torch.zeros(nb, bs, Hkv, D, dtype=dtype, device=device)
    vc = torch.zeros(nb, bs, Hkv, D, dtype=dtype, device=device)
    for t in range(T):
        kc[t // bs, t % bs] = k[t]; vc[t // bs, t % bs] = v[t]
    out = torch.empty(T, H, D, dtype=dtype, device=device)
    cu = torch.tensor([0, T], dtype=torch.int32, device=device)
    su = torch.tensor([T], dtype=torch.int32, device=device)
    bt = torch.arange(nb, dtype=torch.int32, device=device)[None, :]

    import metronome_sink_kernel as mk
    mk.unified_attention(q, kc, vc, out, cu, T, su, T, 1.0 / (D ** 0.5),
                         True, (W - 1, 0), bt, 0.0, None, None, None)
    torch.cuda.synchronize()

    pos = torch.arange(T, device=device)
    qi, ki = pos[:, None], pos[None, :]
    causal = ki <= qi
    cands = {
        "causal": causal,
        f"window-{W}": causal & ((qi - ki) < W),
        f"union(W{W},S{S})": causal & (((qi - ki) < W) | (ki < S)),
        f"union(W{W},S=TILE32)": causal & (((qi - ki) < W) | (ki < 32)),
        f"union(W{W},S=bs16)": causal & (((qi - ki) < W) | (ki < 16)),
        f"sink-only(S{S})": causal & (ki < S),
    }
    print(f"### S={S} W={W}: matching kernel output to candidate masks (per-head max |diff|)")
    per_kv = H // Hkv
    for name, m in cands.items():
        mx = 0.0
        for h in range(H):
            r = ref_mask(q[:, h], k[:, h // per_kv], v[:, h // per_kv], m).to(dtype)
            mx = max(mx, (out[:, h].float() - r.float()).abs().max().item())
        print(f"  {name:24s} max|diff|={mx:.5f}{'   <-- MATCH' if mx < 4e-3 else ''}")


if __name__ == "__main__":
    main()
