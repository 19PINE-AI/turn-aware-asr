"""E10: unit-test the Metronome block-pin (_PinnedSlidingWindowManager.remove_skipped_blocks)
without the full engine — the second half of Part B (the first half, the union kernel, is
validated by E9).

The manager frees KV blocks that scroll out of the sliding window. The sink patch must keep
the first ceil(S/block) blocks (the CTX sink) resident while freeing only the MIDDLE
[sink_blocks, window) blocks. This drives the ACTUAL shipped method (obtained from
spec_manager_map after register()) with a mocked KV state across a growing session, and
checks the invariants:
  * sink blocks [0, ceil(S/block)) are NEVER freed (always non-null),
  * middle blocks behind the window ARE freed (set to null),
  * window blocks (the last W tokens) are kept,
  * against the stock base manager, ONLY the sink blocks differ (base frees them too).

Runs under system python3.10 (vLLM). No GPU needed.
"""
import argparse


class _NullBlock:
    def __repr__(self):
        return "NULL"


class _Block:
    def __init__(self, i):
        self.i = i

    def __repr__(self):
        return f"B{self.i}"


class _Pool:
    def __init__(self):
        self.freed = []

    def free_blocks(self, blocks):
        self.freed.extend(b.i for b in blocks)


def _make(cls, S, block_size, window, null):
    """Instantiate the manager bypassing __init__, then set the attributes the method uses."""
    mgr = cls.__new__(cls)
    mgr.block_size = block_size
    mgr.sliding_window = window
    mgr._null_block = null
    mgr.block_pool = _Pool()
    return mgr


def sink_blocks_for(S, block_size):
    return (S + block_size - 1) // block_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=160)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--total-tokens", type=int, default=2048)
    args = ap.parse_args()

    from metronome_sink import _install_block_pin
    from vllm.v1.core import single_type_kv_cache_manager as m
    from vllm.v1.kv_cache_interface import SlidingWindowSpec

    Base = m.SlidingWindowManager
    _install_block_pin(args.S)
    Pinned = m.spec_manager_map[SlidingWindowSpec]
    assert Pinned is not Base and getattr(Pinned, "_metronome_sink", None) == args.S, \
        "block-pin not installed"

    bs, W, S = args.block_size, args.window, args.S
    nblocks = (args.total_tokens + bs - 1) // bs
    sink_nb = sink_blocks_for(S, bs)
    null = _NullBlock()
    print(f"### E10 block-pin: S={S} block={bs} -> sink_blocks={sink_nb}; window={W}; "
          f"session={args.total_tokens} tokens ({nblocks} blocks)")

    # Fresh block lists for the pinned and the stock base manager, driven identically.
    pin = _make(Pinned, S, bs, W, null)
    base = _make(Base, S, bs, W, null)
    pin.req_to_blocks = {"r": [_Block(i) for i in range(nblocks)]}
    base.req_to_blocks = {"r": [_Block(i) for i in range(nblocks)]}

    ok = True
    # Walk the session forward one block at a time (decode advancing the window).
    for computed in range(bs, args.total_tokens + 1, bs):
        pin.remove_skipped_blocks("r", computed)
        base.remove_skipped_blocks("r", computed)

    pin_blocks = pin.req_to_blocks["r"]
    base_blocks = base.req_to_blocks["r"]
    window_first_tok = max(0, args.total_tokens - W + 1)
    window_first_blk = window_first_tok // bs

    # (1) sink blocks never freed
    sink_ok = all(pin_blocks[i] is not null for i in range(sink_nb))
    # (2) middle blocks (behind window, past sink) freed
    mid_lo, mid_hi = sink_nb, window_first_blk
    mid_ok = all(pin_blocks[i] is null for i in range(mid_lo, mid_hi)) if mid_hi > mid_lo else True
    # (3) window blocks kept
    win_ok = all(pin_blocks[i] is not null for i in range(window_first_blk, nblocks))
    # (4) vs base: differ ONLY on the sink region [0, sink_nb)
    diff_idx = [i for i in range(nblocks) if (pin_blocks[i] is null) != (base_blocks[i] is null)]
    only_sink = all(i < sink_nb for i in diff_idx)
    base_freed_sink = any(base_blocks[i] is null for i in range(sink_nb))

    print(f"  sink kept [0,{sink_nb}):           {'OK' if sink_ok else 'FAIL'}")
    print(f"  middle freed [{mid_lo},{mid_hi}):  {'OK' if mid_ok else 'FAIL'} "
          f"(freed {len(set(pin.block_pool.freed))} blocks)")
    print(f"  window kept [{window_first_blk},{nblocks}):  {'OK' if win_ok else 'FAIL'}")
    print(f"  differs from base ONLY in sink:  {'OK' if only_sink else 'FAIL'} "
          f"(base freed sink blocks: {base_freed_sink})")
    # sink blocks must NOT be in the freed pool
    sink_ids = set(range(sink_nb))
    freed_sink = sink_ids & set(pin.block_pool.freed)
    print(f"  sink blocks never in freed pool: {'OK' if not freed_sink else 'FAIL'} "
          f"({sorted(freed_sink)})")

    ok = sink_ok and mid_ok and win_ok and only_sink and base_freed_sink and not freed_sink
    print("E10_PASS" if ok else "E10_FAIL")


if __name__ == "__main__":
    main()
