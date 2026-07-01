"""Generate a 4-context re10k eval index from the official 2-context one.

The official ``assets/evaluation_index_re10k.json`` has 2 context + 3 target views
per scene. For the pose-robustness experiment we need 4 context views (so that
corrupting 2 of 4 still leaves 2 clean refs). This script keeps the official 2
context views and adds 2 interior anchor frames between them (deterministic,
evenly spaced), matching the train-side 4-anchor ``_select_views`` layout.

Output: ``assets/evaluation_index_re10k_4ctx.json`` with
``context = [c1, c3, c4, c2]`` (sorted; c1<c3<c4<c2) and the official target list
unchanged. ``ref0=c1, ref3=c2`` give the widest baseline for identity_unit_distance
normalisation (same as train).

Run: python scripts/generate_re10k_4ctx_index.py
"""
from __future__ import annotations
import json
import os
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / "assets"
SRC = ASSETS / "evaluation_index_re10k.json"
DST = ASSETS / "evaluation_index_re10k_4ctx.json"


def main():
    with open(SRC) as f:
        idx = json.load(f)
    out = {}
    n_skipped = 0
    for scene, v in idx.items():
        if v is None:
            out[scene] = None
            continue
        ctx = sorted(v["context"])
        if len(ctx) != 2:
            n_skipped += 1
            out[scene] = v  # leave as-is
            continue
        c1, c2 = ctx
        span = c2 - c1
        c3 = c1 + round(span / 3)
        c4 = c1 + round(2 * span / 3)
        # ensure strictly increasing & distinct
        ctx4 = sorted(set([c1, c3, c4, c2]))
        if len(ctx4) != 4:
            # fall back: pick nearest distinct integers
            cand = list(range(c1, c2 + 1))
            ctx4 = [c1] + [cand[len(cand) // 4], cand[len(cand) // 2], cand[3 * len(cand) // 4]][:2] + [c2]
            ctx4 = sorted(set(ctx4))[:4]
        out[scene] = {"context": ctx4, "target": v["target"]}
    with open(DST, "w") as f:
        json.dump(out, f)
    print(f"Wrote {DST} ({len(out)} scenes, skipped {n_skipped}).")
    # sanity: print one
    for s, v in out.items():
        if v is not None:
            print(f"  e.g. {s}: context={v['context']} target={v['target']}")
            break


if __name__ == "__main__":
    main()
