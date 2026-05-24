#!/usr/bin/env python3
"""Bin screened candidate models into runtime tiers + emit per-tier symlink trees.

Reads a parity_sweep.py _summary.json (the timing screen, run with --n-seeds 1)
and the candidate_corpus _manifest.json, then partitions models by *projected
DIFF cost*:

  deterministic : projected = measured wall_seconds
  stochastic    : projected = wall_seconds * ensemble_seeds   (the real DIFF
                  runs N seeds per side, so single-seed time underestimates)

Tiers (by projected cost):
  t1_5s    < 5s        fast — first DIFF batch, regular-suite candidates
  t2_10s   5–10s
  t3_2min  10–120s
  slow     >= 120s, OR hit the screen's timeout cap (cost unknown — rescreen
           at a higher cap to separate <2min from genuinely-slow)
  broken   reference stack crashed/errored — cannot be DIFF'd; triage separately

For every tier except 'broken' we emit a mirror symlink subtree
<dest>/<tier>/<source>/<rel> so a tiered DIFF is just:
    parity_sweep.py --root <dest>/t1_5s --simulator subprocess ...
    parity_sweep.py --root <dest>/t1_5s --simulator bngsim ...
    parity_diff.py  <subprocess_out> <bngsim_out>
"""
import argparse
import json
import os
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, action="append",
                    help="screen _summary.json (n-seeds 1). Repeat for multiple "
                         "stacks (e.g. subprocess + bngsim); each model is tiered "
                         "on the SLOWER stack, and counted broken if EITHER stack "
                         "crashed/errored (can't DIFF a model one side can't run).")
    ap.add_argument("--manifest", required=True, help="candidate_corpus _manifest.json")
    ap.add_argument("--dest", required=True, help="output dir for tier symlink trees")
    ap.add_argument("--ensemble-seeds", type=int, default=10,
                    help="seeds the real DIFF will use for stochastic models")
    args = ap.parse_args()

    # manifest links are relative; parity_sweep emits absolute paths -> normalize
    manifest = {os.path.abspath(m["link"]): m
                for m in json.loads(Path(args.manifest).read_text())}

    # collect per-link per-stack results across all summaries
    stacks = {}  # name -> {link: result}
    cap = None
    for spath in args.summary:
        s = json.loads(Path(spath).read_text())
        name = s.get("simulator", Path(spath).parent.name)
        stacks[name] = {r["bngl"]: r for r in s["results"]}
        for r in s["results"]:
            if r["status"] == "timeout":
                cap = max(cap or 0, r["wall_seconds"])

    all_links = sorted({lk for st in stacks.values() for lk in st})
    rows = []
    for link in all_links:
        man = manifest.get(os.path.abspath(link), {})
        stochastic = man.get("stochastic", False)
        has_free = man.get("has_free", False)
        per_stack = {n: st.get(link) for n, st in stacks.items()}
        statuses = {n: (r["status"] if r else "missing") for n, r in per_stack.items()}
        # broken if ANY stack couldn't run it (can't DIFF a one-sided model)
        if any(s in ("crash", "error", "missing") for s in statuses.values()):
            # a __FREE model that crashes is an UNFILLED PyBNF fitting template
            # (references an undefined free-param symbol); set it aside rather
            # than counting it a real failure. Filled __FREE models run and tier
            # normally, so they never reach here.
            tier = "fitting_template" if has_free else "broken"
            projected, wall = float("nan"), float("nan")
        elif any(s == "timeout" for s in statuses.values()):
            tier, projected = "glacial", float("inf")
            wall = max(r["wall_seconds"] for r in per_stack.values())
        else:  # ok on every stack -> tier on the SLOWER stack
            wall = max(r["wall_seconds"] for r in per_stack.values())
            projected = wall * args.ensemble_seeds if stochastic else wall
            if projected < 10:
                tier = "fast"        # <10s  (regular validation set)
            elif projected < 120:
                tier = "slow"        # 10s-2min
            else:
                tier = "glacial"     # >=2min or timed out
        rows.append({
            "link": link, "source": man.get("source", "?"),
            "methods": man.get("methods", []), "features": man.get("features", []),
            "stochastic": stochastic, "status": statuses,
            "wall": wall, "projected": projected, "tier": tier,
        })

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    tier_counts = Counter(r["tier"] for r in rows)
    src_root = None
    # candidate_corpus root = common prefix of links up to <source>/
    for r in rows:
        p = Path(r["link"])
        # link is <corpus>/<source>/<rel>; corpus = parent of <source> dir
        parts = p.parts
        if r["source"] in parts:
            i = parts.index(r["source"])
            src_root = Path(*parts[:i])
            break

    emitted = Counter()
    for r in rows:
        if r["tier"] in ("broken", "fitting_template"):
            continue
        link = Path(r["link"])
        rel = link.relative_to(src_root)        # <source>/<rel>
        out = dest / r["tier"] / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        target = link.resolve()                 # point at the real model file
        if out.is_symlink() or out.exists():
            out.unlink()
        out.symlink_to(target)
        emitted[r["tier"]] += 1

    (dest / "_tiers.json").write_text(json.dumps(rows, indent=2))

    print(f"screen cap (timeout) inferred: ~{cap}s, ensemble_seeds={args.ensemble_seeds}\n")
    order = ["fast", "slow", "glacial", "broken", "fitting_template"]
    print(f"{'tier':9s} {'count':>6s}  {'det':>5s} {'stoch':>5s}   by-source")
    for t in order:
        sub = [r for r in rows if r["tier"] == t]
        det = sum(1 for r in sub if not r["stochastic"])
        sto = len(sub) - det
        bysrc = Counter(r["source"] for r in sub)
        print(f"{t:9s} {len(sub):6d}  {det:5d} {sto:5d}   {dict(bysrc)}")
    print(f"\ntier symlink trees: {dest}/<tier>/  (emitted {dict(emitted)})")
    print(f"tier table: {dest/'_tiers.json'}")
    # surface feature-bearing models — likeliest to exercise untested paths
    feat = [r for r in rows if r["features"]
            and r["tier"] not in ("broken", "fitting_template")]
    if feat:
        print(f"\nfeature-bearing (non-broken): {len(feat)}")
        fc = Counter(f for r in feat for f in r["features"])
        print(f"  {dict(fc)}")


if __name__ == "__main__":
    main()
