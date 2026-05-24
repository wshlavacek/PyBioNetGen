#!/usr/bin/env python3
"""Stage every NEW (never-DIFF-tested) BNGL model into a symlink tree.

"New" = content-unique (whitespace/comment-normalized md5) against BOTH:
  * the 377-model parity suite (read from a parity report's per_model keys), and
  * everything already selected from earlier sources in this run (cross- and
    intra-source dedup).

We SYMLINK rather than copy: RuleHub / RuleMonkey carry their own licenses, so
vendoring + committing their models is a licensing problem and bloats git. The
symlink tree preserves provenance (candidate_corpus/<source>/<relpath>) and
parity_sweep.py's rglob("*.bngl") walks it fine. Companion (non-.bngl) files
that sit beside a selected model are symlinked too, so relative-path
includes / NFsim data / scan inputs still resolve.

Output:
  <dest>/<source>/<relpath-under-source-root>/<model>.bngl  (symlinks)
  <dest>/_manifest.json   (one record per selected model)
  <dest>/_manifest.csv

The tree itself should be gitignored; commit only this script + the manifest.
"""
import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path

HOME = Path.home()

SOURCES = {
    "rulehub": HOME / "Code" / "RuleHub",
    "rulemonkey": HOME / "Code" / "RuleMonkey",
    "bngl_lib": HOME / "Code" / "BNGL_library" / "bngl_models",
}

DET_METHODS = {"ode", "cvode"}


def read(p):
    try:
        return p.read_text(errors="replace")
    except Exception:
        return None


def norm_hash(text):
    """Content identity ignoring comments + surrounding whitespace."""
    lines = [re.sub(r"#.*$", "", ln).strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return hashlib.md5("\n".join(lines).encode()).hexdigest()


def detect_methods(text):
    t = re.sub(r"#.*", "", text)
    methods = set()
    for blob in re.findall(r"simulate\s*\(\s*\{([^}]*)\}", t, re.DOTALL):
        m = re.search(r"method\s*=>\s*['\"]?(\w+)['\"]?", blob)
        methods.add((m.group(1) if m else "ode").lower())
    for meth, _ in re.findall(r"simulate_(\w+)\s*\(\s*\{([^}]*)\}", t, re.DOTALL):
        methods.add(meth.lower())
    for blob in re.findall(r"(?:parameter_scan|bifurcate)\s*\(\s*\{([^}]*)\}", t, re.DOTALL):
        m = re.search(r"method\s*=>\s*['\"]?(\w+)['\"]?", blob)
        methods.add((m.group(1) if m else "ode").lower())
    return methods


def features(text):
    tl = text.lower()
    f = []
    if re.search(r"begin\s+compartments", tl):
        f.append("compartments")
    if re.search(r"begin\s+energy", tl):
        f.append("energy")
    if "populationmap" in tl:
        f.append("popmap")
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True,
                    help="parity report json whose per_model keys are the 377 suite")
    ap.add_argument("--dest", required=True, help="candidate_corpus output dir")
    ap.add_argument("--exclude-list", action="append", default=[],
                    help="file of content hashes to skip (one norm_hash per line, "
                         "'#'-comments and trailing '  # path' allowed). Repeatable. "
                         "Evicts fitting_template (unfilled PyBNF) models and "
                         "clear-dupe redundant copies — by CONTENT, so duplicate "
                         "copies at other paths are caught too.")
    args = ap.parse_args()

    excluded_hashes = set()
    for ef in args.exclude_list:
        for ln in Path(ef).read_text().splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                excluded_hashes.add(ln.split()[0])

    rep = json.loads(Path(args.report).read_text())
    corpus_hashes = set()
    for k in rep["per_model"]:
        t = read(Path(k))
        if t:
            corpus_hashes.add(norm_hash(t))
    print(f"corpus: {len(rep['per_model'])} models, {len(corpus_hashes)} content hashes")

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    seen = set(corpus_hashes)        # global: corpus + everything selected so far
    companion_dirs = set()           # source dirs whose non-bngl files we've linked
    manifest = []

    for sname, root in SOURCES.items():
        if not root.exists():
            print(f"!! {sname}: {root} MISSING, skipping")
            continue
        files = sorted(root.rglob("*.bngl"))
        n_new = n_dupcorpus = n_dupseen = n_bad = 0
        for src in files:
            t = read(src)
            if t is None:
                n_bad += 1
                continue
            h = norm_hash(t)
            if h in excluded_hashes:
                continue
            if h in corpus_hashes:
                n_dupcorpus += 1
                continue
            if h in seen:
                n_dupseen += 1
                continue
            seen.add(h)
            n_new += 1

            rel = src.relative_to(root)
            link = dest / sname / rel
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(src)

            # link sibling non-.bngl companions once per source dir
            if src.parent not in companion_dirs:
                companion_dirs.add(src.parent)
                for sib in src.parent.iterdir():
                    if sib.is_file() and sib.suffix.lower() != ".bngl":
                        clink = link.parent / sib.name
                        if not (clink.is_symlink() or clink.exists()):
                            clink.symlink_to(sib)

            methods = detect_methods(t)
            manifest.append({
                "source": sname,
                "rel": str(rel),
                "src": str(src),
                "link": str(link),
                "methods": sorted(methods),
                "stochastic": bool(methods - DET_METHODS) or not methods,
                "features": features(t),
                # PyBNF/BioNetFit free-parameter marker. Two flavors that only
                # BNG2.pl can tell apart: UNFILLED templates reference an
                # undefined __FREE symbol (crash) vs FILLED best-fit models that
                # define __FREE-named params with real values (run fine). The
                # binner routes crashed-__FREE models to a 'fitting_template'
                # bucket and keeps the runnable ones.
                "has_free": "__FREE" in t,
                "src_bytes": len(t),
            })
        print(f"{sname:11s}: {len(files):4d} files | NEW {n_new:4d} | "
              f"dup-corpus {n_dupcorpus:3d} | dup-seen {n_dupseen:3d} | bad {n_bad}")

    (dest / "_manifest.json").write_text(json.dumps(manifest, indent=2))
    with open(dest / "_manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "rel", "methods", "stochastic", "features", "src_bytes"])
        for m in manifest:
            w.writerow([m["source"], m["rel"], "|".join(m["methods"]),
                        m["stochastic"], "|".join(m["features"]), m["src_bytes"]])

    n_stoch = sum(1 for m in manifest if m["stochastic"])
    print(f"\nTOTAL selected: {len(manifest)}  "
          f"(deterministic {len(manifest)-n_stoch}, stochastic {n_stoch})")
    print(f"manifest: {dest/'_manifest.json'}")


if __name__ == "__main__":
    main()
