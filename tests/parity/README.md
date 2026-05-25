# Parity corpus

A tracked, reproducible corpus for the BNGsim-vs-subprocess parity check. Every
model that the validation suite runs lives here, so the result no longer depends
on external model repositories or local symlink trees.

## Layout

```
tests/parity/
  manifest.json          # source of truth: one record per model
  NOTICE                 # upstream attribution + licenses for vendored models
  models/<tier>/<source>/<path>/<model>.bngl
```

- **tier** — `fast` | `slow` | `glacial` (candidate-corpus speed tiers, binned by
  projected DIFF cost) or `original` (the curated base suite). Pick a tier for a
  smaller run.
- **source** — `rulehub`, `rulemonkey`, `bngl_models`, `bngl_library`, `curated`.
- Companion files (`.net`, `.species`, `.tfun`, or anything named in a model)
  are vendored alongside the model so relative includes resolve.

## Manifest record

```json
{
  "id": "fast/rulehub/Examples/biology/aktsignaling/akt-signaling.bngl",
  "file": "models/fast/rulehub/.../akt-signaling.bngl",
  "basename": "akt-signaling.bngl",
  "source": "rulehub",
  "license": "MIT",
  "provenance": {"repo": "github.com/RuleWorld/RuleHub", "commit": "<sha>",
                 "path": "Examples/biology/aktsignaling/akt-signaling.bngl"},
  "tier": "fast",
  "regime": "deterministic",
  "expected": "PASS",
  "overrides": {"t_end": 100}     // optional; see below
}
```

`id` (the vendored relpath) is the **unique key** — basenames collide across
sources, so overrides and lookups key on `id`, not the filename.

- **expected** — the verdict bucket the suite asserts: `PASS`, `PASS_REF_BUG`
  (subprocess NFsim reference bug; bngsim validated against ODE / RuleMonkey /
  analytic), or `KNOWN_ARTIFACT` (e.g. an oscillatory/bistable SSA ensemble
  mean). The *reasons* for the non-PASS buckets live in the registries in
  `scripts/parity_diff.py`.
- **overrides** — per-model run fixtures applied identically to both backends:
  `t_end` (cap), `n_scan_pts`, `tol` (`atol`/`rtol`),
  `action_inject` (add a run action), `timeout`. `parity_sweep.py --manifest`
  reads these by `id`.

## Running

```bash
# Full suite / a tier / a hand-picked set:
python scripts/parity_validate.py --all
python scripts/parity_validate.py --tier fast
python scripts/parity_validate.py --models egfr_net,fceri_ji

# Re-bless expected buckets after an intended change:
python scripts/parity_validate.py --all --update-baseline
```

Run from the venv with the pinned **bngsim 0.9.7** wheel. The driver sweeps each
model on both backends (`parity_sweep.py`), buckets them (`parity_diff.py`),
escalates noisy stochastic DIFFs to more seeds, and asserts the result against
`manifest.json`. Exit 0 iff every model matches its expected bucket and the DIFF
/ ERROR buckets are empty.

## Regenerating the corpus

`scripts/build_parity_corpus.py` rebuilds `models/` + `manifest.json` + `NOTICE`
from the canonical per-tier parity reports (it resolves each model to its
upstream file, classifies the source, copies it in, and records provenance +
overrides). Re-run it only when adding/removing models.
