# Non-Atomizer Remaining Cleanup Checklist

Date: 2026-05-08

Scope: `bionetgen/` excluding `bionetgen/atomizer/`

Purpose: capture the still-relevant cleanup work after the recent runtime, round-trip, and dev-check fixes. This is the current non-Atomizer punch list, not a historical record of everything that has already been completed.

## Current Call

Non-Atomizer cleanup is **done** for everything in the scope of this checklist.

Sections 1–7 are all complete (see each section below for the landing commits and outcome notes). The default owned suite passes (`1286 passed, 2 skipped`), the opt-in subprocess model sweep passes (`2 passed` over 21 models), `mypy` is clean, and the tracked `dev/` cleanup notes either describe live work or are explicitly marked as superseded.

This document can stay in-tree as the historical record of the cleanup wave. New non-Atomizer work should start a fresh checklist rather than reopening sections here.

## Recently Completed Work

- [x] Removed tracked AppleDouble/macOS artifact handling noise.
- [x] Made test runs runtime-aware and added model-sweep timeouts.
- [x] Fixed the BNGL round-trip lossless regressions we found in the model/modelapi path.
- [x] Made dev checks provision `BNG2.pl`, prefer local `bngsim` when available, and install `bngsim` otherwise.
- [x] Forced the opt-in full model sweep onto the subprocess/`BNG2.pl` path instead of the bngsim path.
- [x] Finished the `bngsim_bridge.py` arithmetic-expression / exception-cleanup slice, including the two known bngsim-backed BNGL repros and targeted regression coverage.

## Remaining Checklist

### 1. Finish `bngsim_bridge.py` correctness and exception cleanup

- [x] Fix arithmetic-expression handling for numeric action/protocol arguments in the bngsim-backed path.
- [x] Cover both known repros:
  - `simulate({method=>"psa",t_end=>24*60*6,...})`
  - `setConcentration("TNF()",((1/52)*50000/0.04))`
- [x] Make the fix general rather than special-casing those two models.
- [x] Add regression tests that exercise action execution and protocol execution through the bridge.
- [x] Narrow or justify the remaining broad exception handling in `bngsim_bridge.py`, especially around numeric evaluation and protocol/action execution.
- [x] Document the intended error contract for unsupported or invalid numeric expressions.

### 2. Tighten action parsing and validation in `modelapi`

- [x] Centralize action-argument schema/validation so parse-time and direct-construction behavior match.
- [x] Improve validation for positional-vs-keyword actions and malformed argument shapes.
- [x] Replace stale inline TODOs with either tests, code, or tracked notes.

### 3. Do one more model fidelity pass in the core model/modelapi path

- [x] Revisit `bionetgen/modelapi/pattern.py` for equality/validation gaps that still matter.
  - Fixed `Pattern.__eq__` and `Molecule.__eq__` returning True for subset matches (no length check).
  - Fixed mutable default args (`molecules=[]`, `components=[]`, `states=[]`) that caused fresh instances to share lists.
  - Dropped a stale `# TODO: Implement __contains__` comment — `__contains__` was already implemented.
- [x] Revisit `bionetgen/modelapi/xmlparsers.py` for any remaining fidelity edge cases not already covered by the new round-trip tests.
  - No concrete fidelity bugs left after the prior round-trip cleanup; remaining TODOs (operations / rule-mods classes) are architectural feature gaps, not round-trip issues. Deferred.
- [x] Revisit `bionetgen/modelapi/bngfile.py` and `bionetgen/modelapi/blocks.py` for still-relevant contract mismatches or stale TODOs.
  - Remaining TODOs are about action stripping, the write-from-self route, recompile tagging, and deferred parameter evaluation — all feature gaps, none observably affect fidelity. Deferred.

### 4. Decide the fate of the remaining shared block-setter cleanup tail

- [x] Decide whether the remaining setter-cleanup sessions from the older punch list still buy enough value to justify the churn.
- [x] If yes, finish them in small slices with tests.
- [x] If not, retire that part of the plan explicitly so the queue stays honest.

Done: extracted `_set_item_attribute` on `ModelBlock`/`NetworkBlock` and unified all 17 subclass `__setattr__` overrides on Style A wording (ef885dd added gap-filling tests; 8de992a did the helper extraction).

### 5. Do a final non-Atomizer TODO/FIXME cleanup pass

- [x] Re-inventory non-Atomizer `TODO` / `FIXME` / `XXX` markers after the bridge work lands.
- [x] Remove or rewrite stale low-signal markers.
- [x] Move real deferred work into tracked docs/issues instead of leaving vague inline notes behind.

Done: triage in `dev/todo_triage_2026-05-08.md`. Started at 27 markers in 11 files; recalibrated to 20 CULL / 7 ISSUE → 5 distinct upstream issues. `77bd96c` stripped the 20 stale comments. Issues filed at `RuleWorld/PyBioNetGen` #70–#74. `6a2227d` anchored the remaining 7 markers to those issue refs. Zero stale TODO/FIXME/XXX markers in non-Atomizer code now.

### 6. Do a low-noise hygiene pass on legacy test code

- [x] Clean up any remaining broad catches or noisy prints in the owned non-Atomizer tests that we still touch.
- [x] Keep environment-sensitive tests explicit about prerequisites and skip reasons.

Done in `6dc3c36`: in `tests/test_bng_models.py`, the three model-sweep tests now collect failures into one assertion message instead of `print()`-ing, and each remaining `except Exception` carries a one-line comment explaining why the broad catch is intentional (collect every failure in one pass). In `tests/test_bngsim_bridge.py`, the `BNGSIM_AVAILABLE` `skipif` reason and the `test.net` / `BNGSIM_HAS_NFSIM` `pytest.skip` strings now name the actual prerequisite (bngsim importable, the test.net fixture and how it's produced, NFsim-enabled bngsim build). No test logic changed; full owned suite still passes (1286 passed, 2 skipped) and `mypy` is clean.

### 7. Re-run owned validation and refresh the dev notes

- [x] Re-run the owned non-Atomizer validation path after the remaining bridge/modelapi work is done.
- [x] Re-run the opt-in subprocess model sweeps after bridge changes to ensure they still pass cleanly.
- [x] Retire or update stale `dev/` cleanup notes so the remaining queue is accurate.

Validation re-run (2026-05-09, post-Section-6):

- Default owned suite: `1286 passed, 2 skipped` in ~48s under the dev-checks invocation (`PYBNG_DEV_BNGSIM_PATH=~/Code/PyBNF-Private/bngsim uv run --no-project --with-requirements requirements-dev.txt --with-editable ~/Code/PyBNF-Private/bngsim --with lxml --with networkx python -m pytest tests/`). The two skips are the `test.net` integration tests in `test_bngsim_bridge.py` whose fixture isn't generated when those tests run before `test_bng_core` produces it — the new skip reasons (commit `6dc3c36`) name the prerequisite explicitly.
- Opt-in subprocess sweep: `2 passed` in ~5m05s (`BNG_RUN_MODEL_SWEEPS=1 BNG_MODEL_SWEEP_TIMEOUT=300 ... python -m pytest tests/test_bng_models.py::test_model_running_CLI tests/test_bng_models.py::test_model_running_lib -v`). Both `test_model_running_CLI` and `test_model_running_lib` cleared the full 21-model corpus.
- `mypy bionetgen tests`: clean (74 source files).
- The `test_nf_parameter_scan_re_evaluates_seed_species` failure called out in the prior session note (bngsim stale-`.so` issue, `wshlavacek/PyBNF-Private#23`) did not reproduce in this run.

Stale tracked `dev/` notes refreshed:

- `dev/block_setattr_refactor_plan_2026-05-08.md` — banner added marking the plan as completed; the Section 4 work landed in `ef885dd` + `8de992a`.
- `dev/reviewer_concerns_session_handoff_2026-05-05_actionblock.md` — banner added marking the handoff as superseded by this checklist; the primary plan it referenced (`reviewer_concerns_remediation_plan_2026-05-04.md`) is no longer in-tree.
- `dev/reviewer_concerns_session_handoff_2026-05-05_blocks.md` — same banner, same reasoning.

Untracked notes under `dev/` (the directory is `.gitignore`d) are personal scratch and were left alone; the queue itself is accurate from the checklist now.

## Immediate Next Session

Nothing is open against this checklist. Sections 1–7 all landed; see the per-section notes above for landing commits and outcomes. If the next session wants to push the non-Atomizer surface further (e.g. picking up the deferred items in `dev/code-quality-punchlist-2026-05-05.md` like shrinking the long bridge functions, or tightening `mypy` strictness module-by-module), start a fresh checklist rather than reopening sections here.

## Out Of Scope

- Any `bionetgen/atomizer/**` cleanup or modernization
- `uv.lock`
