# Non-Atomizer Remaining Cleanup Checklist

Date: 2026-05-08

Scope: `bionetgen/` excluding `bionetgen/atomizer/`

Purpose: capture the still-relevant cleanup work after the recent runtime, round-trip, and dev-check fixes. This is the current non-Atomizer punch list, not a historical record of everything that has already been completed.

## Current Call

Non-Atomizer cleanup is mostly done, but not fully done.

Sections 1–5 are complete: bridge correctness/exception cleanup, action parsing/validation, model-fidelity pass, shared block-setter dedup, and TODO/FIXME sweep have all landed.

The highest-priority remaining task is now Section 6 — a low-noise hygiene pass on legacy test code (broad catches, noisy prints, environment-sensitive skip reasons).

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

- [ ] Clean up any remaining broad catches or noisy prints in the owned non-Atomizer tests that we still touch.
- [ ] Keep environment-sensitive tests explicit about prerequisites and skip reasons.

### 7. Re-run owned validation and refresh the dev notes

- [ ] Re-run the owned non-Atomizer validation path after the remaining bridge/modelapi work is done.
- [ ] Re-run the opt-in subprocess model sweeps after bridge changes to ensure they still pass cleanly.
- [ ] Retire or update stale `dev/` cleanup notes so the remaining queue is accurate.

## Immediate Next Session

Work Section 6 — low-noise hygiene pass on legacy test code.

Concrete goal: tighten the owned non-Atomizer test files so they fail loudly, skip explicitly, and don't paper over real failures with broad excepts or noisy prints.

Likely touch points:

- `tests/**` (owned non-Atomizer tests only — anything that imports `bionetgen.atomizer.*` is out of scope)
- skip-reason strings + `pytest.importorskip` / `pytest.mark.skipif` decorators on environment-sensitive tests

Definition of done for that session:

- no remaining broad `except Exception:` or bare `except:` in owned tests unless explicitly justified by a comment
- no `print()` calls in owned tests for diagnostic purposes — convert to assertions, fixtures, or dropped entirely
- environment-sensitive tests state their prerequisite explicitly in the skip reason (e.g. "requires BNG2.pl on PATH", "requires bngsim editable install")
- the full owned validation path passes
- the change is committed atomically

## Out Of Scope

- Any `bionetgen/atomizer/**` cleanup or modernization
- `uv.lock`
