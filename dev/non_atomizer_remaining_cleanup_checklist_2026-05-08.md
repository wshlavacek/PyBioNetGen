# Non-Atomizer Remaining Cleanup Checklist

Date: 2026-05-08

Scope: `bionetgen/` excluding `bionetgen/atomizer/`

Purpose: capture the still-relevant cleanup work after the recent runtime, round-trip, and dev-check fixes. This is the current non-Atomizer punch list, not a historical record of everything that has already been completed.

## Current Call

Non-Atomizer cleanup is mostly done, but not fully done.

The highest-priority remaining task is in `bionetgen/core/tools/bngsim_bridge.py`. That task includes the `t_end=>24*60*6` defect and the broader arithmetic-expression handling gap in the bngsim-backed BNGL action/protocol path.

## Recently Completed Work

- [x] Removed tracked AppleDouble/macOS artifact handling noise.
- [x] Made test runs runtime-aware and added model-sweep timeouts.
- [x] Fixed the BNGL round-trip lossless regressions we found in the model/modelapi path.
- [x] Made dev checks provision `BNG2.pl`, prefer local `bngsim` when available, and install `bngsim` otherwise.
- [x] Forced the opt-in full model sweep onto the subprocess/`BNG2.pl` path instead of the bngsim path.

## Remaining Checklist

### 1. Finish `bngsim_bridge.py` correctness and exception cleanup

- [ ] Fix arithmetic-expression handling for numeric action/protocol arguments in the bngsim-backed path.
- [ ] Cover both known repros:
  - `simulate({method=>"psa",t_end=>24*60*6,...})`
  - `setConcentration("TNF()",((1/52)*50000/0.04))`
- [ ] Make the fix general rather than special-casing those two models.
- [ ] Add regression tests that exercise action execution and protocol execution through the bridge.
- [ ] Narrow or justify the remaining broad exception handling in `bngsim_bridge.py`, especially around numeric evaluation and protocol/action execution.
- [ ] Document the intended error contract for unsupported or invalid numeric expressions.

### 2. Tighten action parsing and validation in `modelapi`

- [ ] Centralize action-argument schema/validation so parse-time and direct-construction behavior match.
- [ ] Improve validation for positional-vs-keyword actions and malformed argument shapes.
- [ ] Replace stale inline TODOs with either tests, code, or tracked notes.

### 3. Do one more model fidelity pass in the core model/modelapi path

- [ ] Revisit `bionetgen/modelapi/pattern.py` for equality/validation gaps that still matter.
- [ ] Revisit `bionetgen/modelapi/xmlparsers.py` for any remaining fidelity edge cases not already covered by the new round-trip tests.
- [ ] Revisit `bionetgen/modelapi/bngfile.py` and `bionetgen/modelapi/blocks.py` for still-relevant contract mismatches or stale TODOs.

### 4. Decide the fate of the remaining shared block-setter cleanup tail

- [ ] Decide whether the remaining setter-cleanup sessions from the older punch list still buy enough value to justify the churn.
- [ ] If yes, finish them in small slices with tests.
- [ ] If not, retire that part of the plan explicitly so the queue stays honest.

### 5. Do a final non-Atomizer TODO/FIXME cleanup pass

- [ ] Re-inventory non-Atomizer `TODO` / `FIXME` / `XXX` markers after the bridge work lands.
- [ ] Remove or rewrite stale low-signal markers.
- [ ] Move real deferred work into tracked docs/issues instead of leaving vague inline notes behind.

### 6. Do a low-noise hygiene pass on legacy test code

- [ ] Clean up any remaining broad catches or noisy prints in the owned non-Atomizer tests that we still touch.
- [ ] Keep environment-sensitive tests explicit about prerequisites and skip reasons.

### 7. Re-run owned validation and refresh the dev notes

- [ ] Re-run the owned non-Atomizer validation path after the remaining bridge/modelapi work is done.
- [ ] Re-run the opt-in subprocess model sweeps after bridge changes to ensure they still pass cleanly.
- [ ] Retire or update stale `dev/` cleanup notes so the remaining queue is accurate.

## Immediate Next Session

Work the bridge bug first.

Concrete goal: make the bngsim-backed BNGL action/protocol path handle arithmetic expressions losslessly enough for numeric arguments, starting with the `t_end=>24*60*6` failure and the protocol `setConcentration(...)` expression case.

Likely touch points:

- `bionetgen/core/tools/bngsim_bridge.py`
- targeted bridge/model tests under `tests/`

Definition of done for that session:

- reproduction is captured by tests
- the fix handles general arithmetic expressions, not just the known literals
- targeted tests pass through `scripts/run_dev_checks.py`
- the change is committed atomically

## Out Of Scope

- Any `bionetgen/atomizer/**` cleanup or modernization
- `uv.lock`
