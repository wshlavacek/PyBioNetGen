# Parity notes: legacy subprocess stack vs. BNGsim

This document records observed differences in outcome between the two simulation
backends PyBioNetGen can drive:

- **legacy / subprocess** — `BNG2.pl` + `run_network` (CVODE) + the vendored
  `NFsim` binary (`bionetgen/bng-mac/bin/`). The historical, trusted reference.
- **BNGsim** — the in-process `bngsim` engine (ODE/SSA/PLA + its own NFsim build
  + RuleMonkey), driven via `simulator="bngsim"`.

It was compiled while expanding the parity test suite with new models from
RuleHub, RuleMonkey, and the BNGL library (~600 new simulation models beyond the
core 377). Both backends share the same `BNG2.pl` front-end to generate the
network/XML; the differences below are in the **solver/engine that consumes it**.

> **Caveat that applies throughout:** where the legacy stack produces *no* output
> (it crashes or refuses), there is no trusted reference, so BNGsim's result for
> that model **cannot be parity-validated**. "BNGsim ran it and the output looks
> structured/plausible" is the most the legacy oracle can tell us; confirming
> correctness there needs an independent reference (RuleMonkey standalone, the
> source publication, or an analytical check).

## Summary

On the new corpus, every observed legacy-vs-BNGsim divergence falls into one of:

1. **BNGsim is more correct** than legacy (legacy silently produces wrong output).
2. **BNGsim is more capable** than legacy (legacy's engine hits a hard limit).
3. **BNGsim bugs** we found and filed.
4. **Not a backend difference** (upstream-broken models, a wrapper bug, slow models).

We found **no** case where BNGsim silently diverged on a model the legacy stack
handles correctly.

## 1. BNGsim is more correct (legacy silently wrong)

**`include_products()` / `exclude_products()` on reversible rules under NFsim.**
Models: `ft_exclude_products`, `ft_exclude_reactants`, `ft_include_reactants`,
`combo_exclude_with_complex` (RuleMonkey feature-coverage tests).

Standard NFsim does **not enforce** `include_products`/`exclude_products`
constraints (notably on the auto-generated reverse rule). BNGsim's NFsim **aborts
loudly** rather than risk silently-incorrect results:

```
ReactionRule _reverse__R2 uses include_products()/exclude_products(),
which are not yet enforced in NFsim. Aborting to avoid silently incorrect results.
```

The legacy stack **runs these and emits numbers** — which are therefore suspect.

> **Oracle warning:** our legacy NFsim reference is silently unreliable for any
> model using `include_products`/`exclude_products`. Do not trust legacy output
> for such models.

## 2. BNGsim is more capable (legacy engine limits)

These run on BNGsim but the legacy engine refuses or fails. The cause is **engine
version**, not a RuleMonkey fallback — confirmed these dispatch as `method: nf`
through BNGsim's own (newer) NFsim, or as ODE through BNGsim's integrator.

| Model | Legacy outcome | Why BNGsim handles it |
|---|---|---|
| `AN`, `ANx` | NFsim aborts: *"invalid state 'PLUS'/'MINUS'"* | BNGsim's NFsim supports the relative state-change operator `m~PLUS`/`m~MINUS` (increment/decrement); the old vendored NFsim binary does not |
| `rm_tlbr_rings` | NFsim aborts: *"bond to sites already occupied"* | BNGsim's NFsim handles **ring closure** (intramolecular bonds); legacy NFsim cannot |
| `Dushek_2011` | CVODE convergence failure (error −4, stiff) at t≈28 | BNGsim's ODE integrator completes the stiff trajectory where legacy CVODE gives up |

(No legacy oracle for any of these — see the caveat above.)

## 3. BNGsim bugs found and filed

Tracked at `wshlavacek/PyBNF-Private`:

- **#61** — BNGsim 0.9.2 leaks internal `__bngsim_net_rewrite_obs_*` observables
  into `.gdat` on functional/`Sat()` rate-law models (e.g. `test_sat`,
  `CaOscillate_Sat`). `.cdat` and user observable columns are correct.
- **#62** — BNGsim ODE integrator goes unstable (negative/exploding values) where
  legacy CVODE is stable (e.g. `eco_coevolution_host_parasite`:
  `H_Lo = -1.37e8` vs legacy `2.01`; `predator-prey-dynamics`).
- **#63** — BNGsim's NFsim `initialize()` failures surface only the bare string
  `"Quitting"`, swallowing NFsim's actual diagnostic. Blocks diagnosis of
  functional-rate synthesis/decay models under `nf` (`V1988a_endemic_infection`,
  `V1990_cooke_endemic`, `V1990_kemper_endemic`), which legacy runs cleanly.

## 4. Not a backend difference

- **Upstream-broken (~68 models):** fail on **both** stacks — malformed BNGL,
  missing companion files, etc. These are RuleHub/source problems, not ours or
  BNGsim's. Excluded from the parity corpus as `upstream_broken`.
- **PyBNF/BioNetFit fitting templates (~470):** parameters end in `__FREE` and are
  unfilled (substituted from a `.conf` before running). `BNG2.pl` correctly
  rejects them; they are not standalone-runnable. Excluded as `fitting_template`.
  (Note: *filled* best-fit models — `__FREE` as a defined parameter *name* with a
  real value, e.g. `egg`, `antigen_pulses_harmon2017` — run fine and are kept.)
- **Headerless continuation `.gdat` (legacy quirk; reader fixed, *not* mirrored
  in bngsim):** `simulate(..., continue=>1)` passes `-x` to `run_network`
  (`BNGAction.pm:451`), which intentionally omits the header line and the
  duplicate initial point. This is *correct* for the intended use — continuing
  into the **same** output file (no suffix), giving one trajectory with one header
  at the top. The wart appears only when `continue=>1` is combined with
  `suffix=>"N"`, which redirects each segment to a *separate* file: `-x` still
  fires, so the split-out segments (`_2.gdat`, `_3.gdat`) come out headerless and
  standalone (seen in `calcium-spike-signaling`). bngsim does **not** reproduce
  this — it writes loadable output, which is why such models land in
  "legacy-fails-bngsim-succeeds". The fix is purely on the *reading* side:
  `BNGResult._load_dat` now recovers column names from the base-suffix sibling
  (which has the header) so legacy's output can be loaded and compared; genuinely
  orphaned headerless files (no sibling header) still raise.
- **Slow models / false timeouts:** several models (`Model_ZAP`,
  `blbr_rings_posner1995`, `Zhang_2021`, `e7`, `tcr`) were flagged broken only
  because a contended screening run exceeded a 60s cap; they complete in 14–28s on
  an idle machine and belong in the `slow` tier.
