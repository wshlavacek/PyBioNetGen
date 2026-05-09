# Block `__setattr__` Refactor Plan

Date: 2026-05-08

Owner: next session, fresh context

Scope: deduplicate the 17 near-identical `__setattr__` overrides across
`bionetgen/modelapi/blocks.py` (9) and `bionetgen/network/blocks.py` (8).
Tied to Section 4 of `dev/non_atomizer_remaining_cleanup_checklist_2026-05-08.md`.

This is a *careful* slice — the duplication is annoying but the code is
functionally fine, and we have lots of tests pinning exact log wording
and side-effect timing. The refactor must be behavior-preserving by
default.

## Current state of the duplication

Both files have a base class (`ModelBlock` / `NetworkBlock`) with a
simple `__setattr__` that's irrelevant here. Every "named-item" subclass
re-implements its own override, all of them sharing this skeleton:

```python
def __setattr__(self, name, value):
    changed = False
    if hasattr(self, "items"):
        if name in self.items:
            if isinstance(value, <ItemClass>):
                changed = True
                self.items[name] = value
            elif isinstance(value, str):
                if self.items[name][<str_field>] != value:
                    changed = True
                    self.items[name][<str_field>] = value
                    # ParameterBlock only: self.items[name].write_expr = True
            else:
                # Two of the nine also have a float() coercion branch here.
                # All emit a logger.warning on unsupported types.
                ...
            if changed:
                self._changes[name] = value
                self.__dict__[name] = value
        else:
            self.__dict__[name] = value
    else:
        self.__dict__[name] = value
```

### Per-block variation matrix

modelapi/blocks.py:

| Block (line) | Item class | str-update field | float branch? | float field | write_expr toggle? | Warning style |
|---|---|---|---|---|---|---|
| ParameterBlock (203) | Parameter | `value` | yes | `value` | yes (`write_expr`) | A |
| CompartmentBlock (264) | Compartment | `name` | yes | `size` | no | A |
| ObservableBlock (319) | Observable | `name` | no | — | no | A |
| SpeciesBlock (367) | Species | `name` | no | — | no | A |
| MoleculeTypeBlock (422) | MoleculeType | `name` | no | — | no | B |
| FunctionBlock (466) | Function | `expr` | no | — | no | B |
| RuleBlock (515) | Rule | `name` | no | — | no | B |
| EnergyPatternBlock (688) | EnergyPattern | `name` | no | — | no | B |
| PopulationMapBlock (732) | PopulationMap | `name` | no | — | no | B |

network/blocks.py:

| Block (line) | Item class | str-update field | float branch? | float field | write_expr toggle? | Warning style |
|---|---|---|---|---|---|---|
| NetworkParameterBlock (162) | NetworkParameter | `value` | yes | `value` | yes (`write_expr`) | A |
| NetworkCompartmentBlock (223) | NetworkCompartment | `name` | yes | `size` | no | A |
| NetworkGroupBlock (278) | NetworkGroup | `name` | no | — | no | A |
| NetworkSpeciesBlock (326) | NetworkSpecies | `name` | no | — | no | A |
| NetworkFunctionBlock (381) | NetworkFunction | `expr` | no | — | no | B |
| NetworkReactionBlock (430) | NetworkReaction | `name` | no | — | no | B |
| NetworkEnergyPatternBlock (474) | NetworkEnergyPattern | `name` | no | — | no | B |
| NetworkPopulationMapBlock (518) | NetworkPopulationMap | `name` | no | — | no | B |

### The two warning styles

- **Style A:** `"Unable to set <kind> {!r} to {!r}; keeping existing <field> {!r}"`
  with `loc=…`. Used by Parameter, Compartment, Observable, Species (and
  network counterparts) — the four oldest blocks. Informative, three
  fields.
- **Style B:** `f"can't set <kind> {self.items[name]['name']} to {value}"`
  with `loc=…`. Used by MoleculeType, Function, Rule, EnergyPattern,
  PopulationMap (and network counterparts) — the newer blocks. Terser,
  no quoting, ad-hoc.

This split is almost certainly a copy-paste artifact from when the
newer blocks were added without consulting the older convention. The
refactor is the right time to unify on Style A.

## Existing test coverage

Tests live in `tests/test_blocks.py` and
`tests/test_network_structs_blocks.py`. Each block typically has:

- `test_setattr_with_<kind>_object` — replacing the item with a
  matching-typed object updates `items[name]` and records a `_changes`
  entry.
- `test_setattr_with_string_updates_<field>` — string assignment
  patches the right field on the item.
- `test_setattr_invalid_type_logs_warning` — non-string, non-item-class
  assignment logs a warning through `blocks_module.logger`, leaves the
  item unchanged, and leaves `_changes` empty.

Several tests assert on substrings of the warning message. Those are
the load-bearing pin sites.

### Behavior pinned by tests (do not break)

- `with patch.object(blocks_module, "logger") as mock_logger` — the
  helper must call `logger.warning(...)` on the same module-level
  `logger` symbol that the existing setters use. Routing the warning
  through a different module would break these mocks.
- `mock_logger.warning.assert_called_once()` — exactly one warning
  per failed assignment.
- `loc=` kwarg on the warning must contain the
  `<BlockClassName>.__setattr__()` substring. If the helper lives on
  the base class, we still need to thread the subclass name into
  `loc`.
- `_changes` is left empty on a failed (warning) path; no
  `__dict__[name]` write either.
- `test_setattr_propagates_unexpected_float_error`
  (`tests/test_blocks.py:344`): float coercion catches only
  `(TypeError, ValueError)`. Other exceptions (e.g., `RuntimeError`
  from a custom `__float__`) must propagate.

### Coverage gaps to close before refactoring

These blocks have no `test_setattr_invalid_type_logs_warning`; their
warning paths are currently unverified, so a refactor could silently
break them.

- `tests/test_blocks.py`:
  - `TestRuleBlock` — no warning test.
- `tests/test_network_structs_blocks.py`:
  - `TestNetworkParameterBlock` — no warning test.
  - `TestNetworkFunctionBlock` — no warning test.
  - `TestNetworkReactionBlock` — no warning test.
  - `TestNetworkEnergyPatternBlock` — no warning test.
  - `TestNetworkPopulationMapBlock` — no warning test.

Closing these is **the first slice**. They're cheap, isolated, and
become the safety net the refactor relies on.

## Proposed design

### Helper signature

Place a class method on each base class
(`bionetgen.modelapi.blocks.ModelBlock` and
`bionetgen.network.blocks.NetworkBlock`) — keep them separate so we
don't introduce a cross-package dependency for one helper:

```python
def _set_item_attribute(
    self,
    name,
    value,
    *,
    item_cls,
    str_field,
    num_field=None,
    write_expr_field=None,
    kind,
):
    """Shared setattr path for blocks that hold named items.

    name, value: as passed to __setattr__.
    item_cls: the per-block item type (Parameter / Rule / etc.).
    str_field: dict key used by the str-update branch.
    num_field: dict key used by the float branch (None = no float branch).
    write_expr_field: name of an item attribute toggled True on str update
        and False on numeric update (None = no toggle).
    kind: human-readable label for the warning ("parameter", "rule", ...).
    """
```

Each subclass `__setattr__` becomes a one-liner:

```python
def __setattr__(self, name, value):
    self._set_item_attribute(
        name, value,
        item_cls=Parameter,
        str_field="value",
        num_field="value",
        write_expr_field="write_expr",
        kind="parameter",
    )
```

The helper uses `self.__class__.__name__` for the warning `loc` so the
test pin (`<BlockClass>.__setattr__()` substring) keeps working —
**alternatively**, take a `loc_class` kwarg so the substring stays
stable regardless of where the call lands.

### Style unification

In the same slice, unify on **Style A** for all 17 blocks:

```
"Unable to set {kind} {item_name!r} to {value!r}; keeping existing {kind}"
```

Tests that pin substring `"can't set <kind>"` (5 modelapi + 4 network
sites) must be updated to assert on the Style A wording. This is a
small, mechanical change but it is part of the refactor because Style
B is what motivates a fair chunk of the duplication (different format
= different code shape).

If we keep both styles, the helper grows a `warning_style` flag that
buys us nothing — better to fix it now.

### What stays as-is

- `ModelBlock.__setattr__` and `NetworkBlock.__setattr__` (the bases)
  — they don't follow this pattern; they handle a simpler "if name is
  in items, try `float()`" path that isn't a duplicate.
- `ActionBlock.__setattr__` (modelapi/blocks.py:598) — single-liner
  `self.__dict__[name] = value`; not part of this group.
- The `add_<kind>` convenience constructors. They're independently
  shaped and out of scope.

## Slice ordering

### Slice 1 — close test gaps (small, atomic)

Add the six missing `test_setattr_invalid_type_logs_warning` tests
identified above. Match the patch-and-assert shape of the existing
tests. After this slice, all 17 setattr warning paths have explicit
test coverage and the refactor has a safety net.

Verification: `pytest tests/test_blocks.py tests/test_network_structs_blocks.py`
passes; new tests fail without the existing setters' logger.warning
calls (sanity check that they're load-bearing).

### Slice 2 — extract the helper, route all 17 callers (medium)

1. Add `_set_item_attribute` as a method on `ModelBlock` and a mirror
   on `NetworkBlock`.
2. Replace all 17 subclass `__setattr__` bodies with a delegating call.
3. Unify the warning wording on Style A; update test substring asserts
   to match.
4. Confirm `test_setattr_propagates_unexpected_float_error` still
   passes (RuntimeError must escape; only TypeError/ValueError trigger
   the warning).
5. Run the full suite + mypy.

Estimated diff: ~150 lines deleted, ~60 added (net negative).

### Slice 3 (optional) — ergonomic cleanup

If the helper is in good shape, look at:

- Whether `_changes[name] = value` should be `self._changes[name] = item`
  for the str-update path (currently writes the raw assigned string,
  which loses the existing object reference). Behavior preserved
  for now; flag as a potential future bug only after Slice 2 lands.
- Whether the `hasattr(self, "items")` guard is still needed. It was
  there for partial-init scenarios; with the base class always
  initializing `self.items` in `__init__`, this may be dead. Don't
  touch in this refactor — orthogonal.

Probably skip Slice 3 unless we find something live during Slice 2.

## Risk register

- **Two mirrored files.** Do them in the same slice so they don't drift
  again. Diff each side carefully — the helper signature, defaults,
  and tests should be identical structure.
- **Pinned warning substrings.** Several tests assert on specific
  wording. Audit every `assert ".*"` in the warning tests before
  changing the message format. The grep is already in
  `tests/test_blocks.py:[326-360, 397-415, 471-489, 522-535, 564-577,
  788-799, 837-849]` and the network mirrors.
- **`loc=` substring.** Tests check for `<BlockClass>.__setattr__()`.
  If the helper writes `loc=f"... : {self.__class__.__name__}.__setattr__()"`
  this stays correct; verify with each subclass's test.
- **Mock target.** Tests do `patch.object(blocks_module, "logger")`.
  The helper must call `logger.warning(...)` via the module-level
  `logger` symbol in *each* blocks.py file (i.e., the helper sits in
  the same module as the subclasses, which it does — both are in
  modelapi/blocks.py / network/blocks.py respectively).
- **Numeric error propagation.** `test_setattr_propagates_unexpected_float_error`
  pins that non-`(TypeError, ValueError)` exceptions escape. The
  helper must catch only those two; do not widen to `Exception`.
- **`write_expr` semantics.** Only ParameterBlock toggles this. The
  helper must skip the toggle when `write_expr_field is None`.
- **`changed` and `_changes` recording for the float path.** When
  numeric coercion succeeds, the original code writes
  `self._changes[name] = value` (where `value` was rebound to
  `new_value` mid-branch). Preserve that — easy to drop when
  refactoring; keep an explicit test if not already covered.
- **No atomizer touch.** Confirmed: `bionetgen/atomizer/` does not
  import either blocks module. Out of scope.
- **uv build cache.** Per `feedback_investigate_unrelated_failures`,
  if `test_nf_parameter_scan_re_evaluates_seed_species` fails after
  the refactor, suspect the stale-`.so` issue (bngsim
  github.com/wshlavacek/PyBNF-Private#23) before suspecting the
  refactor.

## Dev-check commands

Targeted (Slice 1):

```bash
PYBNG_DEV_BNGSIM_PATH=~/Code/PyBNF-Private/bngsim \
  uv run --no-project --with-requirements requirements-dev.txt \
  --with-editable ~/Code/PyBNF-Private/bngsim \
  --with lxml --with networkx \
  python -m pytest tests/test_blocks.py tests/test_network_structs_blocks.py
```

Full suite + mypy (Slice 2):

```bash
PYBNG_DEV_BNGSIM_PATH=~/Code/PyBNF-Private/bngsim \
  uv run --no-project --with-requirements requirements-dev.txt \
  --with-editable ~/Code/PyBNF-Private/bngsim \
  --with lxml --with networkx \
  python -m pytest tests/

PYBNG_DEV_BNGSIM_PATH=~/Code/PyBNF-Private/bngsim \
  uv run --no-project --with-requirements requirements-dev.txt \
  --with-editable ~/Code/PyBNF-Private/bngsim \
  --with lxml --with networkx \
  python -m mypy bionetgen tests
```

`test_nf_parameter_scan_re_evaluates_seed_species` is expected to fail
on this machine due to the bngsim stale-`.so` issue
(github.com/wshlavacek/PyBNF-Private#23); this failure is unrelated
to this refactor.

## Definition of done

- All 17 `__setattr__` bodies are one-line delegations.
- Warning wording is Style A everywhere.
- Existing tests still pass; the six new gap-filling tests pass too.
- `mypy bionetgen tests` clean.
- Two commits land:
  1. "Add missing setattr warning tests for block subclasses"
  2. "Deduplicate block __setattr__ via shared helper"
- Section 4 of the punchlist is ticked with a one-line note.

## Out of scope

- Touching `ActionBlock`, `ProtocolBlock`, base `ModelBlock` /
  `NetworkBlock` setattr.
- `add_item` / `add_action` / `__delitem__` cleanup.
- Any atomizer code.
- The TODOs at `blocks.py:46` (`_recompile` from `_changes`) and
  `blocks.py:155` (parameter evaluation) — separate concern.
