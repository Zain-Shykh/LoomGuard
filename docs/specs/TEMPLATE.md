# Phase N: Title

<!--
Copy this file to docs/specs/phase-<N>-<slug>.md and fill in every section.
Write "None" where a section does not apply; do not delete sections.
The spec is written by the phase's developer session before it writes any code,
from the phase entry in docs/IMPLEMENTATION_PLAN.md and the design documents.
It must keep every acceptance criterion of the plan. See docs/WORKFLOW.md.
-->

| | |
|---|---|
| Phase | N |
| Status | Written / Code complete / Runs complete / Done |
| Spec written | YYYY-MM-DD |
| Written by | developer session, phase N |
| Depends on | Phases that must be Done first |
| Plan entry | `docs/IMPLEMENTATION_PLAN.md`, Phase N |

## 1. Objective

One or two sentences: what exists at the end of this phase that did not exist before.

## 2. Background

Why this phase is needed and what it builds on. Link the exact sections the developer must read:

- `docs/ARCHITECTURE.md` section ...
- `docs/EVALUATION.md` section ...
- Decisions that apply: D-xx, D-yy

## 3. Scope

**In scope**

- ...

**Out of scope** (belongs to a later phase or is not wanted)

- ...

## 4. Design

What to build, precisely enough that two developers would produce compatible code.

### 4.1 Files

| File | New or changed | Purpose |
|---|---|---|
| `loomguard/...` | new | |

### 4.2 Interfaces

Public functions and classes with signatures, argument and return types, tensor shapes and dtypes.

```python
def example(x: Tensor) -> Tensor:
    """x: B x 3 x 256 x 256 float32. Returns B x 1 x 256 x 256 float32."""
```

### 4.3 Behaviour and edge cases

Rules the code must follow, including what happens with unusual input.

### 4.4 Configuration

New config keys with default values and meaning.

### 4.5 Dependencies

Packages this phase needs and adds to `pyproject.toml`. The developer session cannot install packages: if one is not installed, it reports `BLOCKED`. Write "None" if there are none.

## 5. Tasks

The plan for the phase: ordered, small, each one verifiable.

### Laptop (developer session)

1. ...

### Training PC (operator)

1. ...

## 6. Acceptance criteria

All acceptance criteria of the plan's entry for this phase, plus any added here. Every criterion says how it is verified. Laptop criteria are checked by the developer session and the reviewer. Training-PC criteria are checked by the operator.

| # | Criterion | Where | Verified by | Met |
|---|---|---|---|---|
| A1 | (from plan) ... | Laptop | `pytest tests/test_x.py::test_y` | [ ] |
| A2 | | Training PC | runbook step 2, value recorded in the progress log | [ ] |

## 7. Test plan

The tests to write, each with what it proves. All run on CPU with synthetic data and no dataset.

| Test | Proves |
|---|---|
| `tests/test_x.py::test_y` | |

## 8. Training-PC runbook

Exact commands, in order, with what to record. Write "None" if the phase has no training-PC tasks.

```bash
# 1. ...
```

| Step | Expected | Record in |
|---|---|---|
| 1 | | `PROGRESS_LOG.md` |

## 9. Documents to update

Which documents this phase is expected to change, beyond the standard ones in the workflow (spec, plan, progress log, decisions).

- ...

## 10. Risks and open points

| Risk or open point | Handling |
|---|---|
| | |

## 11. Completion record

Filled in at the end of the phase. Leave empty while drafting.

| | |
|---|---|
| Spec commit | |
| Implementation commit | |
| Review commit | |
| Review rounds and verdict | |
| Tests | command and passed/failed counts |
| Results commit (training PC) | |
| Decisions made | IDs in `DECISIONS.md` |
| Deviations from this spec | what and why, or "None" |
| Closed by owner | YYYY-MM-DD |
