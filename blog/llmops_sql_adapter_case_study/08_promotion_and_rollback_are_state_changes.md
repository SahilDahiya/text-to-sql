# Promotion and Rollback Are State Changes

Promotion is not copying a model folder.

Promotion is a controlled state change:

```text
this adapter is now the current candidate for this database/environment
```

Rollback is the reverse:

```text
move current back to the previous known-good adapter
```

That state needs to be explicit and auditable.

## Promote, Reject, Investigate

The run contract decides from evidence:

- train summary
- offline eval gates
- endpoint eval gates
- load-test gates
- failure counts
- required thresholds

The decision values are:

- `promote`
- `reject`
- `investigate`

Plain meaning:

- `promote`: required evidence exists and gates passed
- `reject`: required evidence exists and at least one gate failed
- `investigate`: required evidence is missing

That distinction matters.

Missing endpoint eval should not look like a failed endpoint eval. It means the run is incomplete.

## Why Exp056 Beat Exp062

Exp056:

- dev_v2: 11/12
- eval_v1: 12/12
- challenge_v1: 22/24

Exp062:

- dev_v2: 12/12
- eval_v1: 11/12
- challenge_v1: 23/24
- challenge_v2: 12/15

Exp062 improved the newest hard challenge but regressed protected eval.

So for endpoint promotion, Exp062 should not replace Exp056.

That is the principle:

```text
new challenge improvement does not erase protected eval regression
```

## Registry and Pointers

The dev promotion registry plan creates:

- adapter version
- adapter URI
- metadata URI
- current pointer URI
- rollback pointer URI
- run contract URI
- decision URI
- eligibility flag

The current pointer answers:

```text
what adapter should this environment/database use now?
```

The rollback pointer answers:

```text
what adapter can we return to if current breaks?
```

This makes promotion auditable. A platform engineer can inspect a JSON pointer instead of asking which folder is "the good one."

## Why This Matters

Without explicit promotion state:

- a model can be used because it is the latest file
- rollback depends on memory
- failed runs can accidentally become current
- downstream systems cannot tell why a version was selected

With explicit promotion state:

- current is a declared pointer
- rollback is a declared pointer
- decision is stored
- reasons are stored
- run contract is linked

## Interview Answer

```text
I treated promotion as a state change, not a file copy. The run contract produced a promote/reject/investigate decision from explicit gates. The registry plan then recorded adapter version, adapter URI, metadata URI, current pointer, rollback pointer, run contract URI, and decision URI.

That means a promoted adapter is not just weights. It is weights plus evidence plus an auditable pointer saying this is the current candidate for this database/environment.
```

Short line:

```text
Promotion is a controlled pointer update backed by eval evidence.
```

## Sources

- `src/sqlbench_lab/mlops/run_contract.py`
- `src/sqlbench_lab/mlops/promotion_registry.py`
- `src/sqlbench_lab/mlops/dev_cloud_publish.py`
