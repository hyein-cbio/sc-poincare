# Astra 37-fit capsule: receiving-session audit

## What was received and preserved

The companion README and transfer index describe twelve ordinary ZIPs with
disjoint payloads, not byte-split archive pieces. All twelve compressed-file
sizes, SHA256 digests and ZIP CRCs matched the index. Renamed part 10 was matched
by its hash, not guessed from its filename. Entry paths, collisions and symlinks
were checked before extraction into one **new empty directory outside Git**.

The extracted transfer manifest SHA256 is
`3482d39b3bb8d4343d6cea7af0731a1322a7ffd5517d5b5d50cd8deb94068628`.
The original transport script verified **2810 files** before and after receiving-
session analysis. Original run/source/evaluation/checkpoint records were not
rewritten. Python bytecode and receiver outputs are not original payload files.

The already adopted public aggregate is byte-identical to
`docs/embedding-followup-results.json` inside the manifest-covered final
`repository-9f369a9.zip` (aggregate SHA256
`5b21ca5c3aa38a76819dbb48f0d5e55d335be5845d963a2012a284319ba264c3`).
No second raw capsule or dataset copy was put into Git or under the result tree.
Input/reference licenses remain separate from package MIT.

## Status: exact storage checks versus platform recomputation

The original `audit_capsule.py --states --selector` command **did not pass** on
the receiver. It stopped at the first Poincare `current` reconstruction requiring
bit-for-bit equality. This result is retained in the receiver log; the original
script and its assertion were not patched or bypassed and reported as passing.

Original runtime: Python 3.12.14, Torch 2.14.1+cpu. Receiver: Python 3.14.8,
macOS arm64, Torch 2.14.1. The reported numerical package versions otherwise
match the main NumPy/SciPy/sklearn/pandas versions. This is not an exact replica
of the original operating system/runtime and not a cross-platform resume claim.

A separate receiver supplement audited all **37** states and explicitly
separated two classes of comparison:

- **Exact:** semantic checkpoint digest, file hashes, stored best coordinates,
  complete loss traces, best epoch/loss, finite tangent and Adam moment tensors,
  Adam step 1500, run/setup/parameter/input/source lineage, archived feature and
  CSV hashes. Original final states remained at 1500; no update was executed.
- **Recomputed current coordinates:** 19 of 37 states have non-bit-identical
  coordinates after the CPU tangent-to-disk transform. The largest difference
  over all states is **2.220446049250313e-16**. All are within the supplement's
  separately declared absolute bound 1e-15 (zero relative tolerance). This is a
  roundoff-qualified supplement, **not** a pass of the original exact command.

The states and source tags were not changed to force compatibility with current
main or the receiver's Torch build. No target cache was regenerated and no
optimizer continuation or historical research queue/deadline was invoked.

## Selector replay is not fresh evaluation

The original companion's separate `--selector` invocation passed after verifying
all 2810 files. It imports the frozen **local-teacher** selector, captures its
writer in memory and uses archived evaluations. The selected condition is null.
The incomplete global selector was not executed and no null global selection
was fabricated.

That replay alone does not establish independent metric recalculation. The next
section is a separate receiving-session computation, with separate code/output.

## Fresh metrics from all 37 raw coordinate sets

The receiver wrote a new evaluator; it does not import archived evaluation
functions, training queues or main's scoring API. It reads each raw coordinate
array, exact fitted features and source CSV expression/labels, verifies hashes,
and computes metrics **before** consulting the archived expected scores.
Only source/path resolution was adapted; original files were not edited.

Scope:

- All-cell nonself K5/10/15/30 purity, macro purity and every population's count,
  purity, reconstruction MSE and finite-population ceiling.
- Every variable gene's variance-normalized neighbor MSE and aggregate MSE.
- Fitted-feature neighbor recall and neighborhood boundary-tie counts, retaining
  the frozen default NumPy argsort/nearest-neighbor policy rather than swapping
  to main's stable-index tie rule.
- Available Moignard stage ordering with the declared tied-stage map and PS
  fitted-feature medoid root, not a root chosen from favorable results.
- Native Euclidean coordinates remain Euclidean; no disk projection or forces.

Primary distances follow the frozen cdist/asinh formula. A secondary all-run
K15 calculation uses independent broadcast/acosh distances and checks the same
population/gene arithmetic. Floating comparisons use a declared 1e-10 bound;
counts, categories and decision booleans are not silently rounded into agreement.

Results:

| Check | Receiving-session result |
|---|---|
| Raw quality coordinate sets evaluated | 37 / 37 |
| Numeric fields compared to archived evaluations | 87160 |
| Maximum absolute metric discrepancy | 7.549516567451064e-15 |
| Fresh primary comparison failures | 0 |
| Secondary acosh/broadcast K15 discrepancies beyond bound | 0 |
| Training/optimizer updates | 0 |

This is fresh arithmetic reproduction on the original datasets, **not** blind
biological validation, new experiments, confidence intervals or new seeds.
The field count belongs to this receiver comparison; it does not reuse Astra's
103363-field count, which came from a different collection of audit records.
Graph-128 diagnostics, fresh target construction, common-objective/gradient-
conflict analysis and recovery continuation were not rerun by this receiver
metric audit. They must not be inferred from the K-metric agreement.

## Frozen local selector with fresh, not archived, scores

The original local-selector implementation was then replayed with its loader
bound in memory to the **fresh receiver metrics**. The original writer was
captured; saved selection/source files were not overwritten.

- Selected condition: **none**, matching the archived decision.
- Target/guardrail decisions: all exactly equal.
- 536 numeric deltas compared; maximum difference **4.440892098500626e-16**.
- Baseline/balanced/multiscale remain the prescribed diagnostic transfer set.
- Global selection, confirmations and transfer remain unexecuted because that
  screening was user-stopped. The missing eighth global screening fit was not
  launched; cancelled work was not turned into negative outcomes.

## Consequence for main and next work

The handoff now supports a stronger claim than the earlier aggregate-only
review: raw storage/state lineage was checked with an explicit platform caveat,
and all 37 per-run K metrics plus the local selection decision were independently
recomputed in the receiving session. It does **not** reverse the scientific
outcome or justify promoting balanced/multiscale/union-global target variants.

No duplicate teacher/target implementation, new default, floor, label-based
weight or fresh fit was added. Existing observability/test adoption remains
appropriate. Independent-data validation and scalable approximations are still
separate tasks; known Olsson transfer and repeated baseline fits are not new
blind cohorts.

The handoff was initially received under Downloads, then moved at the user's
request into this project's ignored `data/astra37_handoff/` directory. Original
ZIPs are under `archives/`, and receiver files are now under
`data/astra37_handoff/capsule/`:

- `receiver-zip-preflight.json`, `transport-audit.json`, `transport-post-audit.json`
- `selector-only-audit.json`
- `receiver_analysis/original-state-command.log`
- `receiver_analysis/receiver-state-audit.json` and its separate audit script
- `receiver_analysis/fresh_metrics/` and its separate evaluator/policy/log
- `receiver_analysis/fresh-local-selector-audit.json` and its separate replay script

`data/astra37_handoff/RELOCATION.json` maps former Downloads filenames/paths to
the current locations. All 2810 payload files, 49 receiver-inventory files and
12 ZIP checksums were reverified after relocation. Original provenance logs
retain their as-run paths; no scientific records were rewritten.

A compact summary/checksum inventory is linked from the existing latest result
tree; both the raw capsule and archives remain excluded from Git by `data/`. Local extraction verifies delivery,
not the durability of an independently managed off-machine backup service.
No commit or push was performed by this receiving audit.
