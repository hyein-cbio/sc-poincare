# Graph-stress and landmark integration: preservation audit summary

This is a historical summary of the rebase/retirement audits and the
2026-11-01 session-retention inventory, not a current validation certificate.
The repository checks below were independently confirmed against main at
`f06a0d5c7b83d5dc0895b3e73e81e244c0c0a3d4` before adding this note.
No additional development code needed to be recovered from the supplied archive.

## Integration record

- [PR #2](https://github.com/hyein-cbio/sc-poincare/pull/2) integrated the
  graph-stress series. The historical audit mapped all 31 original commits to
  replayed commits: 23 equivalent patches and 8 adapted patches. Five conflict
  episodes combined main's safety guards, numerical controls and tests with the
  branch's diagnostics and opt-in experiments. The eight adapted patches are
  not eight conflicts. Four living-document originals were archived separately.
- [PR #3](https://github.com/hyein-cbio/sc-poincare/pull/3) integrated the
  landmark series. All 46 original commits have replay counterparts with
  matching author metadata and messages. Conflicts in two commits retained
  landmark files already present in main; live documents retained additive main
  context. Seven probe entrypoints subsequently received closed-research path
  guards, followed by reviewed geometry and gradient fixes.
- The final graph-stress fix tip `6c351bc2b334d9bf3ccfb3ef25c49e3e5bab82ca`
  and landmark fix tip `13ce854a00aedeb4f115749159ee96c955da0906` are ancestors
  of the checked main snapshot. Replaying the old branches again is unnecessary.

## Verified preservation at the checked snapshot

| Evidence | Result |
| --- | --- |
| Original graph-stress research JSON | All 9 files byte-identical to the original cursor tip |
| Original landmark research JSON | All 15 files byte-identical to the original cursor tip |
| Graph-stress living-document originals | All 4 archived snapshots byte-identical to the original cursor tip |
| Original commit identities | Preserved through the exact remote legacy tips below, not through rebase identity |

The four document snapshots are under
[graph-stress-remaining-tasks/](graph-stress-remaining-tasks/README.md).
The verified remote branch tips were:

| Remote branch | Exact original tip |
| --- | --- |
| `legacy/graph-stress-by-astra` | `9f369a99406309a3bae43265759c5cc8afd0fd5e` |
| `legacy/graph-stress-by-cursor` | `0b3033adcb67cbed1b60ebf087cf9722a3a63a68` |
| `legacy/landmark-reference-by-cursor` | `3aa5b1240ffbe9e45a02abaaaa0a758924dde814` |

Earlier retirement warnings that no other remote ref protected the original
history were superseded by these legacy refs. Branches remain mutable: verify
an independent durable ref or backup before deleting or rewriting them. A SHA
written in a document is not itself a backup. Deleting a chat session and
deleting a Git ref are separate actions.

## Preservation is not scientific revalidation

Rebase preserves integrated work but creates new commit identities and can
intentionally change code. MU setups/checkpoints bind numerical source hashes;
old-source states must not be forced into current code by bypassing integrity
checks. Preserved historical JSON does not certify corrected algorithms,
current defaults, convergence or biological performance.

Later work already present in the checked snapshot includes the
[Moignard rerun records](../moignard-pr3-revalidation-2026-10-30.md),
[saved-record verifier follow-up](../moignard-verifier-followup-2026-11-01.md),
and [landmark gradient/held-out review](../paul-landmark-critical-review.md).
The preservation check did not rerun experiments or the test suite and does
not replace those records or their stated limitations.

## Source material and retention boundary

Sources: `sc-poincare-session-retention.zip`, its
`session-retention-inventory.json` and `session-retention-report.md`, and the
four historical reports `graph-stress-rebase-audit.md`,
`graph-stress-branch-deletion-audit.md`, `landmark-rebase-audit.md` and
`landmark-retirement-audit.md`. The supplied ZIP's 13 checksummed payloads
matched its SHA256 manifest; the separately supplied report matched the copy
inside the ZIP.

Ten historical audit artifacts (four reports, three scripts, two range-diffs
and one JSON ledger) had no identical blob in the fetched remote history.
Their conflict rationale and original-to-replayed commit mappings have archival
value, but they are not missing package features. Their fixed revisions,
obsolete branch names and VM-specific paths make the scripts unsuitable as
current validators without adaptation; they were not executed for this summary.

This note preserves only the essential conclusions. It does not embed the ZIP,
full ledgers, raw range-diffs, personal details, VM paths or unrelated-project
inventories. Keep the original archive separately if those exact records are
needed. Neither this note nor Git refs preserve complete conversations, raw
execution logs, ignored data/checkpoints, environments or external backups;
this audit does not establish that deleting an entire VM is lossless.
