# Validation-readiness and gene-metadata audit

## Scope

Starting source: main `3cbdd5fca0503655a2639b5551dc3d75b8af6d23`.
This is a metadata/usage audit and post-hoc interpretation, **not another fit**.
No expression preprocessing, model/default change, marker tuning, original-input
rewrite, new download or external upload was performed. No reference algorithm
implementation was inspected for this audit.

Local evidence is under
`data/graph_diagnostics_integration/metadata_audit/manifest.json` and its linked
sidecars. That ignored directory is not part of a fresh Git clone or an external
backup. Original data/reference licenses remain separate from package MIT.

## Gene names recovered from the actual study input

The shared NPZs do not contain gene names, but the exact input CSVs do. Both CSV
checksums match `study_input_manifest.json`. Parsing with the default pandas CSV
reader reproduced **every expression value and every label in the same order**:

| Dataset | Frozen expression shape | CSV headers recovered | Unique |
|---|---:|---:|---|
| Paul | 2730 x 1000 | 1000 | yes |
| Moignard2015 | 3934 x 42 | 42 | yes |

This verifies column-name mapping to the frozen processed study input. It does
not independently establish author barcode identity, preprocessing equivalence
to a different raw dataset, standardized gene IDs or assay-symbol aliases.

The larger Paul H5 has 8716 expression rows. Taking its first 1000 row labels
would be unjustified and was not done. The Moignard workbook remains available,
but there was no need to infer a raw Ct/dCt transformation to attach the exact
headers of the actual fitting/evaluation input.

Sidecar maps record source checksum, expression checksum, parser, order checks,
original column index/name and the author-cell-identity limitation. Frozen NPZs,
fit coordinates, losses and checkpoints were not modified. All ten full-model
coordinate hashes and shared input-file hashes still match their original run
records.

## Named interpretation, with selection limits

A fixed broad hematopoietic/endothelial marker list was saved in the local audit
protocol before loss indices were joined to names. Exact original-header matches
only were accepted; absent exact matches remain missing. An assay alias or older
symbol may explain a missing match: it is not evidence that the gene is absent.
Top gains/losses are separately labeled exploratory. No gene significance test,
expression-direction inference or independent biological validation is implied.

Against the original shared-init RFA baseline, the existing K15 reconstruction
loss counts remain unchanged:

| Dataset | MU fixed worse genes | MU local-radius worse genes |
|---|---:|---:|
| Paul | 419 / 1000 | 410 / 1000 |
| Moignard2015 | 9 / 42 | 8 / 42 |

Examples of the now-named normalized-MSE deltas (candidate minus RFA; positive
means worse reconstruction, not increased expression):

| Dataset / gene | MU fixed | MU local-radius |
|---|---:|---:|
| Paul / Gata1 | +.0046 | +.0033 |
| Paul / Kit | +.0044 | +.0075 |
| Paul / Pf4 | -.2810 | -.2905 |
| Paul / Prg2 | -.2423 | -.2804 |
| Moignard / Runx1 | -.0730 | -.0763 |
| Moignard / Kit | -.0375 | -.0499 |
| Moignard / HoxB4 (exploratory) | +.0647 | +.0687 |
| Moignard / Procr (exploratory) | +.0527 | +.0546 |

Paul fixed's largest aggregate gene losses include Mrvi1, Ppp3cc and Sh2d2a;
local-radius includes Cd2, Mrvi1 and Ppp3cc. Moignard's largest losses include
HoxB4, Procr and Etv6 under both rules. These observations show why average MSE
cannot establish uniform marker preservation. They do not establish which model
has better biological branches, nor can all losses be dismissed as irrelevant.

The named sidecar report links the unchanged source per-gene reports rather than
copying datasets or producing another family of fitted coordinates.

## Candidate usage audit

Accessible local text metadata were searched (1526 files), supplemented with
experiment ledger CSVs and Git subject searches. This scope cannot prove absence
of use outside the accessible records, in deleted environments, or under aliases.

| Candidate | Accessible evidence | Decision |
|---|---|---|
| Pancreas E15.5 | Earlier fits, parameter screens and three-seed confirmation | Already development data; not a new holdout |
| Nestorowa | Earlier fits, parameter screens and three-seed confirmation | Already development data; not a new holdout |
| Farrell | Loading attempt recorded; no completed full fit found in the inspected outputs | Not certified untouched; full exact size is unsuitable |
| Weinreb LARRY in vitro | Download record says not fit; no fit evidence found in searched reports/ledgers | Best current candidate, subject to prospective protocol and remaining gates |

Earlier protocols explicitly excluding full LARRY runs support the last entry,
but do not certify that every possible subset was never seen. It must be called
a new validation candidate, not retroactively a proven blind holdout.

## LARRY feasibility and unresolved gates

Header/metadata inspection, without loading the full count matrix, found:

- 130887 cells, 25289 unique nonempty gene labels.
- Normalized count matrix: 176867560 stored entries, about 2.07 GB compressed.
- Time points 2/4/6; starting-population, library/well and cell-type metadata.
- Library + cell-barcode pairs are unique; bare cell barcodes are not.
- Clone matrix: 5864 clones, 49302 cells with exactly one recorded clone, no
  multi-clone cells. 3894 clones span multiple time points; 971 span all three.

Time/clone measurements may support external time/fate diagnostics. Expression-
derived cell-type annotations are not independent lineage truth, and the supplied
SPRING-x/y are another layout, not biological ground truth. Those metadata,
clone matrices and coordinates must not become fitting features or initializers.

Full exact MU is not feasible under the current guard: four float64 N x N arrays
alone would be about **510.6 GiB**. Blindly calling mmread/densifying the entire
counts matrix is not the next step either. Subset extraction must be bounded and
recorded separately from model fitting.

A uniform 3000-cell sample would have only about **2.3 expected pDC cells** from
the supplied annotation counts. It cannot support strong rare-population purity
claims at K15. No sample was selected or adjusted during this audit, and no
label-guided resampling was performed to obtain a more favorable evaluation.

Before any fit:

1. Confirm the author-supplied row-order contract connecting expression,
   metadata and clone matrix, not just matching dimensions.
2. Confirm data-use/redistribution terms; public hosting does not transfer the
   package MIT license to the files.
3. Freeze the observation/cohort selection rule, dimensions, preprocessing,
   roots and primary/sensitivity endpoints. Keep experimental truth out of the
   model and separate expression-derived label scores from external evidence.
4. Estimate preprocessing/fit memory and time; define abort/rare-coverage limits
   before loading/extracting the count matrix. Do not disable guards.
5. Identify an authorized off-machine backup destination before long/unique runs.

## Revised immediate action

Gene metadata is now ready for interpretation of the existing study; no H5
positional guess or extra training was required. The remaining immediate task
is a **LARRY protocol/readiness decision**, not another optimizer/tau search.
If its provenance, subset extraction or rare-coverage requirements are too costly,
select a genuinely unused smaller dataset with explicit gene/time metadata
instead of forcing LARRY merely because it is already downloaded.

No new long-running job, independent-data success claim, off-machine backup,
commit or push was performed by this audit.
