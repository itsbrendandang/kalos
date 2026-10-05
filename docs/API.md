# Engine API: uploading a run sheet

`POST /api/run` is the "run your own data" path: upload a CSV / TSV / Excel run
sheet, get back the analysis. Campaign endpoints (`/api/campaign*`) are in
[CAMPAIGN_LOOP.md](CAMPAIGN_LOOP.md); the experiment store (`/api/experiments*`)
is in [M2_INTEGRATION.md](M2_INTEGRATION.md); auth and tenancy are in
[HARDENING.md](HARDENING.md).

```bash
curl -F file=@sheet.csv http://127.0.0.1:8050/api/run                  # infer roles from headers
curl -F file=@sheet.csv -F roles=auto http://127.0.0.1:8050/api/run    # let the normalize tier decide roles
curl -F file=@sheet.csv -F target=Titer_g_L -F anonymize=true http://127.0.0.1:8050/api/run
```

## What the upload path guarantees

The portal accepts an uploaded CSV / TSV / Excel run sheet and returns the analysis
(auto-detected target, leakage-safe features, honest grouped-CV, signed drivers, a proposed next
batch). Because the sheet comes from an external client, the upload path is guarded:

- **Size + shape caps.** A raw-byte cap (default 25 MB, override with `KALOS_MAX_UPLOAD_MB`), a
  512-column ceiling, a 100k-row CSV cap, and a 2M-cell xlsx cap (a zip-bomb guard). Every cap
  fails closed: an over-cap upload is REJECTED with a 400, never silently truncated. The xlsx
  cell cap is enforced from the workbook's declared dimensions BEFORE the frame is materialized,
  so a zip-bomb is rejected without the memory spike. Filetype is sniffed by magic bytes
  (`PK\x03\x04` -> Excel, else text/CSV), not by extension.
- **Safe errors.** A bad upload returns a generic HTTP 400 ("Could not parse the uploaded file.
  Check it is a CSV or Excel run-sheet."). This holds even when the failure is a GP fit error
  (`torch.linalg.LinAlgError`) or a leakage-guard assertion, not just a parser error: a catch-all
  normalizes any such failure to the same `{error}` JSON envelope. Parser details, column names,
  cell values, paths, and stack traces are logged server-side and never returned to the caller.
- **Provenance.** The response includes a `provenance` list: for every column, its status
  (`kept_feature`, `target`, `dropped_id`, `dropped_output`, `dropped_constant`,
  `dropped_constant_on_fitted_rows`, `dropped_sparse`, `dropped_all_blank`) and how many cells had
  to be coerced from non-numeric text (units like "34.6 C"). A feature that varies over the full
  sheet but is constant on the target-present rows the GP actually fits is dropped and flagged
  (`dropped_constant_on_fitted_rows`), never silently pinned to a zero-width bound. No more silent
  column drops.
- **Data validation gate.** Before any column is typed or dropped, the sheet runs through
  `kalos.validation`: eleven checks covering unit consistency, physical bounds, duplicates vs
  replicates, missingness, informative missingness, outliers, provenance metadata, replicate
  adequacy, controls presence, constant columns, and columns constant within a group. The report is returned as `validation`, with `status` one of `pass`,
  `pass_with_warnings`, or `fail`. An **error** is a physics violation (pH 40, a negative titer, one
  column mixing g/L and mg/mL); a **warning** is possible but operationally suspect. Row lists are
  capped at 20 with the true total in `detail.n_rows_affected`, so a finding never implies its list
  is complete. `KALOS_VALIDATION_MODE=strict` refuses a `fail` upload; the default `warn` analyzes
  it anyway and returns the findings, so adding the gate did not change what the API accepts.
- **Units are converted, not discarded.** A column written `"34.6 C"` parses at 0% as a bare number,
  so it used to fail the >=80% numeric test and come back `dropped_sparse` - a real process input
  silently thrown away. Single-unit columns are now converted to their base unit before feature
  selection and reported in `validation.conversions` with a human label (`"Celsius"`). Only units the
  registry can actually convert are rewritten: a vessel column of `"5L"`/`"500L"` uses an
  unrecognized token, and rewriting it would turn an identifier into a measurement, so it is left
  alone and the client is told.
- **Proposals stay physically possible.** The design box for each continuous feature is built from
  physically valid observations only, so one `-999` sensor sentinel can no longer widen the search
  space into impossible recipes (it previously produced proposals at -422 C, below absolute zero).
  Any narrowing is reported in `design_box_exclusions`, never silent. This holds in every validation
  mode - it does not depend on the client reading the report.
- **Reproducibility.** The analyze path is seeded, so the same upload yields identical proposals;
  the response carries `seed`, `timestamp`, and `engine_version`. Protein embeddings pin the ESM-2
  checkpoint to an immutable commit, so upstream cannot change model features without a diff here.
- **Opt-in anonymization.** Pass the form field `anonymize=true` to pseudonymize identifier-type
  columns in the response. Feature and target names are kept as-is (the owner UI legitimately shows
  drivers like "Methanol").
- **Decided roles (`roles=auto`).** Pass the form field `roles=auto` to have the normalize tier
  decide target / features / group / ids instead of the bioprocess header patterns (a JSON
  `roles` object still declares them by hand). The response gains a `role_decision` block: which
  tier decided (`typesafe`, `llm`, or `offline`), whether it was applied, and each column's role.
  Rationale notes (with probabilities) are included only for the `typesafe` and `offline` tiers,
  whose notes code builds from names and numbers; an LLM's free-text rationale is withheld, so no
  cell value can ride along. With `anonymize=true`, identifier column names are aliased exactly as
  in `provenance`. If no consistent plan names a target and at least one feature, or the headers
  are not text, roles are inferred exactly as without the field. An explicit `target` that is a
  column of the sheet wins over the decided one, and the decided outcome then moves to `ids` so a
  measured outcome is never modeled as an input.
- **A baseline the GP must beat (`cv_xgboost_baseline`).** Every analysis also scores an XGBoost
  regressor with the same leakage-controlled grouped CV, on the same folds and held-out rows as the
  GP, and reports a paired verdict - `gp_better`, `xgboost_better`, `no_detectable_difference`
  (the GP-minus-XGBoost Spearman CI includes zero), or `not_computable` with a `reason` (fewer
  than 3 held-out groups, or a constant target). Fixed, untuned settings, so the baseline cannot
  overfit the folds it is scored on. Needs the `xgboost` extra; without it the block reports
  `available: false`.
