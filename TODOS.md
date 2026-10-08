# TODOS

## Infrastructure

### Per-prospect throwaway instance

**What:** Provision a single-tenant kalos instance per prospect from the deploy pack, and tear it down with verified, confirmed deletion.

**Why:** Before any paid instance exists, a prospect's real data may only be processed under a signed NDA/DPA on a throwaway instance created for that prospect, and deleted within 30 days with written confirmation.

**Context:** Deferred during the 2026-09-22 eng review of the Scale-Up Readout design (`~/.gstack/projects/itsbrendandang-kalos/brendandang-main-design-20260922-193108.md`, decision D2).
Provision from the deploy pack (commit `b6ce508`) on a fresh VM.
Teardown destroys the VM and its volumes.
Verify deletion by confirming the VM and volume IDs no longer exist in the provider.
Send the prospect a dated written note listing those IDs.
Prospect data never enters kalos-data or the founder's working instance.

**Effort:** M
**Priority:** P3
**Depends on:** The first prospect offering their own data for a backtest under NDA.

## Readout

### Server-side PDF for self-serve readouts

**What:** Generate the Scale-Up Readout PDF on the server instead of browser Save as PDF.

**Why:** Once customers run readouts themselves (Approach B), they need a PDF without a manual print step.

**Context:** Deferred during the 2026-09-22 eng review (decision D5); the concierge-phase readout is one self-contained HTML page with print CSS returned by `POST /api/scale/readout`.
Candidates: WeasyPrint (needs Pango/Cairo system libraries) or headless Chromium (hundreds of MB).
Evaluate both against the CPU-only, non-root deploy image (commit `b6ce508`).

**Effort:** S
**Priority:** P3
**Depends on:** The Approach B trigger: the buyer rule has resolved and at least one prospect has asked to run it themselves (decision D9).

## Completed
