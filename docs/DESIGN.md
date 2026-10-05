# Design notes — Kalos

How the Kalos **engine portal** (`src/kalos/portal/index.html`, served at :8050) looks and feels.

> **Current as of 2026-08-11, confirmed directly by the product owner: the engine portal is kalos blue on Satoshi, matching kalos-web.**
> The emerald-and-clay direction below is **superseded**, including its "no cobalt" rule, and must not be reinstated.
>
> This was asked and answered explicitly, because the two repos contradicted each other in writing.
> `kalos-web/DESIGN.md` (2026-08-08) recorded owner confirmation that kalos owns a blue hue on Satoshi; this file (2026-06-29) named emerald the one brand color and banned cobalt outright.
> Both were faithfully implemented, so the same product had two front doors with two identities.
> Blue is the live direction for both.
> If a future note claims otherwise, it needs a date later than this one and its own owner confirmation.
>
> **`kalos-web/app/globals.css` `:root` is the single source of truth for live token values.**
> The `:root` block in `src/kalos/portal/index.html` is a mirror of it, not a second source; when a brand value changes there, update the portal to match.
> Only the brand family and the type stack changed in this re-hue.
> Layout, shape, spacing, motion, and the anti-slop rules below are untouched, so this is a re-hue rather than a redesign.

## The memorable thing

After someone sees Kalos once, they should remember: **a calm, precise tool that tells you what to run next.**
Bright and light, not a dark instrument console.
Precise where it counts, friendly everywhere else.
Every choice below serves that.

## Color

Two tones, used for meaning, never decoration.

- **Kalos blue `--accent: #204AF4`** is the one brand color.
  It marks the live series, the primary action, the active nav, and good outcomes.
  Hover `#1442BE` (darker, so white button text keeps its ratio), deep `#123EB2` for the filled hero card, soft fill `#EEF4FE` for tints and chips.
- **Explore `--explore: #9CA3AF` is achromatic on purpose.**
  It marks "explore" (high-uncertainty) against blue "exploit", so the pair reads as a hue contrast rather than two similar blues.
  Deep text `#52525B` on soft fill `#EDEDED`.
- **Status carries meaning, never decoration:** amber `#9A5B00` on `#FFF3E0` for warnings and for labelling demo or benchmark-only numbers.
  Severity is never conveyed by color alone; every status also carries a label or an icon.
- **Surfaces:** page `--bg: #F6F8FC`, white cards `--card: #FFFFFF`.
- **Ink:** `--fg: #101828`, `--muted: #667085` (clears WCAG AA 4.5:1 on both page and card).
- **Lines:** `--border: #D9E2F2`, soft. Used sparingly; depth comes from elevation, not boxes.
- **Diverging data** (correlations, signed drivers): blue for positive, neutral gray for negative.
  Never red/green traffic light.

## Type

Two real typefaces, matching kalos-web (no Inter / Roboto / Arial / system default).

- **Display and body — Satoshi** (`--head`, `--sans`), weight 400-700, headings tracked tight, body ~65ch max.
  One face across both roles, which is what ties the portal to the dashboard visually.
  Satoshi is not on Google Fonts, so the portal serves it vendored from `src/kalos/portal/fonts/` under the Fontshare license (see the `/fonts` route in `src/kalos/portal/app.py`).
- **Numbers, IDs, code — JetBrains Mono** (`--mono`), weight 500-600, always `tabular-nums`.
  Mono is reserved for figures, the one cue carried over from instrument UIs because scientists trust aligned, precise numerals.

## Shape, depth, spacing

- **Rounded:** 14px cards, 10px controls, pill badges (`border-radius: 999px`).
- **Elevation, not hairlines:** cards are white on the page tint with a soft, low
  shadow `0 1px 2px rgba(16,24,40,.04), 0 10px 26px rgba(16,24,40,.05)`. Lead
  with depth and whitespace, not borders around everything.
- **Generous spacing:** 20-24px card padding, 12-16px gaps, a max-width canvas.
- **The hero accent card** (the single most important number, e.g. best titer)
  is a filled blue card with white text and a soft blue glow. Exactly one
  per screen.

## Components

- **KPI cards** — soft white card, muted 11px label, 22px mono value, a one-line
  context in muted or blue. The single most important one is the blue hero variant.
  A number that cannot be computed on real client data (a benchmark against a known
  optimum, a demo figure) is labelled as such in amber, right in the KPI label.
- **Primary button** — blue fill, white text, 10px radius, soft blue
  shadow, a leading icon. Secondary is outline on `--border`; ghost is bare.
- **Charts** — themed to the palette: blue line (2.5px) with an
  8%-opacity blue area fill, neutral gray for the secondary series and for
  reference lines, `--grid` gridlines, muted mono axis labels, no animation on chrome.
- **Badges / chips** — pill, soft-fill background with the same-hue *deep* text
  for AA contrast (blue chip = `#1442BE` text on `#EEF4FE`; explore chip =
  `#52525B` text on `#EDEDED`; demo/benchmark chip = `#9A5B00` on `#FFF3E0`).
  Never the mid-tone accent on its own soft fill.
- **Hero on color** — the hero card uses the deeper `#123EB2` so white
  label and value both clear AA, not the mid `#204AF4`.

## Motion

Subtle and smooth, only on genuinely live data (a value updating, a chart
drawing). Chrome stays still. 150-250ms ease. Respect `prefers-reduced-motion`.

## Rules / anti-slop

- No dark terminal aesthetic, no hairline-grid console look, and it must not read
  as a stock shadcn neutral admin template — kalos owns a blue hue on Satoshi.
- No purple gradients, no glow on text, no 3-equal-icon-card rows, no centered
  everything, no decorative blobs.
- Sentence case everywhere. No ALL CAPS headers.
- Mono is for numbers and IDs only, never body copy.
- One brand hue plus achromatic explore on a screen. If you reach for another
  color, it is data or status, not decor.
- Every screen leads with the one number that matters (the blue hero), then
  the trend, then the recommendation. Newcomer-first reading order.
- Never show a number that cannot exist on a client's own data without saying so
  next to it. The demo panels are badged "Example data" for the same reason.
