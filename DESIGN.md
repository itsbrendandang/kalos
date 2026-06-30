# Design notes — Kalos

Authored for this repo. The source of truth for how Kalos looks and feels.
Standalone identity, deliberately unlike any prior project.

## The memorable thing

After someone sees Kalos once, they should remember: **a calm, optimistic
tool about growth.** Bright and warm, not a dark instrument console. Precise
where it counts, friendly everywhere else. Every choice below serves that.

## Color

Two hues, used for meaning, never decoration.

- **Emerald `--accent: #15A375`** is the one brand color (cultivation = growth).
  It marks the live series, the primary action, the active nav, and good
  outcomes. Hover `#0E7C57`. Soft fill `#E4F4ED` for accent tints and chips.
- **Clay `--explore: #C77E2E`** is the only second hue. It marks "explore"
  (high-uncertainty) vs emerald "exploit", and warnings. Soft fill `#FBEFDD`.
- **Surfaces:** warm paper page `--bg: #EFF1EA`, white cards `--card: #FFFFFF`,
  and a single grounding deep-green sidebar `--ink-surface: #13231C`. The dark
  sidebar is the only dark element; it anchors the bright canvas.
- **Ink:** `--fg: #15241C` (near-black with a green cast), `--muted: #586A5E`
  (darkened to clear WCAG AA 4.5:1 on the warm paper), faint `--muted-2: #98A69C`.
- **Lines:** `--border: #E8ECE3`, soft. Used sparingly; depth comes from
  elevation, not boxes.
- **Diverging data** (correlations, signed drivers): emerald for positive, clay
  for negative, with a neutral `#D8DCD2` midpoint. Never red/green traffic light.

## Type

Three roles, each a real typeface (no Inter / Roboto / Arial / system default).

- **Display / headings — Sora** (`--font-head`), weight 600-700, tracked tight.
  Geometric and modern, distinctive without shouting.
- **Body / UI — Plus Jakarta Sans** (`--font-sans`), weight 400-600, ~65ch max.
  Humanist and friendly; this is what makes it approachable.
- **Numbers, IDs, code — JetBrains Mono** (`--font-mono`), weight 500-600,
  always `tabular-nums`. Mono is reserved for figures, the one cue carried over
  from instrument UIs because scientists trust aligned, precise numerals.

## Shape, depth, spacing

- **Rounded:** 14px cards, 10px controls, pill badges (`border-radius: 999px`).
- **Elevation, not hairlines:** cards are white on warm paper with a soft, low
  shadow `0 1px 2px rgba(20,40,30,.04), 0 8px 24px rgba(20,40,30,.05)`. Lead
  with depth and whitespace, not borders around everything.
- **Generous spacing:** 20-24px card padding, 12-16px gaps, a max-width canvas.
- **The hero accent card** (the single most important number, e.g. best titer)
  is a filled emerald card with white text and a soft emerald glow. Exactly one
  per screen.

## Components

- **KPI cards** — soft white card, muted 11px label, 23px mono value, a one-line
  context in muted or emerald. The single most important one is the emerald
  hero variant.
- **Primary button** — emerald fill, white text, 10px radius, soft emerald
  shadow, a leading icon. Secondary is outline on `--border`; ghost is bare.
- **Charts** — themed to the palette: emerald line (2.5px, round caps) with a
  10%-opacity emerald area fill, clay for the second series, `--border` grid,
  muted mono axis labels, no animation on chrome. A filled end-dot with a soft
  halo marks "current best."
- **Badges / chips** — pill, soft-fill background with the same-hue *deep* text
  for AA contrast (emerald chip = `#0E7C57` text on `#E4F4ED`; clay chip =
  `#A05F20` text on `#FBEFDD`). Never the mid-tone accent on its own soft fill.
- **Hero on color** — the emerald hero card uses the deeper `#0E7C57` so white
  label and value both clear AA, not the mid `#15A375`.
- **Sidebar** — deep-green `--ink-surface`, the leaf wordmark, active item in
  emerald fill, inactive in muted sage. The one dark surface.

## Motion

Subtle and smooth, only on genuinely live data (a value updating, a chart
drawing). Chrome stays still. 150-250ms ease. Respect `prefers-reduced-motion`.

## Rules / anti-slop

- No dark terminal aesthetic, no cobalt, no hairline-grid console look (that was
  the prior, separate project — Kalos is its own thing).
- No purple gradients, no glow on text, no 3-equal-icon-card rows, no centered
  everything, no decorative blobs.
- Sentence case everywhere. No ALL CAPS headers.
- Mono is for numbers and IDs only, never body copy.
- Two hues max on a screen. If you reach for a third color, it is data, not decor.
- Every screen leads with the one number that matters (the emerald hero), then
  the trend, then the recommendation. Newcomer-first reading order.
