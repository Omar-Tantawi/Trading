# Market Profile: Testable Hypotheses for the Trading Buddy

Source: James F. Dalton et al., *Mind Over Markets* (updated edition), read
2026-09-23. This document restates the book's ideas in our own words as
**definitions we can compute** and **claims we can test**. It reproduces no
text from the book. Every claim below is a hypothesis until our own backtests
on Binance data say otherwise.

Intended consumers: sub-project 2 (features), 3 (labels and models),
4 (backtests), 7 (bot advisor).

## 0. Read this first: what does and does not transfer to crypto

The book was written about exchange-floor futures (Treasury bonds, S&P, grains)
in the late 1980s. Three differences change how every idea below must be
implemented:

1. **Crypto has no session.** The book's framework hangs on a daily session with
   an open, a first hour ("initial balance"), and a close. Binance trades 24/7.
   We must *choose* a session definition, and the choice is itself a
   hypothesis. Candidates:
   - the UTC calendar day (matches Binance's own daily candle), or
   - regional windows (Asia ~00:00–08:00 UTC, Europe ~07:00–16:00, US ~13:30–20:00).
   Every session-based feature below takes the session definition as a
   parameter, and we test which definition carries signal.

2. **Gaps essentially do not exist on a 24/7 venue.** With contiguous sessions,
   each session opens at the previous close, which lies inside the previous
   range. So "open outside the previous day's range", the book's highest-
   opportunity case, **cannot occur** with UTC-day sessions. Gap ideas only
   become testable with non-contiguous windows (e.g. US session vs the previous
   US session). Treat the gap material as low priority.

3. **The book's statistics are tiny and old.** Its quoted hit rates come from
   single markets over roughly one year (1986–87 bonds). They are
   starting claims to beat, not facts. Expect them to shrink on crypto.

**Data requirement, stated precisely.** Market Profile has two flavours:
- **TPO (time-at-price) profile**: *fully computable from our existing 1-minute
  candles.* A price row gets a TPO for a period if that period's high–low range
  covered it. Aggregate 1m to 30m and every TPO construct below is exact.
- **Volume-at-price profile**: only *approximate* from candles (spread each
  candle's volume across its high–low range). Exact values need trade-level
  data (Binance aggTrades), which the data foundation deliberately deferred.

The book's own appendix argues time-at-price is the more informative of the
two. **So nothing here requires new data to start.** Recommendation: build on
TPO profiles first; test whether an approximate volume profile adds anything;
add an aggTrades collector only if it does.

One more implementation parameter: **price bin size.** BTC's tick is 0.01,
far too fine for profile rows. Bin size should scale with volatility (e.g. a
fraction of ATR) so profiles are comparable across time and symbols. Tune it.

## 1. Core definitions (all computable from 30-minute candles)

For one session:

| Term | Our computable definition |
|---|---|
| TPO period | One 30-minute bar within the session |
| TPO | A (price-bin, period) pair where the period's high–low covered the bin |
| Profile | Count of TPOs per price bin across the session |
| Initial balance (IB) | High–low of the first two periods (first hour) |
| IB width | IB high − IB low, normalised by ATR or by its own rolling percentile |
| Range extension | Any trade above IB high (buying RE) or below IB low (selling RE) after the first hour |
| POC | Bin with the most TPOs; ties broken toward the centre of the range |
| Value area (VA) | Expand from the POC, adding whichever side's next *two* bins hold more TPOs, until ≥70% of all TPOs are included (the book's appendix algorithm) |
| VAH / VAL | Top / bottom of the value area |
| Single prints | Bins touched by exactly one period |
| Tail | ≥2 contiguous single-print bins at the session's extreme, *not* formed in the final period (it needs a later period to confirm rejection) |
| Poor high / poor low | A session extreme with no tail (fewer than 2 single prints) |
| TPO count | TPOs above POC vs TPOs below POC, excluding tail single prints |
| Rotation factor | Per period: +1 if high > previous high, −1 if lower, 0 if equal; same for lows; summed over the session |
| One-timeframing (up) | Each period's low ≥ previous period's low. Down: each high ≤ previous high. Applies at 30m, daily and weekly scales |
| Initiative vs responsive | Buying at or above the previous session's VA is initiative; buying below it is responsive (mirror for selling) |
| Balance / bracket | N consecutive sessions whose value areas overlap |

## 2. Hypotheses

Each entry gives: **Features** (inputs for sub-project 2), **Claim** (what the
book asserts, in our words), and **Test** (how we measure it on Binance data).
The priority reflects expected payoff × how cleanly it transfers.

### H1. Initial-balance width predicts range extension — priority HIGH
- **Features:** IB width percentile.
- **Claim:** a narrow first hour is easily broken, so range extension and trend
  days are more likely; a wide first hour tends to contain the session.
- **Test:** logistic regression of P(range extension) and P(session range >
  k·IB) on IB-width percentile, by session definition. Compare with ATR alone.

### H2. Day-type classification — priority HIGH
- **Features:** a categorical day type assigned at session close from IB width,
  which sides were extended, and the closing location:
  - *Nontrend*: narrow IB, no range extension
  - *Normal*: wide IB, no or little extension
  - *Normal variation*: extension on one side, then balance
  - *Trend*: extension on one side across multiple periods, thin elongated profile, open near one extreme
  - *Double-distribution trend*: narrow IB, then a late one-sided move leaving single prints between two TPO clusters
  - *Neutral-centre / neutral-extreme*: extension on both sides, closing mid-range or at an extreme
- **Claim:** day type reflects the conviction of longer-horizon participants and
  conditions the next session.
- **Test:** (a) is the classifier stable and non-trivial on crypto (all types
  occur)? (b) does yesterday's type predict today's range and direction beyond
  a volatility baseline? This also feeds the regime model.

### H3. Opening type predicts whether the early extreme holds — priority HIGH
- **Features:** opening type, classified from the first few periods:
  - *Open-drive*: moves one way from the start and never returns through the opening range
  - *Open-test-drive*: first probes beyond a reference (previous high/low), fails, then drives the other way through the open
  - *Open-rejection-reverse*: moves one way, is met by opposite activity, and returns through the opening range
  - *Open-auction*: rotates around the open with no direction
- **Claim:** open-drive extremes hold for the session in the large majority of
  cases; open-test-drive a little less; open-rejection-reverse under half the
  time; open-auction inside the prior range implies a low-conviction day.
- **Test:** for each type, P(the early extreme remains the session extreme).
  This is a clean, directly labellable claim. Note it depends heavily on the
  session definition, since crypto has no real opening bell.

### H4. Open location vs previous value predicts range — priority MEDIUM
- **Features:** the open's position relative to the previous VA and range
  (inside VA / outside VA but inside range; outside range is impossible for
  contiguous sessions, see §0), and whether price is *accepted* there
  (two consecutive periods trading at the level).
- **Claim:** opening inside value and being accepted means balance, so the
  session's range will be close to the previous one; the book estimates it by
  projecting the previous range from whichever early extreme looks secure,
  ±10%.
- **Test:** compare that range estimator's error with an ATR-based estimator.
  This matters beyond signals: a good range estimate is a direct input to
  **grid-bot range setting** in sub-project 7.

### H5. The value-area rule — priority HIGH
- **Features:** previous VA; whether price re-enters it and is accepted (two
  consecutive periods inside); distance of entry from the VA; VA width
  percentile; direction of the longer-term trend.
- **Claim:** once price is accepted back inside the previous value area, it
  tends to traverse to the opposite edge. Unconditionally the book calls this
  near a coin flip; it improves when the open is close to value, the value area
  is narrow, and the traverse is in the direction of the larger trend.
- **Test:** hit rate of reaching the far VA edge within the session,
  unconditional and conditioned on each of the three factors. Report the
  conditional lift, not just the raw rate.

### H6. "Three-to-I" days continue — priority HIGH
- **Features:** a buying tail, buying range extension and a TPO count favouring
  buyers, all three *initiative* (at or above the previous VA). Mirror for
  selling. Also the weaker variant where the tail is responsive rather than
  initiative ("2I-1R").
- **Claim:** after a three-to-I day, the next session trades beyond the
  previous VA in the trend's direction early on, and rarely closes on the wrong
  side of it. The book reports very high rates on a single year of bonds,
  lower for the 2I-1R variant.
- **Test:** replicate the book's exact measurement: in the next session's first
  90 minutes, and at the next close, is price better than / within / worse than
  the previous VA? Compare with all days and with matched-volatility days.

### H7. Neutral-extreme days continue — priority MEDIUM
- **Features:** a neutral day (extension on both sides) that closes at one extreme.
- **Claim:** the side that won the close tends to continue into the next session.
- **Test:** same better / within / worse measurement as H6.

### H8. Late-session spikes are judged by the next open — priority LOW–MEDIUM
- **Features:** a breakout from value in the last few periods of a session (the
  spike), and where the next session opens relative to it.
- **Claim:** opening inside the spike means acceptance and balance, with the
  spike's length a good estimate of the next range; opening beyond it in its
  direction means continuation; opening back through its base means rejection.
- **Test:** next-session direction and range, grouped by open location. Hard to
  define without a real session close; depends on the session definition.

### H9. Balance-area breakouts: go with acceptance, fade failure — priority HIGH
- **Features:** a balance area (N sessions of overlapping value); a breakout
  beyond its edge; acceptance (time spent beyond the edge) vs failure (quick
  return inside).
- **Claim:** an accepted breakout starts a directional move; a failed breakout
  often travels to the *opposite* edge of the balance.
- **Test:** continuation probability after accepted breakouts vs failures, and
  P(reaching the opposite edge) after a failure.
- **Why it matters especially here:** overlapping value is precisely the
  environment a grid bot wants, and an accepted breakout is precisely when a
  grid bot should stop. This hypothesis is a candidate **stop condition for
  sub-project 7**, grounded in a measurable definition rather than a guess.

### H10. Tails as excess; poor highs and lows get revisited — priority HIGH
- **Features:** tail length at each extreme; poor-high / poor-low flags;
  count of consecutive sessions with poor lows (or highs).
- **Claim:** a tail marks aggressive rejection and should act as support or
  resistance later; an extreme *without* a tail is an unfinished auction that
  price tends to come back and exceed. Several consecutive poor lows signal
  weak structure and rising risk of a sharp break.
- **Test:** P(the extreme is exceeded within N sessions) for tailed vs poor
  extremes. Separately: does a run of poor lows raise the probability of a
  large down move? That second part is a **risk-engine** input.

### H11. TPO-count imbalance — priority MEDIUM
- **Features:** TPOs above vs below the POC (excluding tails), as a ratio.
- **Claim:** it reveals which side is winning inside the value area when tails
  and extension are absent, and an end-of-day imbalance carries into the next
  session.
- **Test:** next-session return and direction vs the ratio, controlling for the
  session's own return.

### H12. Rotation factor as attempted direction — priority MEDIUM
- **Features:** cumulative rotation factor through the session.
- **Claim:** it measures which way the market is *trying* to go. By itself it
  does not say whether the attempt is succeeding.
- **Test:** use it as a feature, together with a "success" measure such as
  value-area migration (H14). The interaction is the interesting part.

### H13. POC migration shows who is in control — priority MEDIUM
- **Features:** intraday path of the POC; distance of POC from mid-range at
  close; buying above / selling below the POC late in the session.
- **Claim:** a POC that keeps migrating signals longer-horizon participants
  pushing; heavy selling below the POC late in a session ("short in the hole")
  raises the odds of a short-covering bounce next session.
- **Test:** next-session return conditioned on POC migration and on late
  activity relative to the POC.

### H14. Value-area placement sequence = trend vs bracket — priority HIGH
- **Features:** each session's VA relative to the previous one: higher,
  overlapping-higher, overlapping, overlapping-lower, lower; plus rolling counts.
- **Claim:** a sequence of higher (or lower) value areas is a trend being
  accepted; overlapping value is balance. Markets are claimed to spend most
  of their time balancing (the book cites roughly 70%).
- **Test:** (a) measure the balance share on crypto; (b) use the placement
  sequence as a **market-regime feature** and check whether it improves the
  regime model over EMA/ADX-only baselines. This is likely the single most
  valuable feature the book offers, because regime drives the whole bot advisor.

### H15. One-timeframing continues until it stops — priority HIGH
- **Features:** length of the current one-timeframing run at 30m, daily and
  weekly scales; a "run just broke" flag.
- **Claim:** trading against one-timeframing is dangerous; the end of a run
  marks change.
- **Test:** forward returns during active runs vs after they break, at each scale.

### H16. Value-area width — priority MEDIUM
- **Features:** VA width percentile vs its own history.
- **Claim:** narrow value means poor trade facilitation and is easily crossed;
  wide value is sticky.
- **Test:** interaction with H5 (traversal probability) and with H1.

### H17. Bracket extremes get tested several times — priority MEDIUM
- **Features:** bracket definition (value-area highs/lows or excess extremes over
  N sessions); number of prior tests of each extreme.
- **Claim:** extremes are typically tested several times (the book says about
  3–5) before a real breakout, and moves inside a bracket go extreme→middle
  rather than straight across, so responsive trades at the edges have the
  better location.
- **Test:** distribution of test counts before breakout; mean reversion from
  edges vs from the middle. Directly relevant to **grid spacing and range**.

### H18. Auction failure at known references reverses — priority HIGH
- **Features:** a probe beyond a reference (previous session high/low, weekly
  high/low, bracket extreme) with no follow-through.
- **Claim:** when a probe beyond a widely watched level finds no new business,
  price moves back through with speed, and the longer-horizon the reference,
  the larger the move.
- **Test:** forward return after failed probes, grouped by reference horizon.
  Needs a precise "no follow-through" definition (e.g. back inside within k
  periods); tune k.

### H19. Structure shape: elongated vs truncated — priority MEDIUM
- **Features:** profile elongation in the attempted direction (e.g. max TPOs
  per bin relative to range); "too elongated" flag.
- **Claim:** elongated structure in the attempted direction supports
  continuation; truncated structure undermines it; an extremely elongated
  one-sided day can signal exhaustion (inventory has become one-sided).
- **Test:** continuation probability by elongation bucket, watching for the
  predicted non-monotonic shape.

### H20. "Overnight inventory" adjustment — priority LOW (needs sessions)
- **Features:** with a primary session chosen (e.g. US hours), the share of
  off-session trading above vs below the previous primary-session close.
- **Claim:** when off-session trade leaves positioning one-sided, it tends to
  be unwound early in the primary session; if it is *not* unwound, that itself
  signals strength in that direction.
- **Test:** early primary-session return vs off-session inventory sign. Only
  meaningful if a regional session definition wins in §0.

### H21. When to stand aside — priority HIGH (maps to WAIT)
- **Features:** nontrend day flags; low-conviction days (open-auction inside
  prior value, no reference points); time to the next scheduled macro event.
- **Claim:** some sessions offer no edge and should not be traded; this is a
  decision in its own right.
- **Test:** do our models' signals have lower accuracy on these sessions? If so,
  the signal engine should emit **WAIT** there, which is exactly the spec's
  "WAIT is a first-class decision." The macro-event part needs an economic
  calendar, a new external data source for later.

## 3. Topics the extraction could not read

Pages ~121–171 and ~191–216 of the PDF did not extract as text (likely image
pages). They cover: short-covering rallies and long-liquidation breaks,
ledges, high- and low-volume areas, long-term directional performance
(volume, value-area placement, value-area width, composite days, island days),
additional bracket rules, trend-to-bracket transitions, and long-term auction
failures. Parts of these ideas also appear in the pages that did extract, and
are reflected in H10, H14, H16 and H17.

If these matter, options are: find those pages another way (a text-based copy
or OCR), or build equivalent features from general Market Profile literature
and label them as such. Nothing here should be attributed to the book that we
did not actually read.

## 4. Where each idea lands in the project

| Sub-project | Uses |
|---|---|
| 2 — Market intelligence | Profile engine (TPO, POC, VA, tails, IB, rotation factor, one-timeframing); day type (H2); VA placement (H14); balance-area detection (H9, H17) |
| 3 — Prediction ML | H1, H3, H5, H6, H7, H10, H11, H13, H15, H19 as features and as event labels |
| 4 — Backtesting | Each "special situation" (H5, H6, H7, H9, H18) as a standalone, rule-based strategy to backtest after fees against the baselines |
| 5 — Signals and risk | H21 → WAIT; H10 (runs of poor lows) → risk flag |
| 7 — Bot advisor | H14 (balance share) → grid suitability; H9 (accepted breakout) → grid stop condition; H4 and H17 → grid range and spacing |

## 5. Suggested first experiment

Before building the full profile engine, one cheap study answers the biggest
design question: **does any of this carry signal on crypto at all, and under
which session definition?** Compute IB width, day type and VA placement for
BTCUSDT under the UTC-day and US-session definitions, then test H1 and H14
against a volatility-only baseline. If neither beats the baseline under either
definition, deprioritise the rest of this document; if one does, it tells us
which session definition to build everything else on.
