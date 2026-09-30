# AI Trading Buddy for Binance — Original Vision

The user's own vision document, as pasted into the first working session
(2026-09-22) and saved here on 2026-09-30 so later sessions can read it.
Sub-project specs cite its section numbers.

**Removed from scope at the user's request:** everything about image upload,
chart screenshots, vision models and visual training (sections 42–50, 88–90,
93, 96 and Milestone 10 / section 114). Those sections are omitted below and
marked *[removed]*. Mentions of images inside other sections are also out of
scope. The build order actually followed is the programme table in
`docs/superpowers/specs/2026-09-23-data-foundation-design.md` §2.

---

## 1. Project Vision

The AI Trading Buddy is an intelligent cryptocurrency trading analysis and
portfolio-monitoring platform. It helps a trader understand the market,
identify potential opportunities, evaluate risk, monitor existing automated
trading strategies, and make better-informed trading decisions.

It combines: historical cryptocurrency market data; live Binance market data;
technical analysis; statistical analysis; machine-learning models;
market-regime detection; pattern recognition; candlestick analysis; order-flow
analysis; risk-management algorithms; portfolio analysis; backtesting; Binance
simulated/demo trading; trade journaling; existing Binance bot monitoring;
Binance bot recommendation and advisory; bot capital-allocation
recommendations; bot duration and stopping recommendations; futures bot
leverage analysis; profit and loss analytics; model-performance monitoring; a
conversational LLM trading assistant; a complete trading dashboard.

Questions the final system should answer include:

- Should I buy BTC right now?
- Is this a good time to sell ETH?
- Why are you recommending that I wait?
- What are the strongest opportunities right now?
- What would make you change your opinion about BTC?
- How risky would it be if I bought $500 worth of SOL now?
- Which coin currently has the strongest setup?
- How are my Binance bots performing today?
- Which of my bots is the most profitable?
- Why did Bot A lose money yesterday?
- Compare the decisions of my Binance bot with your own prediction model.
- Which Binance bot should I use right now?
- Is a Grid Bot suitable for the current BTC market?
- Should I use a Spot Grid Bot or Futures Grid Bot?
- How much money should I allocate to this bot?
- How long should I leave this bot running?
- Should I stop this bot now or continue running it?
- Is my bot still operating under the market conditions it was designed for?
- What leverage would be reasonable for this Futures bot?
- Is the additional expected return from leverage worth the additional
  liquidation risk?

The system is meant to be much more than a simple trading bot. It should
eventually combine a trading intelligence platform, an AI trading assistant,
a Binance bot advisor, a bot monitoring platform, a trading education and
learning system, and a portfolio and risk dashboard.

## 2. Core Project Objective

Build a system that can answer:

- What is happening in the market?
- Why is it happening?
- What might happen next?
- How confident are we?
- What patterns are present?
- What is the potential reward?
- What is the potential risk?
- Should we trade?
- Should we wait?
- If we trade, what would a reasonable setup look like?
- How are my existing bots performing?
- Are my existing bots making good decisions?
- Which type of Binance bot fits the current market?
- How much capital should I allocate to a bot?
- How long should the bot remain active?
- When should the bot be stopped?
- What leverage is appropriate for the current risk?
- Can the AI learn from additional examples that I provide?

The system never assumes predictions are guaranteed; it reasons
probabilistically. Example output:

```text
BTC/USDT
Bullish probability: 73%
Bearish probability: 18%
Neutral probability: 9%
Market regime: Strong bullish trend
Risk: Moderate
Current resistance: $X
Current support: $Y
Overall recommendation: WAIT FOR BREAKOUT CONFIRMATION
```

The LLM can then explain why.

## 3. High-Level Architecture

```text
BINANCE
 ├─ MARKET DATA → MARKET DATA ENGINE ─┐
 └─ ACCOUNT / BOT DATA → BOT MONITOR ─┴→ DATABASE
DATABASE → price data features, bot trades & profits, historical data
       → FEATURE ENGINE → technical analysis | market regime | pattern AI
       → MACHINE-LEARNING MODELS (15m, 1h, 4h models)
       → ENSEMBLE ENGINE → SIGNAL ENGINE → RISK ENGINE
       → portfolio analysis | bot advisor (selection, capital, duration,
         stop rules, leverage) | trade engine (demo/real)
       → STRUCTURED CONTEXT → LLM TRADING BUDDY → chat, dashboard → USER
```

## 4. Fundamental Architectural Principle

The chatbot and the prediction models must remain separate. The system must
NOT work as "BTC chart → LLM → BUY". Instead:

```text
MARKET DATA
  → quantitative models, technical-analysis algorithms, trend models,
    regime classifier, pattern detectors, order-flow models, portfolio
    engine, risk engine, bot-performance engine, bot-advisory engine
  → structured analysis
  → LLM trading assistant
  → natural-language explanation
```

The LLM is an analyst, interpreter, research assistant and
question-answering interface. It is never the sole source of trading
predictions.

## 5. Market Data Collector

Start with BTC/USDT, ETH/USDT, SOL/USDT and BNB/USDT; later XRP, ADA, DOGE,
AVAX, LINK and others. Collect: timestamp, open, high, low, close, volume,
number of trades, buyer volume, seller volume, bid price, ask price, spread,
order-book depth, individual trades. Historical data is for training; live
data is for real-time analysis.

## 6. Multiple Timeframes

1m, 5m, 15m, 1h, 4h, 1d. A market may look bearish on 15m, neutral on 1h,
bullish on 4h and strongly bullish on the daily. The Buddy should understand
this disagreement rather than collapse it into one signal, e.g.: "The broader
trend remains bullish, but short-term momentum is currently negative.
Entering immediately could expose you to a temporary pullback."

## 7. Historical Data Warehouse

Possible tables: symbols, candles, trades, order_book_snapshots,
technical_features, price_features, volume_features, pattern_features,
market_regimes, training_labels, model_predictions, signals, paper_trades,
live_trades, portfolio_snapshots, bot_definitions, bot_trades,
bot_positions, bot_performance, bot_recommendations, bot_market_suitability,
bot_capital_recommendations, bot_stop_conditions,
bot_leverage_recommendations, model_metrics, trading_metrics. PostgreSQL is
the initial primary database; add analytical storage later if needed.

## 8. Data-Quality Engine

Before models learn, check for: missing timestamps, duplicate candles, wrong
chronological order, impossible OHLC values, negative volume, missing prices,
unexpected gaps, corrupted records, duplicate trades, out-of-order streaming
events. Output per symbol: candles, duplicates, invalid, missing,
completeness, and a PASS / FAIL status. Critical failures prevent training.

## 9. Feature-Engineering Engine

Convert raw prices into machine-readable trading information: 1-, 3-, 6-,
12- and 24-period returns; rolling return; distance from recent high;
distance from recent low; price acceleration; range expansion; range
contraction.

## 10. Technical Indicators

EMA 9, 20, 50, 100 and 200; SMA; RSI; MACD; ATR; ADX; Bollinger Bands;
stochastic oscillator; rate of change.

Indicators do not generate trades. "RSI = 29" is information; it does not
mean BUY. The models must learn when an RSI of 29 matters and when it does
not.

## 11. Trend-Analysis Module

Understand market structure: higher highs, higher lows, lower highs, lower
lows, EMA direction, EMA slope, trend duration, trend strength, trend
acceleration, break of market structure. Example output: direction bullish,
strength strong, duration 26 candles, structure higher highs / higher lows.

## 12. Volatility Module

ATR, rolling standard deviation, historical volatility, Bollinger Band
width, high-low range, volatility percentile. Example: current volatility
HIGH, historical percentile 86%. The risk engine can reduce exposure when
volatility becomes excessive.

## 13. Volume Analysis

Current volume, average volume, volume ratio, volume change, volume Z-score,
buyer volume, seller volume, buy/sell ratio. A breakout on unusually high
volume may be treated differently from a weak-volume breakout.

## 14. Candlestick Analysis

Represent each candle mathematically: body size, upper wick, lower wick,
body/range ratio, opening position, closing position, relative candle size.
This makes traditional candlestick structures measurable rather than purely
visual.

## 15. Chart-Pattern Engine

Search for: double top, double bottom, head and shoulders, inverse head and
shoulders, ascending triangle, descending triangle, symmetrical triangle,
wedge, flag, channel, breakout, false breakout. Patterns are hypotheses, not
guaranteed trading signals.

## 16. Support and Resistance Engine

Identify significant price zones from swing highs, swing lows, price
clustering, pivot points, repeated rejection areas, volume profile and
rolling extrema. These levels become model features.

## 17. Market-Regime AI

A specialised model identifies the current environment: strong bullish, weak
bullish, sideways, weak bearish, strong bearish, low volatility, high
volatility, extreme volatility. Especially important for the Binance Bot
Advisor, because different automated strategies behave very differently
under different regimes.

## 18. Training Targets

Models answer precise questions, for example: "What is the probability BTC
rises at least 1% during the next four hours?", "What is the expected BTC
return during the next 24 hours?", or "Will BTC move UP, DOWN or SIDEWAYS?"
Different models can answer different questions.

## 19. Multiple Prediction Horizons

15-minute, 1-hour, 4-hour and 24-hour prediction models. The LLM can explain
the differences between them.

## 20. Baseline Models

Before advanced ML, build simple benchmarks: random model, buy and hold, EMA
crossover, RSI strategy, MACD strategy, logistic regression. If
sophisticated AI cannot outperform useful baselines after costs and risk
adjustment, it is not providing enough value.

## 21. Machine-Learning Layer

Early: logistic regression, random forest, XGBoost, LightGBM. Later:
temporal CNN, LSTM, GRU, time-series transformer. The winner is determined
by testing, not complexity.

## 22. Model Inputs

Possibly hundreds of features: RSI, MACD, EMA distances, EMA slopes, ATR,
ADX, historical returns, volume ratios, volatility, market regime, support
distance, resistance distance, candlestick properties, pattern detection,
order-book imbalance, trade flow. Output example: bullish 72%, bearish 18%,
neutral 10%.

## 23. Ensemble Engine

Multiple models contribute to a final opinion, e.g. trend model 82% bullish,
XGBoost 75%, momentum model 69%, order-flow model 71%, pattern model 61%,
combining into an overall bullish probability of 73%. Model weights could
later depend on historical performance under the current market regime.

## 24. Signal Engine

STRONG BUY, BUY, WAIT, SELL, STRONG SELL. WAIT is a first-class decision.
The system never trades simply because it is running.

## 25. Risk Engine

Separate from prediction. Controls whether trading is allowed, position
size, maximum account exposure, maximum loss, stop-loss rules, take-profit
rules, daily loss limit, weekly loss limit, portfolio concentration and
maximum simultaneous positions. It also supplies limits to the Bot Advisor.

## 26. Risk/Reward Engine

Each possible trade has entry, stop, target, potential loss, potential gain
and risk/reward ratio. For bots: expected return, expected drawdown, capital
at risk, liquidation risk, fees, funding costs, opportunity cost.

## 27. Position-Sizing Engine

Uses account balance, entry price, stop price and maximum risk to determine
position size. The LLM explains the result rather than inventing it.

## 28. Backtesting Engine

Simulates entries, exits, stops, take profits, fees, spread, slippage,
position sizing, portfolio value and multiple simultaneous positions. The
same framework can later backtest approximations of Binance bot strategies.

## 29. Financial Evaluation

Total return, annualised return, win rate, average win, average loss, profit
factor, Sharpe ratio, Sortino ratio, maximum drawdown, average trade, number
of trades, longest losing streak. The objective is useful risk-adjusted
performance.

## 30. Train / Validation / Test Separation

Time order is always preserved. Periods used for final testing are never
used for training or tuning.

## 31. Walk-Forward Testing

Repeatedly train on the past and test on later unseen periods, to check
whether performance survives changing market conditions.

## 32. Data-Leakage Protection

Every feature must satisfy `feature_timestamp <= prediction_timestamp`. No
future information may leak into model inputs.

## 33. Confidence Calibration

Predicted probabilities should match observed frequencies. The chatbot uses
calibrated probabilities whenever possible.

## 34. Binance Demo Trading

After historical testing: live market → models → signal → risk engine →
Binance demo → simulated order → result.

## 35. Trading Journal

Store every important recommendation with timestamp, symbol, signal,
confidence, model version, market regime, risk, proposed trade and outcome:
a permanent audit trail.

## 36. Self-Performance Analysis

Discover the market environments where the system performs well and badly,
so strategies can be enabled or disabled conditionally.

## 37. LLM Trading Chatbot

A conversational, tool-capable LLM (Mistral, Llama, an OpenAI model or
another) sits above the analytical platform as the primary interface between
the user and all the other systems. The model choice is deferred to
sub-project 8.

## 38. Information Given to the LLM

Structured facts, not raw guesses: current market conditions, model
predictions, market regime, portfolio, open positions, risk status,
support/resistance, bot performance, bot recommendations, capital
recommendations, leverage limits, historical similar situations.

## 39. LLM Internal Tools

get_market_price(), get_market_analysis(), get_model_predictions(),
get_market_regime(), get_support_resistance(), get_portfolio(),
get_open_positions(), get_risk_analysis(), get_trade_history(),
get_bot_performance(), get_bot_recommendation(),
get_bot_market_suitability(), calculate_bot_allocation(),
calculate_leverage_limits(), get_bot_stop_conditions(), compare_bots(),
compare_assets(), get_model_performance(), calculate_trade(). This makes the
LLM an orchestrator over the whole platform.

## 40. LLM Guardrails

The chatbot cannot override deterministic risk limits. It may explain what
the models and risk engines recommend; account-level restrictions stay in
deterministic software.

## 41. LLM Hallucination Protection

The chatbot must never invent prices, model probabilities,
support/resistance, portfolio balances, bot profits, recommended leverage,
liquidation prices, trade history or risk results. Numbers come from system
tools.

## 42–50. *[removed: image upload, image analysis, image training examples, vision model, user-taught image examples, image knowledge library]*

## 51. Binance Bot Monitoring System

Monitor the user's existing bots: grid, spot, futures, DCA, rebalancing,
custom and other automated strategies. Enter bots manually at first; sync
automatically later where supported.

## 52. Bot Registry

Each bot has a profile: name, type, symbol, starting capital, status,
strategy, leverage, risk classification, plus start price, current price,
configured range, grid count, initial investment, current equity, profit
target, stop condition, start market regime.

## 53. Bot Trade Monitoring

Per bot: orders, entries, exits, open positions, closed trades, fees,
funding costs, realised P&L, unrealised P&L, trade timestamps, position
sizes.

## 54. Bot Performance Analytics

Total, daily, weekly and monthly profit; return %; maximum drawdown; win
rate; average win; average loss; profit factor; number of trades; fees;
funding; capital deployed; return on deployed capital; risk-adjusted
return.

## 55. Bot Comparison

Compare bots with each other and with the AI strategy: return, drawdown and
profit per bot, including an "AI Trading Buddy" row.

## 56. AI Evaluation of Existing Bots

Analyse performance, not just display it. Example: "The bot is profitable
overall, but most of its profit came while BTC oscillated inside its
configured range. The market is now starting to trend strongly upward, which
may reduce the effectiveness of the current grid configuration."

## 57. Bot vs AI Analysis

Log disagreements between bots and the Trading Buddy, to reveal conditions
where particular bots outperform the AI or vice versa.

## 58. Bot Health Monitoring

Detect unexpected inactivity, unusually high loss rate, sudden
trade-frequency increases, excessive fees, excessive funding, drawdown above
normal, bot outside configured range, abnormal leverage risk.

## 59. Binance Bot Advisory Engine

Not a replacement for Binance bots. It helps decide which bot to use, when
to start it, which asset to run it on, how much capital to give it, how long
to leave it running, when to reduce capital, when to stop it, whether to
restart it, whether to change parameters, and how much leverage is
acceptable. Inputs: market-regime model, volatility model, trend model,
price prediction models, historical and current bot performance, portfolio
exposure, risk limits, fees, funding costs, liquidation risk, user's
available capital. Output goes to the LLM for explanation.

## 60. Bot Suitability Model

Each bot type gets a suitability score (e.g. BTCUSDT sideways, moderate
volatility, high range stability: Spot Grid 88/100, Futures Grid 71, DCA 52,
trend strategy 31). Scores come from historical performance and statistical
analysis, not arbitrary rules.

## 61. Bot Environment Learning

Learn: under which regimes a bot performs best; what volatility range
produces the best results; how it behaves in breakouts and strong trends;
when drawdown typically starts increasing; how long profitable runs last;
which conditions commonly appear before it becomes unprofitable. Example:
"BTC Grid Bot historically strongest when ADX < 20, moderate volatility,
price inside range, no strong directional breakout."

## 62. Recommended Bot Selection

"Which Binance bot should I start right now?" Analyse candidates (BTC Spot
Grid, ETH Spot Grid, SOL Futures Grid, BTC DCA, ETH DCA) and return the best
match, its suitability and the reasons; the LLM explains.

## 63. Bot Capital Allocation Engine

"How much money should I put in this bot?" Never an arbitrary number. Inputs:
total portfolio, available cash, current bot exposure, existing positions,
bot historical drawdown, current volatility, bot risk level, leverage,
correlation with other active bots, user-defined maximum risk. Output: a
recommended allocation range and the maximum under current risk rules.

## 64. Portfolio-Level Bot Allocation

Bots are not analysed independently. The portfolio engine understands total
crypto exposure, directional exposure, asset correlation, leverage exposure,
liquidation risk and available reserves, e.g. "The BTC bot itself is
acceptable, but adding another $5,000 would push total crypto exposure above
the portfolio risk limit."

## 65. Bot Duration Advisor

Duration is condition-based, not a fixed number of days: keep running while
price stays in range, ADX stays below threshold, volatility stays suitable,
drawdown stays below limit and net returns stay positive after fees; stop
earlier if any of these break.

## 66. Bot Stop Conditions

Every bot recommendation defines its invalidation conditions before
activation (e.g. strong close outside range, ADX above threshold, volatility
breakout, drawdown > X%, negative net profitability after costs, funding
above expected grid profit, regime change to strong bullish/bearish). The
system monitors them continuously.

## 67. Bot Stop Recommendation

"Should I stop my BTC bot?" Analyse the original thesis, current regime,
current range, P&L, drawdown, recent trade efficiency, fees and trend
strength, then recommend (e.g. STOP / PAUSE) with reasons; the LLM explains.

## 68. Bot Thesis Tracking

Store why each bot was launched (regime, range, volatility, expected
duration, model confidence), and later ask whether that reason still holds,
so bots don't keep running just because they were switched on.

## 69. Bot Lifecycle

CANDIDATE → RECOMMENDED → RUNNING → HEALTHY → WATCH → STOP RECOMMENDED →
STOPPED → REVIEW, visible on the dashboard.

## 70. Futures Bot Leverage Advisor

The question is not the maximum leverage Binance allows, but what leverage
is compatible with expected volatility, drawdown, liquidation distance,
portfolio exposure and acceptable loss. Inputs: asset volatility, ATR,
expected and historical worst-case bot drawdown, liquidation distance,
maintenance margin, position size, total portfolio, other leveraged
positions, market regime, funding costs.

## 71. Leverage Recommendations

Per leverage level, a risk verdict (e.g. 1x low, 2x acceptable, 3x elevated,
5x excessive) and a recommended maximum with the reason. The value comes
from quantitative risk rules, never from the LLM guessing.

## 72. Leverage Stress Testing

Simulate −3%, −5% and −10% moves, historical volatility shocks, flash
crashes, trend breakouts and funding increases; compute estimated P&L,
margin remaining, distance to liquidation, portfolio loss and maximum
acceptable leverage.

## 73. Bot Risk Score

A 0–100 risk score and classification per active or proposed bot, from
volatility, leverage, regime mismatch, historical drawdown, capital
allocation, correlation, liquidity, funding and range-break probability.

## 74. Bot Opportunity Score

A separate 0–100 opportunity score, so the system looks for high opportunity
with acceptable risk rather than the highest expected return.

## 75. Bot Ranking System

"Rank the best Binance bots for me right now" returns bots ranked with
opportunity and risk scores; the LLM explains why, for example, a
lower-opportunity, low-risk bot may be preferable to a high-opportunity,
high-risk one.

## 76. Bot Simulation Before Activation

Simulate a proposed configuration (capital, range, grid count) over similar
historical periods: number of comparable periods, share profitable, average
return, median and worst drawdown.

## 77. Bot Recommendation History

Store every recommendation (date, bot, action, capital, expected regime,
stop condition) and its actual result, to measure whether the Bot Advisor is
useful.

## 78. Bot Advisor Accuracy

Metrics: recommended bots, profitable recommendations, average return,
average drawdown, correct stop recommendations, capital allocation
efficiency, leverage recommendation performance. E.g. "When the advisor
scores a bot above 80, how often does it outperform doing nothing?"

## 79. Bot Recommendation Dashboard

A "Bot Opportunities" page: per candidate bot, suitability, risk, suggested
allocation, suggested duration (condition-based), leverage and status
(RECOMMENDED / WATCH).

## 80. Active Bot Advisor Dashboard

For running bots: profit, runtime, current vs original regime, health and
recommendation (CONTINUE / CONSIDER STOPPING).

## 81. Main Dashboard

Home page: portfolio value, today's and this week's P&L, active bots, bot
health, bot recommendations, bot capital deployed, total leveraged exposure,
open positions, current AI signals, market regime, account risk, model
health, best current opportunities.

## 82. Portfolio Dashboard

Total portfolio, available cash, holdings per asset, allocation %,
unrealised and realised P&L, capital allocated to bots, leveraged exposure.

## 83. Bot Dashboard

Active and paused bots, capital, profit (today, week, month, total), open
positions, recent trades, risk, drawdown, suitability, regime compatibility,
recommended action, suggested stop condition, current leverage, plus a
detail page per bot.

## 84. AI Signals Dashboard

Quantitative trading signals shown separately from bot recommendations.

## 85. Market Dashboard

Per market: candlestick chart, indicators, support, resistance, volume,
volatility, market regime, model predictions, open positions, bot activity,
suggested bot types, AI commentary.

## 86. Profit Dashboard

"How much money am I actually making?" Broken down by manual trades, Binance
bots, the AI Trading Buddy, per asset, fees and funding costs.

## 87. Model Performance Dashboard

Current production model, version, recent prediction accuracy, calibration,
profit factor, maximum drawdown, and performance by asset, timeframe and
regime.

## 88–90. *[removed: AI learning dashboard for images, upload interface, image annotation tools]*

## 91. Trading Chat

The chat is the main way to interact with all systems: best setup now; which
bot to use; whether a bot is still appropriate; how much to put in a bot;
whether to stop a bot; whether a leverage level is too much; bot profits;
worst-performing bot; why a bot is losing; bot vs AI comparison.

## 92. Chatbot Context

Before answering, the assistant may retrieve market data, model predictions,
portfolio, open positions, risk limits, bot performance, bot configuration,
bot suitability, bot start thesis, leverage exposure, trade journal and
historical similar situations.

## 93. *[removed: image and market-data linking]*

## 94. Training Pipeline

New examples enter a controlled dataset-versioning and model-validation
pipeline before affecting production.

## 95. Model Versioning

Every important model is versioned (e.g. xgboost_v4, ensemble_v6,
regime_model_v3, bot_suitability_v2, bot_stop_model_v1,
leverage_risk_model_v3, trading_strategy_v7).

## 96. *[removed: testing the vision AI]*

## 97. LLM Test Suite

Correct use of model results, risk-rule compliance, bot-performance
interpretation, bot-recommendation interpretation, capital-allocation
reasoning, leverage-limit compliance, missing-information handling,
consistency, hallucination resistance.

## 98. Bot Advisor Tests

Historical scenarios with known regime and bot outcome: in a strong sideways
market the advisor ranks a Spot Grid above a trend bot; in a strong
directional breakout it downgrades or recommends stopping the grid.

## 99. Capital Allocation Tests

The same bot recommendation must produce different allocations depending on
portfolio conditions (e.g. low vs already-high bot exposure on a $50,000
portfolio).

## 100. Leverage Tests

Very high volatility with proposed 10x leverage must not be approved because
expected returns are high. Stress-loss and liquidation calculations
dominate.

## 101. Stress Testing

Fail safely on: Binance unavailable, WebSocket disconnect, database
unavailable, LLM unavailable, stale prices, duplicate messages, extreme
volatility, incomplete bot data, unexpected liquidation risk.

## 102. Technology Stack

Quant/AI: Python, pandas, NumPy, scikit-learn, XGBoost, LightGBM, PyTorch.
Backend: FastAPI. Database: PostgreSQL. Cache: Redis. Live market data:
Binance WebSockets. Frontend: React or Next.js (details deferred). LLM layer:
provider-independent abstraction (Mistral, Llama, OpenAI model, future LLM).

## 103. Proposed Project Structure

```text
trading-buddy/
├── data/        collectors/, cleaning/, storage/
├── features/    indicators/, price/, volume/, volatility/, candles/, patterns/
├── models/      baseline/, trend/, regime/, prediction/, bot_suitability/,
│                bot_duration/, leverage_risk/, ensemble/
├── training/    datasets/, bot_datasets/, pipelines/, validation/, calibration/
├── bots/        registry/, monitoring/, performance/, suitability/, allocation/,
│                duration/, stop_rules/, leverage/, comparison/
├── backtesting/  signals/  risk/  portfolio/
├── execution/   demo/, live/
├── journal/  monitoring/
├── assistant/   llm/, tools/, prompts/, context/, guardrails/
├── api/
├── frontend/    dashboard/, markets/, bots/, bot_advisor/, chat/, portfolio/
├── tests/
└── infrastructure/
```

## 104. Main Application Pages

Overview, Markets, AI Chat, Portfolio, Active Trades, Binance Bots, Bot
Advisor, Bot Comparison, AI Signals, Profit Analytics, Trading Journal, Model
Performance, Risk Center, Leverage Center, System Health.

## 105. Milestone 1 — Foundation

Python project, database, Binance connectivity, historical data collector,
live data collector, data validation. Goal: trustworthy market-data
infrastructure.

## 106. Milestone 2 — Market Intelligence

Indicators, trend detection, volume analysis, volatility, candlestick
features, support/resistance, patterns, market regimes. Goal: transform
market history into machine-understandable information.

## 107. Milestone 3 — First Prediction AI

Training datasets, labels, baseline models, XGBoost, validation,
walk-forward testing, calibration. Goal: statistically measurable
predictions.

## 108. Milestone 4 — Backtesting

Historical simulator, fees, spread, slippage, stops, targets, portfolio
simulation, performance reports. Goal: determine whether predictions can
produce useful strategies.

## 109. Milestone 5 — Trading Intelligence

Multiple prediction horizons, ensemble engine, signal engine, risk engine,
position sizing, portfolio engine. Goal: complete trading recommendations.

## 110. Milestone 6 — Binance Bot Monitoring

Bot registry, bot data import, trade tracking, profit calculation, bot
metrics, bot comparison. Goal: monitor automated systems already running on
Binance.

## 111. Milestone 7 — Binance Bot Advisor

Bot suitability model, market-to-bot matching, bot ranking,
capital-allocation engine, bot duration logic, stop-condition engine, bot
thesis tracking, leverage-risk engine, bot recommendation history. Goal:
answer which bot, which asset, how much money, how much leverage, how long,
and when to stop.

## 112. Milestone 8 — LLM Chatbot

LLM service, tool calling, context builder, conversation system, risk
guardrails, hallucination protection, bot-advisor tools. Goal: one
conversational interface for the whole platform.

## 113. Milestone 9 — Dashboard

Overview, portfolio, bot monitoring, Bot Advisor, profit analytics, market
pages, AI signals, journal, model analytics, risk dashboard, leverage
dashboard. Goal: the main operating environment.

## 114. *[removed: Milestone 10 — vision and image learning]*

## 115. Milestone 11 — Binance Demo Trading

Run the complete system against live market data with virtual funds. The
Bot Advisor can also be evaluated by simulating recommended bot
configurations.

## 116. Milestone 12 — Shadow Production

The production system generates trade, bot, stop, capital and leverage
recommendations but sends no real orders.

## 117. Milestone 13 — Controlled Real Deployment

Only after all previous testing passes: very small capital, strict risk
limits, manual confirmation, full logging, kill switch.

## 118. Continuous Learning

New market data, trades, bot performance, bot recommendation results and
prediction results feed future datasets. Production models never update
themselves; every new model passes the testing pipeline again.

## 119. What the Final System Feels Like

A dashboard showing portfolio, today's P&L, active bots with health, the Bot
Advisor's best new opportunity (with suitability, suggested allocation,
leverage and expected mode), the Buddy's market view and the risk state.
In chat, "What should I do right now?" gets a reasoned answer covering which
bots to keep running, which to watch, where new capital fits best and how
much could be allocated within the configured limits. A follow-up like
"What if I want to use 5x leverage?" is answered from the leverage engine's
stress results, not from the LLM's opinion.

## 120. Final Philosophy

Several forms of intelligence behind one interface:

- **Quantitative:** what does the data predict? (ML models)
- **Historical:** what happened in similar situations? (historical database)
- **Risk:** how much risk should we take? (risk engine)
- **Portfolio:** how does this affect everything I own and every running bot?
  (portfolio engine)
- **Bot:** how are my bots performing? (monitoring)
- **Bot advisory:** which bot, how much, how long, when to stop, what
  leverage? (Bot Advisor)
- **Conversational:** what does it all mean? (LLM)
- **Learning:** can the system improve from new data and outcomes? (training
  pipeline)

The Binance Bot Advisor is one more intelligence layer inside the platform:
the same infrastructure that understands the market also helps manage the
user's automated strategies.
