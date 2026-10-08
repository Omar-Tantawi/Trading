// Read-only dashboard page. All text is set with textContent, never HTML.
"use strict";

const $ = (id) => document.getElementById(id);

async function getJSON(url) {
  const r = await fetch(url);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) {
    const d = body.detail;
    throw new Error(typeof d === "string" ? d : `${r.status} ${r.statusText}`);
  }
  return body;
}

function cell(row, text, cls) {
  const td = document.createElement("td");
  td.textContent = text == null ? "-" : String(text);
  if (cls) td.className = cls;
  row.appendChild(td);
  return td;
}

function pct(v) { return v == null ? "-" : `${v >= 0 ? "+" : ""}${(v * 100).toFixed(1)}%`; }

function age(minutes) {
  if (minutes == null) return "-";
  const d = Math.floor(minutes / 1440), h = Math.floor((minutes % 1440) / 60), m = minutes % 60;
  return d ? `${d}d ${h}h` : h ? `${h}h ${m}m` : `${m}m`;
}

// ---- tabs
const loaded = {};
document.querySelectorAll("nav .tab").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll("nav .tab").forEach((x) => x.classList.toggle("active", x === b));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === b.dataset.tab));
  if (!loaded[b.dataset.tab]) { loaded[b.dataset.tab] = true; LOADERS[b.dataset.tab](); }
}));

// ---- chart tab
const LC = window.LightweightCharts;
let priceChart, rsiChart, candleSeries, emaSeries, rsiSeries;

function makeCharts() {
  // autoSize follows the window; the same price-axis width on both charts
  // keeps the candles and the RSI points lined up.
  const opts = { autoSize: true, rightPriceScale: { minimumWidth: 70 },
                 layout: { background: { color: "#ffffff" }, textColor: "#222" },
                 timeScale: { timeVisible: true, secondsVisible: false },
                 localization: { timeFormatter: (t) => new Date(t * 1000).toISOString().slice(0, 16).replace("T", " ") } };
  priceChart = LC.createChart($("price-chart"), opts);
  rsiChart = LC.createChart($("rsi-chart"), opts);
  candleSeries = priceChart.addSeries(LC.CandlestickSeries, {});
  emaSeries = priceChart.addSeries(LC.LineSeries, { color: "#f39c12", lineWidth: 2, title: "EMA 200" });
  // Fixed 0-100 scale so the 30 and 70 guide lines are always visible.
  rsiSeries = rsiChart.addSeries(LC.LineSeries, {
    color: "#8e44ad", lineWidth: 1, title: "RSI 14",
    autoscaleInfoProvider: () => ({ priceRange: { minValue: 0, maxValue: 100 } }) });
  for (const level of [30, 70]) {
    rsiSeries.createPriceLine({ price: level, color: "#999", lineStyle: 2, lineWidth: 1 });
  }
  // keep both time axes in step
  priceChart.timeScale().subscribeVisibleLogicalRangeChange((r) => r && rsiChart.timeScale().setVisibleLogicalRange(r));
  rsiChart.timeScale().subscribeVisibleLogicalRangeChange((r) => r && priceChart.timeScale().setVisibleLogicalRange(r));
}

const line = (bars, key) => bars.map((b) => (b[key] == null ? { time: b.time } : { time: b.time, value: b[key] }));

let candleRequest = 0;
async function loadCandles() {
  const mine = ++candleRequest;
  const symbol = $("symbol").value, tf = $("timeframe").value;
  const status = $("chart-status");
  status.className = "status";
  status.textContent = "loading ...";
  try {
    const data = await getJSON(`/api/candles?symbol=${encodeURIComponent(symbol)}&timeframe=${encodeURIComponent(tf)}&bars=500`);
    if (mine !== candleRequest) return;   // a newer request has started
    candleSeries.setData(data.bars.map(({ time, open, high, low, close }) => ({ time, open, high, low, close })));
    emaSeries.setData(line(data.bars, "ema_200"));
    rsiSeries.setData(line(data.bars, "rsi_14"));
    priceChart.timeScale().fitContent();
    const l = data.latest;
    if (!data.bars.length) {
      status.textContent = `${symbol} ${tf}: no candles stored yet (run tb backfill).`;
    } else if (!l) {
      status.textContent = `${symbol} ${tf}: no features built yet (run tb features build).`;
    } else {
      const ageMin = Math.floor((Date.now() - Date.parse(l.bar_close)) / 60000);
      const adx = l.adx_14 == null ? "-" : l.adx_14.toFixed(1);
      const vp = l.vol_pct_365d == null ? "-" : l.vol_pct_365d.toFixed(0);
      status.textContent = `newest bar closed ${l.bar_close.slice(0, 16).replace("T", " ")} UTC (${age(ageMin)} ago)`
        + ` · trend ${l.trend_regime || "-"} (ADX ${adx}) · volatility ${l.volatility_regime || "-"} (percentile ${vp})`;
    }
  } catch (e) {
    if (mine !== candleRequest) return;
    status.className = "status error";
    status.textContent = `Could not load: ${e.message}`;
  }
}

async function loadChartTab() {
  const s = await getJSON("/api/symbols");
  for (const [id, values, pick] of [["symbol", s.symbols, s.symbols[0]], ["timeframe", s.timeframes, "1h"]]) {
    for (const v of values) { const o = document.createElement("option"); o.value = o.textContent = v; $(id).appendChild(o); }
    $(id).value = pick;
    $(id).addEventListener("change", loadCandles);
  }
  $("refresh").addEventListener("click", loadCandles);
  makeCharts();
  await loadCandles();
}

// ---- models tab
async function loadModels() {
  const body = $("runs").querySelector("tbody");
  body.textContent = "";
  try {
    const runs = await getJSON("/api/runs");
    if (!runs.length) { const r = body.insertRow(); cell(r, "no runs yet (tb ml evaluate)"); return; }
    for (const run of runs.slice().reverse()) {
      const r = body.insertRow();
      r.className = "clickable";
      cell(r, run.run_id); cell(r, run.target); cell(r, run.kind); cell(r, `${run.horizon}h`);
      cell(r, run.created_at.slice(0, 16).replace("T", " ")); cell(r, run.n_predictions.toLocaleString());
      cell(r, pct(run.xgb_skill));
      r.addEventListener("click", () => showRun(run.run_id));
    }
  } catch (e) { const r = body.insertRow(); cell(r, `Could not load: ${e.message}`, "stale"); }
}

let currentRun = null;
async function showRun(id) {
  currentRun = id;
  $("run-detail").hidden = false;
  $("run-title").textContent = `Run ${id}: loading ...`;
  $("run-report").textContent = "";
  $("diagnosis").textContent = "";
  $("run-charts").textContent = "";
  let run;
  try {
    run = await getJSON(`/api/runs/${id}`);
  } catch (e) {
    if (currentRun === id) $("run-title").textContent = `Run ${id}: could not load: ${e.message}`;
    return;
  }
  if (currentRun !== id) return;          // another run was clicked meanwhile
  $("run-title").textContent = `Run ${id}: ${run.target}, next ${run.horizon}h (${run.kind})`;
  $("diagnose-row").hidden = run.target !== "move3";   // vol3/dir2 already split the question
  $("run-report").textContent = run.report;
  $("diagnosis").textContent = "";
  const charts = $("run-charts");
  charts.textContent = "";
  for (const model of Object.keys(run.metrics.models)) {
    if (model === "base_rate_v1") continue;
    for (const kind of ["skill", "calibration"]) {
      const img = document.createElement("img");
      img.src = `/api/runs/${id}/${kind}/${encodeURIComponent(model)}.svg`;
      img.alt = `${model} ${kind}`;
      charts.appendChild(img);
    }
  }
}

$("diagnose").addEventListener("click", async () => {
  const out = $("diagnosis"), button = $("diagnose"), id = currentRun;
  out.textContent = "computing ...";
  button.disabled = true;
  try {
    const d = await getJSON(`/api/runs/${id}/diagnose`);
    if (id !== currentRun) return;
    const t = document.createElement("table");
    const h = t.insertRow();
    for (const name of ["model", "size skill (move vs flat)", "95% CI", "direction skill (moves only)", "95% CI", "side right"]) cell(h, name);
    for (const [model, s] of Object.entries(d)) {
      const r = t.insertRow();
      cell(r, model);
      cell(r, pct(s.size.skill)); cell(r, `${pct(s.size.skill_lo)} to ${pct(s.size.skill_hi)}`);
      cell(r, pct(s.direction.skill)); cell(r, `${pct(s.direction.skill_lo)} to ${pct(s.direction.skill_hi)}`);
      cell(r, `${(s.direction.accuracy * 100).toFixed(1)}% (base ${(s.direction.base_accuracy * 100).toFixed(1)}%)`);
    }
    out.textContent = "";
    out.appendChild(t);
  } catch (e) { if (id === currentRun) out.textContent = `Could not compute: ${e.message}`; }
  finally { button.disabled = false; }
});

// ---- health tab
async function loadHealth() {
  const body = $("health-table").querySelector("tbody");
  body.textContent = "";
  $("health-status").textContent = `checked ${new Date().toISOString().slice(0, 16).replace("T", " ")} UTC`;
  try {
    for (const s of await getJSON("/api/health")) {
      const r = body.insertRow();
      cell(r, s.symbol);
      cell(r, s.last_1m ? s.last_1m.slice(0, 16).replace("T", " ") : "no candles");
      cell(r, s.stale ? `${age(s.age_minutes)} STALE` : age(s.age_minutes), s.stale ? "stale" : "");
      for (const tf of ["5m", "15m", "1h", "4h", "1d"]) {
        cell(r, s.features[tf] ? s.features[tf].slice(0, 16).replace("T", " ") : "not built");
      }
      cell(r, s.verdict ? `${s.verdict} (${s.verdict_at.slice(0, 10)})` : "never checked");
    }
  } catch (e) { $("health-status").textContent = `Could not load: ${e.message}`; }
}

const LOADERS = { chart: loadChartTab, models: loadModels, health: loadHealth };
loaded.chart = true;
loadChartTab().catch((e) => { $("chart-status").className = "status error"; $("chart-status").textContent = `Could not load: ${e.message}`; });
