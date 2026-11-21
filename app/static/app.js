"use strict";

const $ = (id) => document.getElementById(id);
const state = { meta: null, lastPrediction: null };
const fmt = (value, digits = 4) => (typeof value === "number" && Number.isFinite(value)
  ? value.toFixed(digits) : "—");

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

async function api(path, options) {
  const response = await fetch(path, options);
  const text = await response.text();
  let data;
  try { data = text ? JSON.parse(text) : {}; } catch { data = { error: text }; }
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

/* ---------------- inputs ---------------- */
function bindPair(rangeId, numberId, outId, format) {
  const range = $(rangeId), number = $(numberId), out = $(outId);
  const sync = (value, source) => {
    const v = Math.max(Number(range.min), Math.min(Number(range.max), Number(value) || 0));
    if (source !== "range") range.value = v;
    if (source !== "number") number.value = v;
    out.textContent = format(v);
  };
  range.addEventListener("input", () => sync(range.value, "range"));
  number.addEventListener("input", () => sync(number.value, "number"));
  sync(range.value, "range");
}

function inputs() {
  return {
    study_hours: Number($("hours").value),
    attendance: Number($("attendance").value),
    participation: Number($("participation").value),
  };
}

/* ---------------- prediction ---------------- */
function renderPrediction(result) {
  state.lastPrediction = result;
  $("scoreValue").textContent = result.predicted_total_score.toFixed(1);
  $("scoreBar").style.width = `${Math.max(2, Math.min(100, result.predicted_total_score))}%`;
  $("scoreRange").textContent =
    `Expected range ${result.expected_range[0].toFixed(1)} – ${result.expected_range[1].toFixed(1)} ` +
    `(typical test error ${(result.expected_range[1] - result.expected_range[0]) / 2 > 0 ? "±" + ((result.expected_range[1] - result.expected_range[0]) / 2).toFixed(1) : ""} points)`;

  const badge = $("gradeBadge");
  badge.textContent = result.predicted_grade;
  badge.className = `grade-badge ${result.predicted_grade.toLowerCase()}`;
  $("gradeConfidence").textContent =
    `${Math.round(result.grade_confidence * 100)}% grade confidence · ${result.performance_band}`;
  $("interpretation").textContent = result.interpretation;

  const probs = $("probs");
  probs.replaceChildren();
  Object.entries(result.grade_probabilities).forEach(([label, value]) => {
    const row = el("div", "prob");
    row.append(el("b", "", label));
    const track = el("div", "mini-bar");
    const fill = el("span");
    fill.style.width = `${Math.max(1, value * 100)}%`;
    track.append(fill);
    row.append(track, el("em", "", `${(value * 100).toFixed(1)}%`));
    probs.append(row);
  });

  const list = $("drivers");
  list.replaceChildren();
  const maxImportance = Math.max(...result.drivers.map((d) => Math.abs(d.importance_rmse_points)), 0.0001);
  result.drivers.forEach((driver) => {
    const li = el("li");
    li.append(el("span", "", driver.feature.replaceAll("_", " ")));
    const track = el("div", "track");
    const fill = el("span");
    fill.style.width = `${Math.min(100, (Math.abs(driver.importance_rmse_points) / maxImportance) * 100)}%`;
    track.append(fill);
    li.append(track);
    li.append(el("small", "", `${driver.importance_rmse_points.toFixed(4)} RMSE points · ${driver.level}`));
    list.append(li);
  });

  const tbody = $("whatIf").querySelector("tbody");
  tbody.replaceChildren();
  result.suggestions.forEach((s) => {
    const tr = el("tr");
    tr.append(el("td", "", `${s.lever.replaceAll("_", " ")}: ${s.from.toFixed(1)} → ${s.to.toFixed(1)}`));
    tr.append(el("td", "", s.predicted_score.toFixed(2)));
    const delta = el("td", Math.abs(s.predicted_change) < 0.2 ? "zero" : "pos",
      `${s.predicted_change >= 0 ? "+" : ""}${s.predicted_change.toFixed(2)}`);
    tr.append(delta);
    tbody.append(tr);
  });
}

async function predict(event) {
  if (event) event.preventDefault();
  const button = $("predictBtn");
  const error = $("predictError");
  error.hidden = true;
  button.disabled = true;
  button.textContent = "Predicting…";
  try {
    const result = await api("/api/predict", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(inputs()),
    });
    renderPrediction(result);
  } catch (err) {
    error.hidden = false;
    error.textContent = `Prediction failed: ${err.message}`;
  } finally {
    button.disabled = false;
    button.textContent = "Predict performance";
  }
}

/* ---------------- chat ---------------- */
function addBubble(role, text, meta) {
  const bubble = el("div", `bubble ${role}`);
  const paragraph = el("p", "", text);
  bubble.append(paragraph);
  if (meta) bubble.append(meta);
  $("chat").append(bubble);
  $("chat").scrollTop = $("chat").scrollHeight;
  return bubble;
}

function metaLine(parts) {
  const wrap = el("div", "meta");
  parts.filter(Boolean).forEach((part) => wrap.append(el("span", "", part)));
  return wrap;
}

async function ask(question) {
  if (!question.trim()) return;
  addBubble("user", question);
  const thinking = addBubble("assistant", "Thinking…");
  const button = $("askBtn");
  button.disabled = true;
  try {
    const result = await api("/api/ask", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, context: inputs() }),
    });
    const isLlm = result.source.startsWith("ollama");
    const sourceLabel = !isLlm
      ? "deterministic engine (trained model)"
      : (result.source === "ollama+tools"
        ? `Ollama (${result.ollama.model}) + trained model tools`
        : `Ollama (${result.ollama.model}) + trained model facts`);
    const tools = result.tool_calls && result.tool_calls.length
      ? `tools: ${result.tool_calls.map((c) => c.name).join(", ")}` : null;
    const meta = metaLine([`evidence: ${result.evidence}`, `source: ${sourceLabel}`, tools]);
    const bubble = el("div", "bubble assistant");
    bubble.append(el("p", "", result.answer));
    bubble.append(meta);
    if (result.source.startsWith("ollama")) {
      const facts = el("div", "facts");
      facts.textContent = "Authoritative output from the trained model:\n" +
        (result.details && result.details.predicted_total_score !== undefined
          ? `predicted total score ${result.details.predicted_total_score} / 100 · grade ` +
            `${result.details.predicted_grade} · grade confidence ` +
            `${Math.round(result.details.grade_confidence * 100)}%`
          : (result.deterministic_answer || "see the Dataset & model tab for measured statistics"));
      bubble.append(facts);
    }
    if (result.audit && result.audit.length) {
      bubble.append(el("div", "facts", `Numbers not found in model output: ${result.audit.join(", ")}`));
    }
    thinking.replaceWith(bubble);
    if (result.note) $("assistantMode").textContent = result.note;
  } catch (err) {
    thinking.replaceWith(el("div", "bubble assistant"));
    $("chat").lastElementChild.append(el("p", "", `Request failed: ${err.message}`));
  } finally {
    button.disabled = false;
    $("chat").scrollTop = $("chat").scrollHeight;
  }
}

/* ---------------- meta panels ---------------- */
function renderMeta(meta) {
  state.meta = meta;
  const m = meta.metrics;
  const cards = [
    { label: "MAE (test split)", value: fmt(m.test_metrics.mae, 3), sub: `${(m.test_metrics.n_samples || 0).toLocaleString()} unseen students` },
    { label: "RMSE (test split)", value: fmt(m.test_metrics.rmse, 3), sub: "score points" },
    { label: "R² (test split)", value: fmt(m.test_metrics.r2, 4), sub: `trained on ${(m.trained_on_rows || 0).toLocaleString()} rows` },
    { label: "Grade accuracy", value: fmt(m.grade_model.accuracy, 4), sub: `weighted F1 ${fmt(m.grade_model.f1_weighted, 4)}` },
    { label: "ROC-AUC (grade)", value: fmt(m.grade_model.roc_auc_ovr_macro, 4), sub: "one-vs-rest macro" },
    { label: "Exams model R²", value: fmt(m.exams_model.r2, 4), sub: `MAE ${fmt(m.exams_model.mae, 2)} (secondary)` },
  ];
  const container = $("metricCards");
  container.replaceChildren();
  cards.forEach((card) => {
    const box = el("div", "metric");
    box.append(el("span", "", card.label), el("b", "", card.value), el("span", "", card.sub));
    container.append(box);
  });

  const maxImp = Math.max(...meta.importance.map((r) => Math.abs(r.importance_rmse_points)), 0.0001);
  const importance = $("importanceList");
  importance.replaceChildren();
  meta.importance.forEach((row) => {
    const line = el("div");
    line.append(el("span", "", row.feature.replaceAll("_", " ")));
    const track = el("div", "track");
    const fill = el("span");
    fill.style.width = `${Math.min(100, (Math.abs(row.importance_rmse_points) / maxImp) * 100)}%`;
    track.append(fill);
    line.append(track, el("em", "", row.importance_rmse_points.toFixed(4)));
    importance.append(line);
  });

  const stats = meta.dataset;
  const kv = $("datasetSummary");
  kv.replaceChildren();
  const items = stats.available ? [
    ["Test rows", stats.rows.toLocaleString()],
    ["Mean score", stats.mean_score.toFixed(2)],
    ["Median score", stats.median_score.toFixed(2)],
    ["Standard deviation", stats.std_score.toFixed(2)],
    ["Range", `${stats.min_score} – ${stats.max_score}`],
    ["Scores exactly 100", `${(stats.share_scoring_100 * 100).toFixed(1)}%`],
    ["corr(study hours, score)", stats.correlation_with_score.weekly_self_study_hours.toFixed(3)],
    ["corr(attendance, score)", stats.correlation_with_score.attendance_percentage.toFixed(3)],
    ["corr(participation, score)", stats.correlation_with_score.class_participation.toFixed(3)],
  ] : [["Dataset", stats.reason || "unavailable"]];
  items.forEach(([key, value]) => {
    const box = el("div");
    box.append(el("span", "", key), el("b", "", String(value)));
    kv.append(box);
  });

  const limits = $("limitsList");
  limits.replaceChildren();
  meta.limitations.forEach((item) => limits.append(el("li", "", item)));

  const chipModel = $("chipModel");
  chipModel.textContent = `model: ${m.primary_model}`;
  chipModel.className = "chip ok";
  $("footerModel").textContent = `${m.feature_count || 6} features · target ${m.target}`;
}

function renderOllama(status) {
  const chip = $("chipOllama");
  if (status.available && status.model) {
    chip.textContent = `ollama: ${status.model}`;
    chip.className = "chip ok";
    $("assistantMode").textContent =
      "Answers are written by the local Ollama model using tool results from the trained model.";
  } else {
    chip.textContent = "ollama: offline";
    chip.className = "chip warn";
    $("assistantMode").textContent =
      "Ollama is not running, so answers come from the deterministic engine that reads the dataset and calls the trained model.";
  }
}

/* ---------------- wiring ---------------- */
function initTabs() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
      document.querySelectorAll(".panel").forEach((panel) => {
        panel.classList.toggle("active", panel.id === `panel-${tab.dataset.panel}`);
      });
    });
  });
}

function init() {
  bindPair("hours", "hoursNum", "hoursOut", (v) => v.toFixed(1));
  bindPair("attendance", "attendanceNum", "attendanceOut", (v) => v.toFixed(0));
  bindPair("participation", "participationNum", "participationOut", (v) => v.toFixed(1));

  $("studentForm").addEventListener("submit", predict);
  $("resetBtn").addEventListener("click", () => {
    $("hours").value = 15; $("attendance").value = 85; $("participation").value = 6.5;
    ["hours", "attendance", "participation"].forEach((id) => $(id).dispatchEvent(new Event("input")));
  });

  $("askForm").addEventListener("submit", (event) => {
    event.preventDefault();
    const question = $("askInput").value;
    $("askInput").value = "";
    ask(question);
  });
  document.querySelectorAll("#suggestions .pill").forEach((pill) => {
    pill.addEventListener("click", () => ask(pill.textContent));
  });

  $("examsForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const payload = Object.fromEntries(new FormData(event.target).entries());
    const out = $("examsResult");
    out.textContent = "Scoring…";
    try {
      const result = await api("/api/exams", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      out.textContent = `Predicted maths score: ${result.predicted_math_score.toFixed(2)} / 100 ` +
        `· category: ${result.predicted_category} · model: ${result.model}. ${result.note}`;
    } catch (err) {
      out.textContent = `Scoring failed: ${err.message}`;
    }
  });

  initTabs();
  api("/api/meta").then((meta) => {
    renderMeta(meta);
    renderOllama(meta.ollama);
  }).catch((err) => {
    $("chipModel").textContent = `model: error (${err.message})`;
    $("chipModel").className = "chip warn";
  });
  api("/api/health").then((health) => renderOllama(health.ollama))
    .catch(() => renderOllama({ available: false }));
  predict();
}

document.addEventListener("DOMContentLoaded", init);
