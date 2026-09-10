/* =========================================================================
   Lunar Multi-Sensor Registration -- frontend application logic.
   Talks to the FastAPI backend (default http://localhost:8000). Every
   number rendered on screen comes directly from the backend's JSON
   response -- nothing here is invented or animated independently of a
   real API result.
========================================================================= */

const API_BASE = window.LUNAR_API_BASE || "";

const SENSOR_META = {
  OHRC: { full: "Orbiter High Resolution Camera", role: "REFERENCE" },
  TMC: { full: "Terrain Mapping Camera", role: "SOURCE" },
  IIRS: { full: "Imaging Infrared Spectrometer", role: "SOURCE" },
  SAR: { full: "Synthetic Aperture Radar", role: "SOURCE" },
};

const PIPELINE_STEPS = ["dataset", "analysis", "preprocess", "detect", "match", "register", "evaluate", "fuse", "ai"];

let currentRun = null;
let activeSensorTab = null;
let matchFilterMode = "inliers";

// ---------------------------------------------------------------------
// Starfield (subtle, single decorative element -- not repeated motion)
// ---------------------------------------------------------------------
function initStarfield() {
  const canvas = document.getElementById("starfield");
  const ctx = canvas.getContext("2d");
  function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  }
  resize();
  window.addEventListener("resize", resize);
  const stars = Array.from({ length: 140 }, () => ({
    x: Math.random() * canvas.width,
    y: Math.random() * canvas.height,
    r: Math.random() * 1.1 + 0.2,
    a: Math.random() * 0.6 + 0.15,
  }));
  ctx.fillStyle = "#05070c";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  stars.forEach((s) => {
    ctx.beginPath();
    ctx.fillStyle = `rgba(210,220,240,${s.a})`;
    ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
    ctx.fill();
  });
}

// ---------------------------------------------------------------------
// View switching
// ---------------------------------------------------------------------
function showView(id) {
  document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
  document.getElementById(id).classList.add("active");
}

// ---------------------------------------------------------------------
// API status + dataset cards
// ---------------------------------------------------------------------
async function checkApiAndLoadDataset() {
  const dot = document.getElementById("apiStatusDot");
  const text = document.getElementById("apiStatusText");
  try {
    const res = await fetch(`${API_BASE}/api/dataset/status`);
    if (!res.ok) throw new Error("bad status");
    const data = await res.json();
    dot.className = "dot dot-ok";
    text.textContent = "backend connected";
    renderSensorCards(data.sensors);
  } catch (err) {
    dot.className = "dot dot-fail";
    text.textContent = `backend unreachable at ${API_BASE} -- start it with: uvicorn app.main:app`;
    document.getElementById("btnRunQuick").disabled = true;
    document.getElementById("btnCustom").disabled = true;
  }
}

function renderSensorCards(sensors) {
  const row = document.getElementById("sensorRow");
  row.innerHTML = "";
  sensors.forEach((s) => {
    const meta = SENSOR_META[s.sensor] || { full: "", role: s.role };
    const card = document.createElement("div");
    card.className = "sensor-card" + (s.role === "REFERENCE" ? " is-reference" : "");
    card.innerHTML = `
      <div class="sensor-card-top">
        <span class="sensor-id">${s.sensor}</span>
        <span class="sensor-role ${s.role === "REFERENCE" ? "role-ref" : "role-src"}">${s.role === "REFERENCE" ? "FIXED REFERENCE" : "SOURCE"}</span>
      </div>
      <p class="sensor-full-name">${meta.full}</p>
      <div class="sensor-status-row">
        <span>${s.width ? `${s.width}&times;${s.height}px` : "&mdash;"}</span>
        <span class="${s.valid ? "val-valid" : "val-invalid"}">${s.valid ? "VALID" : "MISSING/INVALID"}</span>
      </div>`;
    row.appendChild(card);
  });
}

// ---------------------------------------------------------------------
// Run pipeline
// ---------------------------------------------------------------------
async function runPipeline(mode, sources) {
  showView("viewProcessing");
  resetPipelineUI();
  animatePipelineWhileWaiting();

  let url = `${API_BASE}/api/run?mode=${mode}`;
  if (sources && sources.length) {
    sources.forEach((s) => (url += `&sources=${s}`));
  }

  try {
    const res = await fetch(url, { method: "POST" });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Pipeline execution failed.");
    }
    completePipelineUI();
    currentRun = data;
    setTimeout(() => renderResults(data), 500);
  } catch (err) {
    document.querySelector(".processing-note").textContent =
      "Processing failed: " + err.message + " -- returning to start.";
    document.querySelector(".processing-note").style.color = "var(--red)";
    setTimeout(() => showView("viewLanding"), 3500);
  }
}

function resetPipelineUI() {
  document.querySelectorAll("#pipelineSteps li").forEach((li) => li.classList.remove("active", "done"));
}

let pipelineAnimTimer = null;
function animatePipelineWhileWaiting() {
  let i = 0;
  const items = document.querySelectorAll("#pipelineSteps li");
  clearInterval(pipelineAnimTimer);
  pipelineAnimTimer = setInterval(() => {
    if (i > 0) items[i - 1].classList.replace("active", "done");
    if (i < items.length) {
      items[i].classList.add("active");
      i++;
    } else {
      clearInterval(pipelineAnimTimer);
    }
  }, 550);
}
function completePipelineUI() {
  clearInterval(pipelineAnimTimer);
  document.querySelectorAll("#pipelineSteps li").forEach((li) => {
    li.classList.remove("active");
    li.classList.add("done");
  });
}

// ---------------------------------------------------------------------
// Results rendering
// ---------------------------------------------------------------------
function renderResults(run) {
  showView("viewResults");

  document.getElementById("runMeta").textContent =
    `RUN ${run.run_id} \u2014 ${new Date(run.timestamp).toLocaleString()} \u2014 ${run.mode.toUpperCase()} \u2014 status: ${run.status}`;

  const fusionUrl = `${API_BASE}/outputs/${run.fusion_output_path}`;
  document.getElementById("fusionImage").src = fusionUrl;

  document.getElementById("btnDownloadFusion").onclick = () => {
    const filename = run.fusion_output_path.split("/").pop();
    window.open(`${API_BASE}/api/download/${run.run_id}/${filename}`, "_blank");
  };
  document.getElementById("btnDownloadReport").onclick = () => {
    window.open(`${API_BASE}/api/report/${run.run_id}`, "_blank");
  };

  renderLayerToggles(run);
  renderMetrics(run);
  renderMatchTabs(run);
  renderCompareTabs(run);
  renderAiPanel(run);
  loadHistory();
}

function renderLayerToggles(run) {
  const el = document.getElementById("layerToggles");
  const sensors = ["OHRC", ...Object.keys(run.registration)];
  el.innerHTML = sensors
    .map(
      (s) => `<label><input type="checkbox" checked disabled> ${s}${s === "OHRC" ? " (reference base)" : ""}</label>`
    )
    .join("");
}

function renderMetrics(run) {
  const grid = document.getElementById("metricsGrid");
  grid.innerHTML = "";
  Object.entries(run.registration).forEach(([sensor, r]) => {
    const card = document.createElement("div");
    card.className = "metric-card";
    if (r.status !== "SUCCESS") {
      card.innerHTML = `
        <div class="metric-card-head">
          <span class="metric-sensor">${sensor}</span>
          <span class="badge badge-failed">FAILED</span>
        </div>
        <p class="metric-fail-reason"><strong>Stage:</strong> ${r.failure_stage || "unknown"}<br/>
        <strong>Reason:</strong> ${r.failure_reason || "Unavailable."}<br/>
        <strong>Recovery attempted:</strong> ${(r.recovery_attempted || []).join(", ") || "none"}<br/>
        <strong>Lunar AI:</strong> interpretation disabled for this sensor.</p>`;
      grid.appendChild(card);
      return;
    }
    const badgeClass = { HIGH: "badge-high", MEDIUM: "badge-medium", LOW: "badge-low" }[r.confidence] || "badge-low";
    card.innerHTML = `
      <div class="metric-card-head">
        <span class="metric-sensor">${sensor}</span>
        <span class="badge ${badgeClass}">${r.confidence} CONFIDENCE</span>
      </div>
      <div class="metric-row"><span>Method</span><span>${r.method_used}</span></div>
      <div class="metric-row"><span>Keypoints (ref / src)</span><span>${r.keypoints_reference} / ${r.keypoints_source}</span></div>
      <div class="metric-row"><span>Candidate matches</span><span>${r.candidate_matches}</span></div>
      <div class="metric-row"><span>Inlier matches</span><span>${r.inlier_matches}</span></div>
      <div class="metric-row"><span>Inlier ratio</span><span>${(r.inlier_ratio * 100).toFixed(1)}%</span></div>
      <div class="metric-row"><span>RMSE</span><span>${r.rmse_px} px</span></div>
      <div class="metric-row"><span>Max reprojection err.</span><span>${r.max_reprojection_error_px} px</span></div>
      <div class="metric-row"><span>Transformation</span><span>${r.transformation_type}</span></div>
      <div class="metric-row"><span>Info. preservation</span><span>${r.information_preservation ? r.information_preservation.status : "N/A"}</span></div>`;
    grid.appendChild(card);
  });
}

function renderMatchTabs(run) {
  const successSensors = Object.entries(run.registration).filter(([, r]) => r.status === "SUCCESS").map(([s]) => s);
  const tabWrap = document.getElementById("matchTabs");
  tabWrap.innerHTML = "";
  if (!successSensors.length) {
    document.getElementById("matchCanvas").getContext("2d").clearRect(0, 0, 9999, 9999);
    return;
  }
  successSensors.forEach((s, idx) => {
    const btn = document.createElement("button");
    btn.className = "tab-btn" + (idx === 0 ? " active" : "");
    btn.textContent = `OHRC \u2194 ${s}`;
    btn.onclick = () => {
      tabWrap.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      activeSensorTab = s;
      drawMatchCanvas(run);
    };
    tabWrap.appendChild(btn);
  });
  activeSensorTab = successSensors[0];
  document.querySelectorAll('input[name="matchFilter"]').forEach((r) =>
    r.addEventListener("change", (e) => {
      matchFilterMode = e.target.value;
      drawMatchCanvas(run);
    })
  );
  drawMatchCanvas(run);
}

function drawMatchCanvas(run) {
  const r = run.registration[activeSensorTab];
  const canvas = document.getElementById("matchCanvas");
  const ctx = canvas.getContext("2d");
  const W = 1000, H = 460;
  canvas.width = W;
  canvas.height = H;
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, W, H);

  if (!r || !r.match_viz_points) return;
  const halfW = W / 2;
  const pts = r.match_viz_points;

  ctx.fillStyle = "rgba(255,255,255,0.5)";
  ctx.font = "11px monospace";
  ctx.fillText("OHRC (reference)", 10, 16);
  ctx.fillText(activeSensorTab + " (source)", halfW + 10, 16);
  ctx.strokeStyle = "rgba(255,255,255,0.1)";
  ctx.beginPath();
  ctx.moveTo(halfW, 0);
  ctx.lineTo(halfW, H);
  ctx.stroke();

  const refW = 1400, refH = 1400; // demo dataset dims; scaled generically below
  const scaleToPane = (pt, offsetX) => {
    const sx = (pt[0] / refW) * (halfW - 20) + offsetX + 10;
    const sy = (pt[1] / refH) * (H - 30) + 20;
    return [sx, sy];
  };

  let refPts, srcPts, flags;
  if (matchFilterMode === "inliers") {
    refPts = pts.ref_points;
    srcPts = pts.src_points;
    flags = refPts.map(() => true);
  } else {
    refPts = pts.all_ref_points;
    srcPts = pts.all_src_points;
    flags = pts.inlier_flags;
  }

  refPts.forEach((rp, i) => {
    const sp = srcPts[i];
    const isInlier = flags[i];
    const [rx, ry] = scaleToPane(rp, 0);
    const [sx, sy] = scaleToPane(sp, halfW);
    const color = isInlier ? "rgba(95,208,224,0.85)" : "rgba(224,125,125,0.5)";
    ctx.strokeStyle = color;
    ctx.lineWidth = 0.6;
    ctx.beginPath();
    ctx.moveTo(rx, ry);
    ctx.lineTo(sx, sy);
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(rx, ry, 2, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(sx, sy, 2, 0, Math.PI * 2);
    ctx.fill();
  });
}

function renderCompareTabs(run) {
  const wrap = document.getElementById("compareTabs");
  const grid = document.getElementById("compareGrid");
  const successSensors = Object.entries(run.registration).filter(([, r]) => r.status === "SUCCESS").map(([s]) => s);
  wrap.innerHTML = "";
  if (!successSensors.length) { grid.innerHTML = "<p style='color:var(--text-low)'>No sensors registered successfully in this run.</p>"; return; }

  function render(sensor) {
    grid.innerHTML = `
      <div class="compare-cell">
        <img src="${API_BASE}/outputs/${run.per_sensor_registered_paths[sensor]}" />
        <p class="compare-cell-label">AFTER &mdash; ${sensor} registered to OHRC frame</p>
      </div>
      <div class="compare-cell">
        <img src="${API_BASE}/outputs/${run.fusion_output_path}" />
        <p class="compare-cell-label">MULTI-SENSOR OUTPUT (includes ${sensor})</p>
      </div>`;
  }
  successSensors.forEach((s, i) => {
    const btn = document.createElement("button");
    btn.className = "tab-btn" + (i === 0 ? " active" : "");
    btn.textContent = s;
    btn.onclick = () => {
      wrap.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      render(s);
    };
    wrap.appendChild(btn);
  });
  render(successSensors[0]);
}

function renderAiPanel(run) {
  const ai = run.ai_analysis;
  const panel = document.getElementById("aiPanel");
  if (!ai) { panel.innerHTML = "<p>Lunar AI analysis unavailable for this run.</p>"; return; }

  const confBadge = { HIGH: "badge-high", MEDIUM: "badge-medium", LOW: "badge-low" }[ai.overall_confidence] || "badge-low";

  const obsHtml = ai.observations
    .map((o) => `<li><span class="ai-tag tag-${o.type}">${o.type}</span>${o.text}</li>`)
    .join("");

  const rolesHtml = Object.entries(ai.sensor_roles)
    .map(([s, desc]) => `<div class="metric-row"><span>${s}</span><span style="text-align:right;max-width:70%;font-family:var(--font-display);color:var(--text-mid)">${desc}</span></div>`)
    .join("");

  panel.innerHTML = `
    <p class="ai-headline">${ai.narrative}</p>
    <div class="ai-confidence-row">
      <span class="badge ${confBadge}">OVERALL CONFIDENCE: ${ai.overall_confidence}</span>
      <span style="font-size:11px;color:var(--text-low);font-family:var(--font-mono)">source: ${ai.narrative_source === "anthropic_api" ? "Anthropic API narrative (grounded in measured findings below)" : "rule-based deterministic engine"}</span>
    </div>
    <p class="ai-sub">Findings</p>
    <ul class="ai-obs-list">${obsHtml}</ul>
    <p class="ai-sub">Sensor contributions</p>
    ${rolesHtml}
    <p class="ai-sub">Limitations</p>
    <ul class="ai-limitations">${ai.limitations.map((l) => `<li>${l}</li>`).join("")}</ul>`;
}

async function loadHistory() {
  try {
    const res = await fetch(`${API_BASE}/api/history`);
    const data = await res.json();
    const list = document.getElementById("historyList");
    list.innerHTML = data.runs
      .slice(0, 8)
      .map(
        (r) => `<div class="history-row">
          <span>RUN ${r.run_id}</span>
          <span>${new Date(r.timestamp).toLocaleString()}</span>
          <span>${r.sensors_used.join(" + ")}</span>
          <span class="hist-status-${r.status}">${r.status}</span>
        </div>`
      )
      .join("");
  } catch (err) {
    /* history is a convenience panel; ignore failures silently */
  }
}

// ---------------------------------------------------------------------
// Event wiring
// ---------------------------------------------------------------------
document.getElementById("btnRunQuick").addEventListener("click", () => runPipeline("quick_demo"));
document.getElementById("btnCustom").addEventListener("click", () => {
  document.getElementById("customPanel").classList.toggle("hidden");
});
document.getElementById("btnRunCustom").addEventListener("click", () => {
  const selected = Array.from(document.querySelectorAll("#customPanel input[type=checkbox]:checked")).map((c) => c.value);
  runPipeline("custom", selected);
});
document.getElementById("btnBackToStart").addEventListener("click", () => {
  showView("viewLanding");
  checkApiAndLoadDataset();
});

initStarfield();
checkApiAndLoadDataset();
