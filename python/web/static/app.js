const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];

const PALETTE = [
  "#3ee0c4", "#6ea8ff", "#ffc857", "#c084fc", "#fb7185",
  "#34d399", "#60a5fa", "#f472b6", "#fbbf24", "#2dd4bf",
  "#a3e635", "#38bdf8", "#e879f9", "#fdba74", "#4ade80",
  "#818cf8", "#f87171", "#22d3ee", "#d8b4fe", "#86efac",
];

function colorFor(jobId) {
  if (!jobId) return null;
  let h = 0;
  for (let i = 0; i < jobId.length; i++) h = (h * 33 + jobId.charCodeAt(i)) >>> 0;
  return PALETTE[h % PALETTE.length];
}

function fmtPct(x) {
  return `${(x * 100).toFixed(1)}%`;
}

function fmtNum(x) {
  return Number.isInteger(x) ? String(x) : x.toFixed(2);
}

function placementText(job) {
  if (!job.placement || job.placement.length === 0) return "—";
  return job.placement
    .map((p) => `${p.node_id}: GPU ${p.gpu_indices.join(",")}`)
    .join("； ");
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg = data.detail || res.statusText;
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return data;
}

const DEMO_QUEUES = [
  { id: "research", gpu_quota: 16 },
  { id: "prod", gpu_quota: 12 },
  { id: "default", gpu_quota: 4 },
];

let presets = { clusters: [], jobs: [] };
let nodeRows = [];
let queueRows = [];
let sessionQueues = [];

function renderNodeEditor() {
  const root = $("#node-editor");
  root.innerHTML = "";
  nodeRows.forEach((row, idx) => {
    const div = document.createElement("div");
    div.className = "node-row";
    div.innerHTML = `
      <input data-k="id" value="${row.id}" />
      <input data-k="gpu_count" type="number" min="1" value="${row.gpu_count}" />
      <button class="ghost" data-del="${idx}" title="删除">×</button>
    `;
    div.querySelector('[data-k="id"]').addEventListener("input", (e) => {
      nodeRows[idx].id = e.target.value;
    });
    div.querySelector('[data-k="gpu_count"]').addEventListener("input", (e) => {
      nodeRows[idx].gpu_count = Number(e.target.value);
    });
    div.querySelector("[data-del]").addEventListener("click", () => {
      nodeRows.splice(idx, 1);
      renderNodeEditor();
    });
    root.appendChild(div);
  });
}

function fillClusterSelects() {
  const opts = presets.clusters
    .map((c) => `<option value="${c.id}">${c.name}</option>`)
    .join("");
  $("#cluster-preset").innerHTML = opts;
  $("#cmp-cluster").innerHTML = opts;
  $("#cmp-jobs").innerHTML = presets.jobs
    .map((j) => `<option value="${j.id}">${j.name}</option>`)
    .join("");
  applyClusterPreset($("#cluster-preset").value);
}

function clusterGpuTotal() {
  return nodeRows.reduce((sum, n) => sum + Number(n.gpu_count || 0), 0);
}

function defaultQueueRows() {
  return [{ id: "default", gpu_quota: Math.max(1, clusterGpuTotal()) }];
}

function renderQueueEditor() {
  const root = $("#queue-editor");
  if (!root) return;
  root.innerHTML = "";
  queueRows.forEach((row, idx) => {
    const div = document.createElement("div");
    div.className = "node-row";
    div.innerHTML = `
      <input data-k="id" value="${row.id}" />
      <input data-k="gpu_quota" type="number" min="1" value="${row.gpu_quota}" />
      <button class="ghost" data-del="${idx}" title="删除">×</button>
    `;
    div.querySelector('[data-k="id"]').addEventListener("input", (e) => {
      queueRows[idx].id = e.target.value;
    });
    div.querySelector('[data-k="gpu_quota"]').addEventListener("input", (e) => {
      queueRows[idx].gpu_quota = Number(e.target.value);
    });
    div.querySelector("[data-del]").addEventListener("click", () => {
      queueRows.splice(idx, 1);
      if (queueRows.length === 0) queueRows = defaultQueueRows();
      renderQueueEditor();
    });
    root.appendChild(div);
  });
}

function fillJobQueueSelect(queues) {
  const sel = $("#job-queue");
  if (!sel) return;
  const current = sel.value;
  const list = queues && queues.length ? queues : [{ id: "default" }];
  sel.innerHTML = list.map((q) => `<option value="${q.id}">${q.id}</option>`).join("");
  if (list.some((q) => q.id === current)) sel.value = current;
}

function renderQueueBars(queues, root = $("#queue-bars")) {
  if (!root) return;
  if (!queues || queues.length === 0) {
    root.innerHTML = "";
    return;
  }
  root.innerHTML = queues
    .map((q) => {
      const pct = q.gpu_quota > 0 ? Math.min(100, (q.used_gpus / q.gpu_quota) * 100) : 0;
      const full = q.used_gpus >= q.gpu_quota ? " full" : "";
      return `<div class="queue-bar">
        <div class="meta"><span>${q.id}</span><span>${q.used_gpus} / ${q.gpu_quota} · 等待 ${q.pending_count}</span></div>
        <div class="track"><div class="fill${full}" style="width:${pct}%"></div></div>
      </div>`;
    })
    .join("");
}

function applyClusterPreset(id) {
  const c = presets.clusters.find((x) => x.id === id);
  if (!c) return;
  nodeRows = c.nodes.map((n) => ({ id: n.id, gpu_count: n.gpu_count }));
  queueRows = defaultQueueRows();
  renderNodeEditor();
  renderQueueEditor();
}

function renderMetrics(m, time) {
  if (!m) return;
  $("#m-util").textContent = fmtPct(m.gpu_utilization);
  $("#m-pending").textContent = m.pending_count;
  $("#m-wait").textContent = fmtNum(m.avg_wait_time);
  $("#m-cross").textContent = m.cross_node_jobs;
  $("#m-preemptions").textContent = m.total_preemptions;
  $("#m-used").textContent = `${m.used_gpus} / ${m.total_gpus}`;
  if ($("#m-fair")) $("#m-fair").textContent = m.fair_share_pending ?? 0;
  if ($("#m-frag")) $("#m-frag").textContent = m.fragmentation_pending ?? 0;
  $("#m-time").textContent = time;
}

function renderCluster(snapshot, root = $("#cluster-view"), emptyText = "请先创建调度会话") {
  if (!root) return;
  if (!snapshot) {
    root.textContent = emptyText;
    return;
  }
  root.innerHTML = "";
  snapshot.nodes.forEach((node) => {
    const card = document.createElement("div");
    card.className = "node-card";
    card.innerHTML = `<header><span>${node.id}</span><span>空闲 ${node.free_count}/${node.gpu_count}</span></header>`;
    const gpus = document.createElement("div");
    gpus.className = "gpus";
    node.gpus.forEach((g) => {
      const cell = document.createElement("div");
      cell.className = "gpu" + (g.job_id ? " busy" : "");
      cell.textContent = g.job_id ? g.job_id.replace(/^job-/, "") : `G${g.index}`;
      cell.title = g.job_id ? `${node.id} GPU ${g.index} ← ${g.job_id}` : `${node.id} GPU ${g.index} 空闲`;
      if (g.job_id) cell.style.background = colorFor(g.job_id);
      gpus.appendChild(cell);
    });
    card.appendChild(gpus);
    root.appendChild(card);
  });
}

function renderJobs(snapshot, tbody = $("#job-table"), { interactive = true } = {}) {
  if (!tbody) return;
  tbody.innerHTML = "";
  (snapshot?.jobs || []).forEach((job) => {
    const tr = document.createElement("tr");
    const finishBtn =
      interactive && job.state === "running"
        ? `<button class="ghost tiny" data-finish="${job.id}">结束</button>`
        : "";
    tr.innerHTML = `
      <td>${job.id}</td>
      <td>${job.queue_id || "default"}</td>
      <td>P${job.priority}</td>
      <td>${job.gpu_request}</td>
      <td><span class="badge ${job.state}">${job.state}</span></td>
      <td>${placementText(job)}</td>
      <td>${fmtNum(job.wait_time)}</td>
      <td>${job.remaining_duration}</td>
      <td>${job.preemption_count}</td>
      <td class="reason-cell">${job.reason || ""}</td>
      <td>${finishBtn}</td>
    `;
    tbody.appendChild(tr);
  });
  if (!interactive) return;
  tbody.querySelectorAll("[data-finish]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      try {
        const snap = await api(`/api/jobs/${encodeURIComponent(btn.dataset.finish)}/finish`, {
          method: "POST",
          body: "{}",
        });
        applySnapshot(snap);
      } catch (err) {
        $("#last-reason").textContent = err.message;
      }
    });
  });
}

function applySnapshot(snapshot) {
  sessionQueues = snapshot.queues || [];
  fillJobQueueSelect(sessionQueues);
  renderMetrics(snapshot.metrics, snapshot.time);
  renderQueueBars(sessionQueues);
  renderCluster(snapshot);
  renderJobs(snapshot);
}

function metricsBlock(title, m, other) {
  const rows = [
    ["时间平均 GPU 利用率", fmtPct(m.avg_utilization), "avg_utilization", true],
    ["结束时 GPU 利用率", fmtPct(m.gpu_utilization), "gpu_utilization", true],
    ["等待任务数", m.pending_count, "pending_count", false],
    ["峰值等待任务", m.max_pending, "max_pending", false],
    ["平均等待时间", fmtNum(m.avg_wait_time), "avg_wait_time", false],
    ["跨 Node GPU 任务数", m.cross_node_jobs, "cross_node_jobs", false],
    ["累计抢占次数", m.total_preemptions, "total_preemptions", false],
    ["公平份额等待", m.fair_share_pending ?? 0, "fair_share_pending", false],
    ["碎片等待", m.fragmentation_pending ?? 0, "fragmentation_pending", false],
    ["Makespan", m.makespan, "makespan", false],
    ["已完成 / 运行中", `${m.finished_count} / ${m.running_count}`, null, null],
  ];
  const html = rows
    .map(([k, v, key, higherBetter]) => {
      let cls = "";
      if (other && key) {
        const a = m[key];
        const b = other[key];
        if (a !== b) {
          const win = higherBetter ? a > b : a < b;
          if (win) cls = "winner";
        }
      }
      return `<div class="cmp-metric"><span>${k}</span><strong class="${cls}">${v}</strong></div>`;
    })
    .join("");
  return `<div class="card"><h2>${title}</h2>${html}</div>`;
}

function pinCompareJobLists() {
  $$("#compare-result .job-list-clip").forEach((wrap) => {
    requestAnimationFrame(() => {
      wrap.scrollTop = wrap.scrollHeight;
    });
  });
}

function jobTable(jobs) {
  const rows = jobs
    .map(
      (j) => `<tr>
        <td>${j.id}</td>
        <td>${j.queue_id || "default"}</td>
        <td>P${j.priority}</td>
        <td>${j.gpu_request}</td>
        <td><span class="badge ${j.state}">${j.state}</span></td>
        <td>${j.nodes_used}</td>
        <td>${placementText(j)}</td>
        <td>${j.remaining_duration}</td>
        <td>${j.preemption_count}</td>
        <td class="reason-cell">${j.reason || ""}</td>
      </tr>`
    )
    .join("");
  return `<div class="table-wrap job-list-clip"><table>
    <thead><tr><th>任务</th><th>队列</th><th>优先级</th><th>GPU</th><th>状态</th><th>跨节点数</th><th>分配</th><th>剩余</th><th>抢占次数</th><th>原因</th></tr></thead>
    <tbody>${rows}</tbody>
  </table></div>`;
}

function renderCompare(data) {
  const ff = data.first_fit;
  const ta = data.topology_aware;
  $("#compare-result").innerHTML = `
    <div class="compare-grid">
      ${metricsBlock("First Fit", ff.metrics, ta.metrics)}
      ${metricsBlock("Topology-aware", ta.metrics, ff.metrics)}
    </div>
    <div class="compare-grid">
      <div class="card"><h2>First Fit 分配结果</h2>
        <div class="queue-bars">${(ff.final_snapshot.queues || []).map((q) =>
          `<div class="queue-bar"><div class="meta"><span>${q.id}</span><span>${q.used_gpus} / ${q.gpu_quota} · 等待 ${q.pending_count}</span></div></div>`
        ).join("")}</div>
        ${jobTable(ff.final_snapshot.jobs)}</div>
      <div class="card"><h2>Topology-aware 分配结果</h2>
        <div class="queue-bars">${(ta.final_snapshot.queues || []).map((q) =>
          `<div class="queue-bar"><div class="meta"><span>${q.id}</span><span>${q.used_gpus} / ${q.gpu_quota} · 等待 ${q.pending_count}</span></div></div>`
        ).join("")}</div>
        ${jobTable(ta.final_snapshot.jobs)}</div>
    </div>
  `;
  pinCompareJobLists();
}

function snapshotAt(timeline, t, fallback) {
  if (!timeline || timeline.length === 0) return fallback;
  let last = timeline[0];
  for (const snap of timeline) {
    if (snap.time > t) break;
    last = snap;
  }
  return last;
}

function liveLine(m) {
  return `利用率 ${fmtPct(m.gpu_utilization)} · 等待 ${m.pending_count} · 公平份额 ${m.fair_share_pending ?? 0} · 碎片 ${m.fragmentation_pending ?? 0} · 跨节点 ${m.cross_node_jobs} · 抢占 ${m.total_preemptions} · 已用 ${m.used_gpus}/${m.total_gpus}`;
}

function ensureLiveLayout(manual) {
  $("#compare-result").innerHTML = `
    <div class="live-controls">
      <p id="cmp-clock" class="live-clock">准备播放...</p>
      <button id="cmp-step-back" class="ghost" ${manual ? "" : "hidden"}>时间 -1</button>
      <button id="cmp-step-fwd" class="ghost" ${manual ? "" : "hidden"}>时间 +1</button>
    </div>
    <div class="compare-grid">
      <div class="card">
        <h2>First Fit</h2>
        <p id="cmp-ff-live-m" class="live-m"></p>
        <div id="cmp-ff-queues" class="queue-bars"></div>
        <div id="cmp-ff-cluster" class="cluster-view"></div>
        <div class="table-wrap job-list-clip" style="margin-top:12px">
          <table>
            <thead>
              <tr><th>任务</th><th>队列</th><th>优先级</th><th>GPU</th><th>状态</th><th>节点 / GPU</th><th>等待</th><th>剩余</th><th>抢占次数</th><th>原因</th><th></th></tr>
            </thead>
            <tbody id="cmp-ff-jobs"></tbody>
          </table>
        </div>
      </div>
      <div class="card">
        <h2>Topology-aware</h2>
        <p id="cmp-ta-live-m" class="live-m"></p>
        <div id="cmp-ta-queues" class="queue-bars"></div>
        <div id="cmp-ta-cluster" class="cluster-view"></div>
        <div class="table-wrap job-list-clip" style="margin-top:12px">
          <table>
            <thead>
              <tr><th>任务</th><th>队列</th><th>优先级</th><th>GPU</th><th>状态</th><th>节点 / GPU</th><th>等待</th><th>剩余</th><th>抢占次数</th><th>原因</th><th></th></tr>
            </thead>
            <tbody id="cmp-ta-jobs"></tbody>
          </table>
        </div>
      </div>
    </div>
    <div id="cmp-summary"></div>
  `;
}

function renderLiveFrame(data, t, makespan, { done = false, manual = false } = {}) {
  const ff = snapshotAt(data.first_fit.timeline, t, data.first_fit.final_snapshot);
  const ta = snapshotAt(data.topology_aware.timeline, t, data.topology_aware.final_snapshot);
  let status;
  if (done) status = "播放结束";
  else if (manual) status = "手动";
  else status = "自动播放";
  const unit = manual ? "点「时间 +1」推进" : "每秒 1 tick";
  $("#cmp-clock").textContent = `${status} · 模拟时间 t = ${t} / ${makespan}（${unit}）`;
  $("#cmp-ff-live-m").textContent = liveLine(ff.metrics);
  $("#cmp-ta-live-m").textContent = liveLine(ta.metrics);
  renderQueueBars(ff.queues, $("#cmp-ff-queues"));
  renderQueueBars(ta.queues, $("#cmp-ta-queues"));
  renderCluster(ff, $("#cmp-ff-cluster"));
  renderCluster(ta, $("#cmp-ta-cluster"));
  renderJobs(ff, $("#cmp-ff-jobs"), { interactive: false });
  renderJobs(ta, $("#cmp-ta-jobs"), { interactive: false });
  pinCompareJobLists();
  const back = $("#cmp-step-back");
  const fwd = $("#cmp-step-fwd");
  if (back) back.disabled = t <= 0;
  if (fwd) fwd.disabled = t >= makespan;
}

function showCompareSummary(data) {
  $("#cmp-summary").innerHTML = `
    <div class="compare-grid" style="margin-top:16px">
      ${metricsBlock("First Fit 最终指标", data.first_fit.metrics, data.topology_aware.metrics)}
      ${metricsBlock("Topology-aware 最终指标", data.topology_aware.metrics, data.first_fit.metrics)}
    </div>
  `;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

let comparePlay = { gen: 0, data: null, t: 0, makespan: 0, manual: false };

function stopComparePlay() {
  comparePlay.gen += 1;
  comparePlay.manual = false;
  $("#stop-compare").disabled = true;
  $("#run-compare").disabled = false;
}

function stepCompare(delta) {
  if (!comparePlay.manual || !comparePlay.data) return;
  const max = comparePlay.makespan;
  comparePlay.t = Math.max(0, Math.min(max, comparePlay.t + delta));
  const done = comparePlay.t >= max;
  renderLiveFrame(comparePlay.data, comparePlay.t, max, { done, manual: true });
  if (done) showCompareSummary(comparePlay.data);
  else $("#cmp-summary").innerHTML = "";
}

async function playCompare(data, { manual = false } = {}) {
  const gen = ++comparePlay.gen;
  const makespan = Math.max(
    data.first_fit.metrics.makespan || 0,
    data.topology_aware.metrics.makespan || 0
  );
  comparePlay.data = data;
  comparePlay.t = 0;
  comparePlay.makespan = makespan;
  comparePlay.manual = manual;
  ensureLiveLayout(manual);
  $("#stop-compare").disabled = false;
  $("#run-compare").disabled = true;

  if (manual) {
    $("#cmp-step-back").addEventListener("click", () => stepCompare(-1));
    $("#cmp-step-fwd").addEventListener("click", () => stepCompare(1));
    renderLiveFrame(data, 0, makespan, { manual: true });
    $("#run-compare").disabled = false;
    return;
  }

  for (let t = 0; t <= makespan; t++) {
    if (gen !== comparePlay.gen) return;
    comparePlay.t = t;
    renderLiveFrame(data, t, makespan, { manual: false });
    if (t < makespan) await sleep(1000);
  }
  if (gen !== comparePlay.gen) return;
  renderLiveFrame(data, makespan, makespan, { done: true, manual: false });
  showCompareSummary(data);
  if (gen === comparePlay.gen) {
    $("#stop-compare").disabled = true;
    $("#run-compare").disabled = false;
  }
}

$$(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    $$(".tab").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    $("#panel-interactive").classList.toggle("hidden", btn.dataset.tab !== "interactive");
    $("#panel-compare").classList.toggle("hidden", btn.dataset.tab !== "compare");
  });
});

$("#cluster-preset").addEventListener("change", (e) => applyClusterPreset(e.target.value));
$("#add-node").addEventListener("click", () => {
  const n = nodeRows.length + 1;
  nodeRows.push({ id: `node-${n}`, gpu_count: 8 });
  renderNodeEditor();
});

$("#add-queue").addEventListener("click", () => {
  queueRows.push({ id: `queue-${queueRows.length + 1}`, gpu_quota: 8 });
  renderQueueEditor();
});

$("#fill-demo-queues").addEventListener("click", () => {
  queueRows = DEMO_QUEUES.map((q) => ({ ...q }));
  renderQueueEditor();
});

$("#create-session").addEventListener("click", async () => {
  try {
    const snap = await api("/api/session", {
      method: "POST",
      body: JSON.stringify({
        nodes: nodeRows,
        strategy: $("#strategy").value,
        enable_preemption: $("#enable-preemption").checked,
        queues: queueRows,
      }),
    });
    $("#last-reason").textContent = `会话已创建，策略 ${snap.strategy}`;
    applySnapshot(snap);
  } catch (err) {
    $("#last-reason").textContent = err.message;
  }
});

$("#submit-job").addEventListener("click", async () => {
  try {
    const payload = {
      id: $("#job-id").value.trim() || `job-${Date.now()}`,
      gpu_request: Number($("#job-gpus").value),
      duration: Number($("#job-duration").value),
      priority: Number($("#job-priority").value),
      queue_id: $("#job-queue").value || "default",
    };
    const data = await api("/api/jobs", { method: "POST", body: JSON.stringify(payload) });
    $("#last-reason").textContent = data.result.reason;
    $("#job-id").value = "";
    applySnapshot(data.snapshot);
  } catch (err) {
    $("#last-reason").textContent = err.message;
  }
});

$("#tick").addEventListener("click", async () => {
  try {
    const snap = await api("/api/tick", { method: "POST", body: "{}" });
    applySnapshot(snap);
  } catch (err) {
    $("#last-reason").textContent = err.message;
  }
});

$("#run-compare").addEventListener("click", async () => {
  const btn = $("#run-compare");
  const visualize = Boolean($("#cmp-visualize") && $("#cmp-visualize").checked);
  const manual = Boolean(visualize && $("#cmp-manual") && $("#cmp-manual").checked);
  stopComparePlay();
  btn.disabled = true;
  $("#compare-result").innerHTML = visualize
    ? `<p class="hint">${manual ? "正在计算模拟结果，随后可手动逐步推进时间..." : "正在计算模拟结果，随后按 1 秒 / tick 并排播放 GPU 占用..."}</p>`
    : `<p class="hint">正在模拟...</p>`;
  try {
    const data = await api("/api/compare", {
      method: "POST",
      body: JSON.stringify({
        cluster_id: $("#cmp-cluster").value,
        jobs_id: $("#cmp-jobs").value,
        enable_preemption: $("#cmp-preemption").checked,
      }),
    });
    if (visualize) {
      await playCompare(data, { manual });
    } else {
      renderCompare(data);
      btn.disabled = false;
    }
  } catch (err) {
    $("#compare-result").innerHTML = `<p class="reason">${err.message}</p>`;
    btn.disabled = false;
  }
});

$("#cmp-visualize").addEventListener("change", () => {
  const on = $("#cmp-visualize").checked;
  $("#cmp-manual").disabled = !on;
  if (!on) $("#cmp-manual").checked = false;
});

$("#stop-compare").addEventListener("click", () => {
  stopComparePlay();
  const clock = $("#cmp-clock");
  if (clock && !clock.textContent.includes("已停止")) {
    clock.textContent = `已停止 · ${clock.textContent}`;
  }
});

(async function init() {
  presets = await api("/api/presets");
  fillClusterSelects();
})();
