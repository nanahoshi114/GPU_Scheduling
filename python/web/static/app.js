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

let presets = { clusters: [], jobs: [] };
let nodeRows = [];

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

function applyClusterPreset(id) {
  const c = presets.clusters.find((x) => x.id === id);
  if (!c) return;
  nodeRows = c.nodes.map((n) => ({ id: n.id, gpu_count: n.gpu_count }));
  renderNodeEditor();
}

function renderMetrics(m, time) {
  if (!m) return;
  $("#m-util").textContent = fmtPct(m.gpu_utilization);
  $("#m-pending").textContent = m.pending_count;
  $("#m-wait").textContent = fmtNum(m.avg_wait_time);
  $("#m-cross").textContent = m.cross_node_jobs;
  $("#m-used").textContent = `${m.used_gpus} / ${m.total_gpus}`;
  $("#m-time").textContent = time;
}

function renderCluster(snapshot) {
  const root = $("#cluster-view");
  if (!snapshot) {
    root.textContent = "请先创建调度会话";
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

function renderJobs(snapshot) {
  const tbody = $("#job-table");
  tbody.innerHTML = "";
  (snapshot?.jobs || []).forEach((job) => {
    const tr = document.createElement("tr");
    const finishBtn =
      job.state === "running"
        ? `<button class="ghost tiny" data-finish="${job.id}">结束</button>`
        : "";
    tr.innerHTML = `
      <td>${job.id}</td>
      <td>${job.gpu_request}</td>
      <td><span class="badge ${job.state}">${job.state}</span></td>
      <td>${placementText(job)}</td>
      <td>${fmtNum(job.wait_time)}</td>
      <td class="reason-cell">${job.reason || ""}</td>
      <td>${finishBtn}</td>
    `;
    tbody.appendChild(tr);
  });
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
  renderMetrics(snapshot.metrics, snapshot.time);
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

function jobTable(jobs) {
  const rows = jobs
    .map(
      (j) => `<tr>
        <td>${j.id}</td>
        <td>${j.gpu_request}</td>
        <td><span class="badge ${j.state}">${j.state}</span></td>
        <td>${j.nodes_used}</td>
        <td>${placementText(j)}</td>
        <td class="reason-cell">${j.reason || ""}</td>
      </tr>`
    )
    .join("");
  return `<div class="table-wrap"><table>
    <thead><tr><th>任务</th><th>GPU</th><th>状态</th><th>跨节点数</th><th>分配</th><th>原因</th></tr></thead>
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
      <div class="card"><h2>First Fit 分配结果</h2>${jobTable(ff.final_snapshot.jobs)}</div>
      <div class="card"><h2>Topology-aware 分配结果</h2>${jobTable(ta.final_snapshot.jobs)}</div>
    </div>
  `;
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

$("#create-session").addEventListener("click", async () => {
  try {
    const snap = await api("/api/session", {
      method: "POST",
      body: JSON.stringify({
        nodes: nodeRows,
        strategy: $("#strategy").value,
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
  btn.disabled = true;
  $("#compare-result").innerHTML = `<p class="hint">正在模拟...</p>`;
  try {
    const data = await api("/api/compare", {
      method: "POST",
      body: JSON.stringify({
        cluster_id: $("#cmp-cluster").value,
        jobs_id: $("#cmp-jobs").value,
      }),
    });
    renderCompare(data);
  } catch (err) {
    $("#compare-result").innerHTML = `<p class="reason">${err.message}</p>`;
  } finally {
    btn.disabled = false;
  }
});

(async function init() {
  presets = await api("/api/presets");
  fillClusterSelects();
})();
