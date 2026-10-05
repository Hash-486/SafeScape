(() => {
  const CLIP_MS = 2000; // same clip length the mobile app sends
  const $ = (id) => document.getElementById(id);
  const pct = (x) => `${(x * 100).toFixed(1)}%`;
  const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  let MODELS = []; // from /models: [{key, name, owner, metrics, bundle_kb}]
  let CLIPS = [];  // from /testclips: [{file, label, url}]

  async function getJSON(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${url}: status ${r.status}`);
    return r.json();
  }

  async function predictAll(blob, name) {
    const form = new FormData();
    form.append("file", blob, name || "clip.wav");
    const r = await fetch("/predict/all", { method: "POST", body: form });
    if (!r.ok) throw new Error(`status ${r.status}: ${(await r.text()).slice(0, 200)}`);
    return (await r.json()).results;
  }

  // ---- live cards ----
  function probRows(probs) {
    const top = Object.entries(probs).reduce((a, b) => (b[1] > a[1] ? b : a))[0];
    return Object.entries(probs).map(([cls, p]) => {
      const kind = cls === top ? "top" : cls === "ambience" ? "sf" : "hz";
      return `<div class="prob"><span class="lbl">${cls}</span>
        <div class="bar"><span class="${kind}" style="width:${(p * 100).toFixed(1)}%"></span></div>
        <span class="val">${(p * 100).toFixed(0)}%</span></div>`;
    }).join("");
  }

  function renderCards(results, truth) {
    $("cards").innerHTML = MODELS.map((m) => {
      const r = results && results[m.key];
      if (!r) {
        return `<div class="model-card"><div class="mc-name">${m.name}</div>
          <div class="mc-owner">${m.owner}</div><div class="mc-empty">Waiting for a clip…</div></div>`;
      }
      const badge = truth
        ? `<span class="badge ${r.predicted_class === truth ? "ok" : "bad"}">${r.predicted_class === truth ? "✓ correct" : "✗ wrong"}</span>`
        : "";
      return `<div class="model-card ${r.is_hazard ? "hazard" : "safe"}">
        <div class="mc-name">${m.name}</div><div class="mc-owner">${m.owner}</div>
        <div class="mc-label">${r.is_hazard ? "⚠ " : ""}${r.predicted_class}${badge}</div>
        <div class="mc-conf">confidence ${pct(r.confidence)}</div>
        ${probRows(r.probabilities)}
        <div class="mc-lat">server inference ${r.latency_ms.toFixed(1)} ms</div></div>`;
    }).join("");
  }

  async function runOne(blob, name, truth, info) {
    $("inputInfo").textContent = `${info} — classifying with ${MODELS.length} models…`;
    try {
      const results = await predictAll(blob, name);
      renderCards(results, truth);
      $("inputInfo").textContent = truth ? `${info} — ground truth: ${truth}` : info;
      return results;
    } catch (e) {
      $("inputInfo").textContent = `${info} — request failed: ${e.message}`;
    }
  }

  async function playClip(i) {
    const c = CLIPS[i];
    const blob = await (await fetch(c.url)).blob();
    $("player").src = c.url;
    $("clipPick").value = String(i);
    return runOne(blob, c.file, c.label, `Test clip ${c.file}`);
  }

  // ---- inputs ----
  $("clipPick").addEventListener("change", (e) => {
    if (e.target.value !== "") playClip(Number(e.target.value));
  });

  $("upload").addEventListener("change", (e) => {
    const f = e.target.files[0];
    if (!f) return;
    $("player").src = URL.createObjectURL(f);
    $("clipPick").value = "";
    runOne(f, f.name, null, `Uploaded ${f.name}`);
    e.target.value = "";
  });

  function micUnavailable(reason) {
    $("recBtn").disabled = true;
    $("micNote").textContent = `Microphone unavailable (${reason}) — use upload or a test clip.`;
  }

  async function record() {
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (e) {
      micUnavailable(e.message);
      return;
    }
    const chunks = [];
    const rec = new MediaRecorder(stream);
    rec.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };
    rec.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      $("recBtn").classList.remove("recording");
      $("recBtn").textContent = "● Record 2 s";
      const blob = new Blob(chunks, { type: chunks[0] ? chunks[0].type : "audio/webm" });
      $("player").src = URL.createObjectURL(blob);
      $("clipPick").value = "";
      runOne(blob, "mic.webm", null, "Microphone recording");
    };
    $("recBtn").classList.add("recording");
    $("recBtn").textContent = "Recording…";
    rec.start();
    setTimeout(() => rec.state !== "inactive" && rec.stop(), CLIP_MS);
  }

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    micUnavailable("browsers only allow it on localhost or HTTPS");
  } else {
    $("recBtn").addEventListener("click", record);
  }

  // ---- UI test run ----
  async function runAll() {
    $("runAll").disabled = true;
    const correct = Object.fromEntries(MODELS.map((m) => [m.key, 0]));
    $("grid").querySelector("thead").innerHTML =
      `<tr><th>clip</th><th>truth</th>${MODELS.map((m) => `<th>${m.name}</th>`).join("")}</tr>`;
    const body = $("grid").querySelector("tbody");
    const foot = $("grid").querySelector("tfoot");
    body.innerHTML = "";
    for (let i = 0; i < CLIPS.length; i++) {
      const c = CLIPS[i];
      $("progress").textContent = `${i + 1} / ${CLIPS.length}`;
      const blob = await (await fetch(c.url)).blob();
      let results;
      try {
        results = await predictAll(blob, c.file);
      } catch (e) {
        body.insertAdjacentHTML("beforeend",
          `<tr><td>${c.file}</td><td>${c.label}</td><td colspan="${MODELS.length}" class="bad">${esc(e.message)}</td></tr>`);
        continue;
      }
      renderCards(results, c.label);
      const cells = MODELS.map((m) => {
        const got = results[m.key].predicted_class;
        const ok = got === c.label;
        if (ok) correct[m.key]++;
        return `<td class="${ok ? "ok" : "bad"}">${ok ? "✓" : "✗"} ${got}</td>`;
      }).join("");
      body.insertAdjacentHTML("beforeend", `<tr><td>${c.file}</td><td>${c.label}</td>${cells}</tr>`);
      foot.innerHTML = `<tr><td colspan="2">accuracy (${i + 1} clips)</td>${MODELS.map((m) =>
        `<td>${correct[m.key]}/${i + 1} · ${pct(correct[m.key] / (i + 1))}</td>`).join("")}</tr>`;
    }
    $("progress").textContent = `done — ${CLIPS.length} clips`;
    $("runAll").disabled = false;
  }
  $("runAll").addEventListener("click", runAll);

  // ---- offline metrics ----
  function renderOffline() {
    const evald = MODELS.filter((m) => m.metrics);
    const n = evald[0] ? evald[0].metrics.test_windows : 0;
    $("offlineHint").textContent = n ? `held-out test split, ${n} windows, no byte-identical copy in training` : "";
    const best = (f, lower) => {
      const vals = evald.map((m) => f(m));
      return lower ? Math.min(...vals) : Math.max(...vals);
    };
    const cols = [
      ["accuracy", (m) => m.metrics.accuracy, pct],
      ["macro-F1", (m) => m.metrics.macro_f1, (x) => x.toFixed(3)],
      ["distress recall", (m) => m.metrics.distress_recall, pct],
      ["params", (m) => m.metrics.params, (x) => x.toLocaleString("en-US"), true],
      ["served size", (m) => m.bundle_kb, (x) => `${x.toFixed(0)} KB`, true],
      ["CPU latency / window", (m) => m.metrics.latency_ms, (x) => `${x.toFixed(2)} ms`, true],
    ];
    const head = `<tr><th>model</th><th>owner</th>${cols.map((c) => `<th class="num">${c[0]}</th>`).join("")}</tr>`;
    const rows = MODELS.map((m) => {
      if (!m.metrics) {
        return `<tr><td>${m.name}</td><td>${m.owner}</td><td colspan="${cols.length}" class="num">not evaluated</td></tr>`;
      }
      return `<tr><td>${m.name}</td><td>${m.owner}</td>${cols.map(([, f, fmt, lower]) => {
        const v = f(m);
        return `<td class="num ${v === best(f, lower) && evald.length > 1 ? "best" : ""}">${fmt(v)}</td>`;
      }).join("")}</tr>`;
    }).join("");
    $("metrics").innerHTML = `<thead>${head}</thead><tbody>${rows}</tbody>`;

    $("bars").innerHTML = cols.slice(0, 3).map(([title, f]) => `<div class="metric-group"><h3>${title}</h3>${
      evald.map((m) => `<div class="prob"><span class="lbl">${m.name}</span>
        <div class="bar"><span class="top" style="width:${(f(m) * 100).toFixed(1)}%"></span></div>
        <span class="val">${(f(m) * 100).toFixed(1)}</span></div>`).join("")}</div>`).join("");

    $("cms").innerHTML = evald.map((m) => `<figure>
      <img src="/figures/${m.metrics.confusion_png.split("/").pop()}" alt="${m.name} confusion matrix" />
      <figcaption>${m.name}</figcaption></figure>`).join("");
  }

  // ---- init ----
  async function init() {
    MODELS = (await getJSON("/models")).models;
    CLIPS = (await getJSON("/testclips")).clips;
    CLIPS.forEach((c, i) => $("clipPick").insertAdjacentHTML("beforeend",
      `<option value="${i}">${c.file} (${c.label})</option>`));
    renderCards(null, null);
    renderOffline();

    // ?demo=clip&n=3 / ?demo=runall / ?demo=offline: hands-free states for screenshots.
    // Headless Edge can neither click nor screenshot a scrolled page cleanly, so the
    // runall/offline states show only their own panel.
    const q = new URLSearchParams(location.search);
    const only = (id) => document.querySelectorAll(".panel").forEach((p) => { if (p.id !== id) p.style.display = "none"; });
    if (q.get("demo") === "clip") await playClip(Number(q.get("n") || 0));
    if (q.get("demo") === "runall") { only("uitest"); await runAll(); }
    if (q.get("demo") === "offline") only("offline");
  }
  init().catch((e) => { $("inputInfo").textContent = `Could not load: ${e.message}`; });
})();
