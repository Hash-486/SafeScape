(() => {
  const CLIP_MS = 2000; // record in short rolling clips
  const ALERT_COOLDOWN_MS = 5000; // avoid spamming History with repeat detections

  const els = {
    connDot: document.getElementById("connDot"),
    statusCard: document.getElementById("statusCard"),
    statusIcon: document.getElementById("statusIcon"),
    statusText: document.getElementById("statusText"),
    statusSub: document.getElementById("statusSub"),
    toggleBtn: document.getElementById("toggleBtn"),
    lastResult: document.getElementById("lastResult"),
    historyList: document.getElementById("historyList"),
    historyEmpty: document.getElementById("historyEmpty"),
    serverUrl: document.getElementById("serverUrl"),
    testConnBtn: document.getElementById("testConnBtn"),
    testConnResult: document.getElementById("testConnResult"),
  };

  let listening = false;
  let mediaRecorder = null;
  let mediaStream = null;
  let lastAlertAt = 0;

  // ---- persistence ----
  function getServerUrl() {
    return localStorage.getItem("safescape_server_url") || `${location.protocol}//${location.host}`;
  }
  function setServerUrl(url) {
    localStorage.setItem("safescape_server_url", url);
  }
  function getHistory() {
    try {
      return JSON.parse(localStorage.getItem("safescape_history") || "[]");
    } catch {
      return [];
    }
  }
  function pushHistory(entry) {
    const h = getHistory();
    h.unshift(entry);
    localStorage.setItem("safescape_history", JSON.stringify(h.slice(0, 200)));
    renderHistory();
  }

  // ---- navigation (Jakob's Law: familiar bottom-tab pattern) ----
  document.querySelectorAll(".nav-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".nav-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".screen").forEach((s) => s.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(`screen-${btn.dataset.screen}`).classList.add("active");
    });
  });

  // ---- status card rendering (Nielsen: visibility of system status; color+icon+text, not color alone) ----
  function setState(state, text, sub) {
    els.statusCard.className = `status-card state-${state}`;
    els.statusIcon.textContent = state === "hazard" ? "⚠" : state === "listening" ? "◉" : "○";
    els.statusText.textContent = text;
    els.statusSub.textContent = sub || "";
  }

  function renderHistory() {
    const h = getHistory();
    els.historyList.innerHTML = "";
    els.historyEmpty.style.display = h.length ? "none" : "block";
    for (const item of h) {
      const li = document.createElement("li");
      li.className = "history-item";

      const left = document.createElement("div");
      const clsDiv = document.createElement("div");
      clsDiv.className = "hi-class";
      clsDiv.textContent = item.cls;
      const metaDiv = document.createElement("div");
      metaDiv.className = "hi-meta";
      metaDiv.textContent = new Date(item.ts).toLocaleString();
      left.appendChild(clsDiv);
      left.appendChild(metaDiv);

      const confDiv = document.createElement("div");
      confDiv.className = "hi-meta";
      confDiv.textContent = `${(item.confidence * 100).toFixed(0)}%`;

      li.appendChild(left);
      li.appendChild(confDiv);
      els.historyList.appendChild(li);
    }
  }

  // ---- connection test (Nielsen: visibility of system status) ----
  async function testConnection(silent) {
    const url = els.serverUrl.value.trim() || getServerUrl();
    try {
      const r = await fetch(`${url}/health`, { signal: AbortSignal.timeout(5000) });
      if (!r.ok) throw new Error(`status ${r.status}`);
      const data = await r.json();
      els.connDot.className = "conn-dot conn-ok";
      if (!silent) {
        els.testConnResult.textContent = `Connected — model: ${data.model_name}`;
        els.testConnResult.className = "test-result ok";
      }
      return true;
    } catch (e) {
      els.connDot.className = "conn-dot conn-bad";
      if (!silent) {
        els.testConnResult.textContent = `Could not connect: ${e.message}`;
        els.testConnResult.className = "test-result bad";
      }
      return false;
    }
  }

  els.testConnBtn.addEventListener("click", () => testConnection(false));
  els.serverUrl.addEventListener("change", () => setServerUrl(els.serverUrl.value.trim()));

  // ---- recording loop ----
  async function sendClip(blob) {
    const url = els.serverUrl.value.trim() || getServerUrl();
    const form = new FormData();
    form.append("file", blob, "clip.webm");
    try {
      const r = await fetch(`${url}/predict`, { method: "POST", body: form, signal: AbortSignal.timeout(10000) });
      if (!r.ok) throw new Error(`status ${r.status}`);
      const out = await r.json();
      els.connDot.className = "conn-dot conn-ok";
      handlePrediction(out);
    } catch (e) {
      els.connDot.className = "conn-dot conn-bad";
      els.lastResult.textContent = `(offline: ${e.message})`;
    }
  }

  function handlePrediction(out) {
    els.lastResult.textContent =
      `Last: ${out.predicted_class} (${(out.confidence * 100).toFixed(0)}%)`;
    if (out.is_hazard) {
      setState("hazard", `Hazard: ${out.predicted_class}`, `Confidence ${(out.confidence * 100).toFixed(0)}%`);
      const now = Date.now();
      if (now - lastAlertAt > ALERT_COOLDOWN_MS) {
        lastAlertAt = now;
        pushHistory({ cls: out.predicted_class, confidence: out.confidence, ts: now });
      }
    } else if (listening) {
      setState("listening", "Listening", "All clear");
    }
  }

  function recordOneClip() {
    if (!listening || !mediaStream) return;
    const chunks = [];
    let rec;
    try {
      rec = new MediaRecorder(mediaStream);
    } catch (e) {
      setState("idle", "Recorder error", e.message);
      return;
    }
    mediaRecorder = rec;
    rec.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };
    rec.onstop = () => {
      if (chunks.length) sendClip(new Blob(chunks, { type: chunks[0].type }));
      if (listening) setTimeout(recordOneClip, 0);
    };
    rec.start();
    setTimeout(() => { if (rec.state !== "inactive") rec.stop(); }, CLIP_MS);
  }

  async function startListening() {
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (e) {
      setState("idle", "Microphone permission denied", e.message);
      return;
    }
    listening = true;
    els.toggleBtn.textContent = "Stop Listening";
    els.toggleBtn.classList.add("listening");
    setState("listening", "Listening", "Monitoring for hazard sounds");
    recordOneClip();
  }

  function stopListening() {
    listening = false;
    els.toggleBtn.textContent = "Start Listening";
    els.toggleBtn.classList.remove("listening");
    if (mediaRecorder && mediaRecorder.state !== "inactive") mediaRecorder.stop();
    if (mediaStream) mediaStream.getTracks().forEach((t) => t.stop());
    setState("idle", "Not listening", "Tap start to begin monitoring");
  }

  els.toggleBtn.addEventListener("click", () => (listening ? stopListening() : startListening()));

  // ---- init ----
  els.serverUrl.value = getServerUrl();
  renderHistory();
  testConnection(true);
  setInterval(() => testConnection(true), 15000);
})();
