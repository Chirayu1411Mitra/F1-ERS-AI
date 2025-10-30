const runBtn = document.getElementById("runBtn");
const loader = document.getElementById("loader");
const results = document.getElementById("results");
const errorPanel = document.getElementById("errorPanel");
const downloadBtn = document.getElementById("downloadBtn");
const downloadTrackBtn = document.getElementById("downloadTrackBtn");

function formatSecondsToLap(secsStr) {
  const secs = parseFloat(secsStr);
  if (!isFinite(secs)) return secsStr;
  const minutes = Math.floor(secs / 60);
  const remain = secs - minutes * 60;
  return `${String(minutes)}:${remain.toFixed(3).padStart(6, "0")}`;
}

function renderBadges(eventName, trackCondition, drsEnabled) {
  const wrap = document.getElementById("contextBadges");
  wrap.innerHTML = '';
  const b1 = `<span class="badge blue">Track: ${eventName}</span>`;
  const b2 = `<span class="badge orange">${trackCondition === 'WET' ? 'Wet' : 'Dry'}</span>`;
  const b3 = `<span class="badge green">DRS: ${drsEnabled ? 'On' : 'Off'}</span>`;
  wrap.insertAdjacentHTML('beforeend', b1 + b2 + b3);
}

function renderStrategySummary(strategy) {
  const counts = strategy.reduce((acc, s) => { acc[s] = (acc[s] || 0) + 1; return acc; }, {});
  const total = strategy.length;
  const items = [
    `Total segments: ${total}`,
    `COAST: ${counts.COAST || 0}`,
    `DEPLOY: ${counts.DEPLOY || 0}`,
    `HARVEST: ${counts.HARVEST || 0}`,
  ];
  const ul = document.getElementById("strategySummary");
  ul.innerHTML = items.map(t => `<li>${t}</li>`).join('');
}

function renderSegmentsTable(strategy) {
  const tbody = document.querySelector('#segmentsTable tbody');
  tbody.innerHTML = '';
  strategy.forEach((s, idx) => {
    const cls = s === 'DEPLOY' ? 'pill-deploy' : s === 'HARVEST' ? 'pill-harvest' : 'pill-coast';
    const row = `<tr>
      <td>${idx + 1}</td>
      <td><span class="pill ${cls}">${s}</span></td>
    </tr>`;
    tbody.insertAdjacentHTML('beforeend', row);
  });
}

async function runOptimization() {
  const trackCondition = document.getElementById("trackCondition").value;
  const eventName = document.getElementById("trackEvent").value;
  const drsEnabled = document.getElementById("drsToggle").checked;

  errorPanel.classList.add('hidden');
  loader.classList.remove("hidden");
  results.classList.add("hidden");
  runBtn.disabled = true;
  downloadBtn.disabled = true;
  downloadTrackBtn.disabled = true;

  try {
    const response = await fetch("/run-ai", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ event: eventName, track_condition: trackCondition, drs_enabled: drsEnabled }),
    });
    let data = null;
    try { data = await response.json(); } catch (_) {}
    if (!response.ok) {
      const msg = data && data.error ? data.error : `HTTP ${response.status}`;
      throw new Error(msg);
    }

    loader.classList.add("hidden");
    results.classList.remove("hidden");

    // Lap time formatted
    document.getElementById("lapTime").textContent = formatSecondsToLap(data.lap_time);
    // Badges and summary
    renderBadges(eventName, trackCondition, drsEnabled);
    renderStrategySummary(data.strategy);
    renderSegmentsTable(data.strategy);
    // Chart
    const imgUrl = data.image_path + "?t=" + new Date().getTime();
    const imgEl = document.getElementById("lapImage");
    imgEl.src = imgUrl;
    // Download
    downloadBtn.disabled = false;
    downloadBtn.onclick = () => {
      const a = document.createElement('a');
      a.href = imgUrl;
      a.download = `ers_opt_${eventName}_${trackCondition}_drs-${drsEnabled ? 'on' : 'off'}.png`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    };

    // Track map
    const trackUrl = (data.track_image_path || '') + (data.track_image_path ? ("?t=" + new Date().getTime()) : '');
    const trackEl = document.getElementById("trackImage");
    if (trackUrl) {
      trackEl.src = trackUrl;
      downloadTrackBtn.disabled = false;
      downloadTrackBtn.onclick = () => {
        const a = document.createElement('a');
        a.href = trackUrl;
        a.download = `track_map_${eventName}_${trackCondition}_drs-${drsEnabled ? 'on' : 'off'}.png`;
        document.body.appendChild(a);
        a.click();
        a.remove();
      };
    } else {
      trackEl.removeAttribute('src');
      downloadTrackBtn.disabled = true;
    }
  } catch (error) {
    console.error("Error:", error);
    loader.classList.add("hidden");
    errorPanel.textContent = `An error occurred: ${error.message}. Try a different track or toggle DRS.`;
    errorPanel.classList.remove('hidden');
  } finally {
    runBtn.disabled = false;
  }
}

runBtn.addEventListener("click", runOptimization);
