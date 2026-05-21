const video = document.querySelector("#video");
const canvas = document.querySelector("#canvas");
const emptyState = document.querySelector("#emptyState");
const startButton = document.querySelector("#startButton");
const stopButton = document.querySelector("#stopButton");
const statusDot = document.querySelector("#statusDot");
const statusText = document.querySelector("#statusText");
const latencyText = document.querySelector("#latencyText");
const caption = document.querySelector("#caption");
const labels = document.querySelector("#labels");
const objects = document.querySelector("#objects");
const textSeen = document.querySelector("#textSeen");
const alerts = document.querySelector("#alerts");

let stream = null;
let timer = null;
let inFlight = false;

async function refreshHealth() {
  try {
    const response = await fetch("/api/health");
    const health = await response.json();
    setStatus(titleCase(health.model.state), health.model.state);
  } catch {
    setStatus("Offline", "error");
  }
}

async function startCamera() {
  setStatus("Opening camera", "loading");
  stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: "environment",
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
    audio: false,
  });
  video.srcObject = stream;
  emptyState.classList.add("hidden");
  startButton.disabled = true;
  stopButton.disabled = false;
  setStatus("Ready", "ready");
  captureAndRecognize();
  timer = window.setInterval(captureAndRecognize, 2000);
}

function stopCamera() {
  if (timer) {
    window.clearInterval(timer);
    timer = null;
  }
  if (stream) {
    for (const track of stream.getTracks()) {
      track.stop();
    }
    stream = null;
  }
  video.srcObject = null;
  emptyState.classList.remove("hidden");
  startButton.disabled = false;
  stopButton.disabled = true;
  setStatus("Stopped", "cold");
}

async function captureAndRecognize() {
  if (!stream || inFlight || video.readyState < 2) {
    return;
  }
  inFlight = true;
  setStatus("Recognizing", "loading");

  try {
    const blob = await captureFrame();
    const formData = new FormData();
    formData.append("image", blob, "frame.jpg");
    const response = await fetch("/api/recognize", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Recognition failed.");
    }
    renderResult(payload);
    setStatus("Ready", "ready");
    latencyText.textContent = `${payload.latency_ms} ms`;
  } catch (error) {
    setStatus(error.message || "Recognition failed", "error");
  } finally {
    inFlight = false;
  }
}

async function captureFrame() {
  const maxEdge = 896;
  const scale = Math.min(1, maxEdge / Math.max(video.videoWidth, video.videoHeight));
  canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
  canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
  const context = canvas.getContext("2d");
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error("Frame capture failed."))),
      "image/jpeg",
      0.82,
    );
  });
}

function renderResult(payload) {
  caption.textContent = payload.caption || "No caption returned.";
  renderChips(labels, payload.labels || []);
  renderList(objects, payload.objects || []);
  renderList(textSeen, payload.text_seen || []);
  renderList(alerts, payload.alerts || []);
}

function renderChips(container, items) {
  container.replaceChildren();
  for (const item of items.slice(0, 12)) {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = item;
    container.append(chip);
  }
}

function renderList(container, items) {
  container.replaceChildren();
  for (const item of items.slice(0, 12)) {
    const li = document.createElement("li");
    li.textContent = item;
    container.append(li);
  }
}

function setStatus(text, state) {
  statusText.textContent = text;
  statusDot.className = `statusDot ${state === "error" ? "error" : ""} ${
    state === "ready" ? "ready" : ""
  }`;
  if (state !== "ready") {
    latencyText.textContent = "";
  }
}

function titleCase(value) {
  if (!value) {
    return "Cold";
  }
  return `${value[0].toUpperCase()}${value.slice(1)}`;
}

startButton.addEventListener("click", () => {
  startCamera().catch((error) => {
    setStatus(error.message || "Camera unavailable", "error");
    stopCamera();
  });
});

stopButton.addEventListener("click", stopCamera);
refreshHealth();
