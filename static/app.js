"use strict";

const catalog = JSON.parse(document.getElementById("catalog-data").textContent);
let counts = JSON.parse(document.getElementById("counts-data").textContent);
const canvas = document.getElementById("drawing-canvas");
const ctx = canvas.getContext("2d");
const participant = document.getElementById("participant-id");
const selector = document.getElementById("symbol-select");
const status = document.getElementById("status");
const saveButton = document.getElementById("save-button");
const clearButton = document.getElementById("clear-button");
const exportButton = document.getElementById("export-button");
const exportStatus = document.getElementById("export-status");
const downloadLink = document.getElementById("download-link");
const predictButton = document.getElementById("predict-button");
const collectMode = document.getElementById("collect-mode");
const queryMode = document.getElementById("query-mode");
const candidates = document.getElementById("candidates");
const copyStatus = document.getElementById("copy-status");
let modelStatus = JSON.parse(document.getElementById("model-data").textContent);
let mode = "collection";
let busy = false;
let hasDrawing = false;
let activePointer = null;
let lastPoint = null;
let inputMethod = "unknown";

try {
  participant.value = localStorage.getItem("latex-participant") || "";
} catch (_error) {
  // El lienzo también funciona cuando el navegador bloquea el almacenamiento local.
}

function updateControls() {
  saveButton.disabled = busy;
  clearButton.disabled = busy;
  exportButton.disabled = busy;
  participant.disabled = busy;
  selector.disabled = busy || hasDrawing;
  predictButton.disabled = busy || !modelStatus.ready;
  collectMode.disabled = busy;
  queryMode.disabled = busy;
}

function showStatus(message, isError = false) {
  status.textContent = message;
  status.classList.toggle("error", isError);
}

function resetCanvas() {
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  hasDrawing = false;
  activePointer = null;
  lastPoint = null;
  inputMethod = "unknown";
  updateControls();
}

function updatePrompt() {
  const symbol = catalog.symbols.find(item => item.id === selector.value);
  document.getElementById("prompt-glyph").textContent = symbol.unicode;
  document.getElementById("prompt-name").textContent = symbol.name;
  document.getElementById("prompt-command").textContent = symbol.latex;
}

function updateCounts() {
  let total = 0;
  for (const symbol of catalog.symbols) {
    const count = counts[symbol.id] || 0;
    total += count;
    document.getElementById("count-" + symbol.id).textContent =
      count + " / " + catalog.target_per_class;
    document.getElementById("progress-" + symbol.id).value = count;
  }
  document.getElementById("total-count").textContent = total + " dibujos";
}

function chooseBalancedCategory() {
  const minimum = Math.min(...catalog.symbols.map(symbol => counts[symbol.id] || 0));
  selector.value = catalog.symbols.find(symbol => (counts[symbol.id] || 0) === minimum).id;
  updatePrompt();
}

function canvasPoint(event) {
  const box = canvas.getBoundingClientRect();
  const contentWidth = canvas.clientWidth;
  const contentHeight = canvas.clientHeight;
  return {
    x: (event.clientX - box.left - canvas.clientLeft) * canvas.width / contentWidth,
    y: (event.clientY - box.top - canvas.clientTop) * canvas.height / contentHeight
  };
}

canvas.addEventListener("pointerdown", event => {
  if (busy || activePointer !== null || event.button !== 0) return;
  event.preventDefault();
  candidates.replaceChildren();
  copyStatus.textContent = "";
  activePointer = event.pointerId;
  canvas.setPointerCapture(event.pointerId);
  inputMethod = event.pointerType || "unknown";
  lastPoint = canvasPoint(event);
  ctx.fillStyle = "#000000";
  ctx.beginPath();
  ctx.arc(lastPoint.x, lastPoint.y, 4, 0, 2 * Math.PI);
  ctx.fill();
  hasDrawing = true;
  updateControls();
});

canvas.addEventListener("pointermove", event => {
  if (busy || event.pointerId !== activePointer) return;
  event.preventDefault();
  const coalesced = event.getCoalescedEvents ? event.getCoalescedEvents() : [];
  const points = coalesced.length ? coalesced : [event];
  ctx.strokeStyle = "#000000";
  ctx.lineWidth = 8;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  for (const point of points) {
    const next = canvasPoint(point);
    ctx.beginPath();
    ctx.moveTo(lastPoint.x, lastPoint.y);
    ctx.lineTo(next.x, next.y);
    ctx.stroke();
    lastPoint = next;
  }
});

function stopDrawing(event) {
  if (event.pointerId === activePointer) {
    activePointer = null;
    lastPoint = null;
  }
}
canvas.addEventListener("pointerup", stopDrawing);
canvas.addEventListener("pointercancel", stopDrawing);
canvas.addEventListener("lostpointercapture", stopDrawing);

selector.addEventListener("change", updatePrompt);
clearButton.addEventListener("click", () => {
  resetCanvas();
  candidates.replaceChildren();
  copyStatus.textContent = "";
  showStatus(mode === "collection" ? "Lienzo limpio. Dibuja el símbolo indicado." : "Lienzo limpio. Dibuja un símbolo del catálogo.");
});

async function changeMode(nextMode) {
  if (busy || mode === nextMode) return;
  // Evita guardar un dibujo de consulta bajo la categoría anterior de recolección.
  mode = nextMode;
  resetCanvas();
  candidates.replaceChildren();
  copyStatus.textContent = "";
  document.getElementById("collection-fields").hidden = mode !== "collection";
  document.getElementById("collection-panel").hidden = mode !== "collection";
  document.getElementById("query-help").hidden = mode !== "query";
  document.getElementById("query-panel").hidden = mode !== "query";
  saveButton.hidden = mode !== "collection";
  predictButton.hidden = mode !== "query";
  collectMode.setAttribute("aria-pressed", String(mode === "collection"));
  queryMode.setAttribute("aria-pressed", String(mode === "query"));
  document.getElementById("mode-label").textContent = mode === "collection" ? "Modo de recolección" : "Modo de consulta";
  showStatus(mode === "collection" ? "Dibuja el símbolo indicado para guardar una muestra." : "Dibuja un símbolo y busca su comando.");
  if (mode === "query") {
    try {
      const response = await fetch("/api/model");
      if (!response.ok) throw new Error("No se pudo consultar el modelo.");
      modelStatus = await response.json();
      document.getElementById("model-status").textContent = modelStatus.message;
    } catch (error) {
      modelStatus = { ready: false };
      document.getElementById("model-status").textContent = error.message;
    }
  }
  updateControls();
}
collectMode.addEventListener("click", () => changeMode("collection"));
queryMode.addEventListener("click", () => changeMode("query"));

function displayCandidates(result) {
  candidates.replaceChildren();
  for (const candidate of result.candidates) {
    const card = document.createElement("article");
    card.className = "candidate";
    const heading = document.createElement("div");
    heading.className = "candidate-heading";
    const glyph = document.createElement("span");
    glyph.className = "candidate-glyph";
    glyph.textContent = candidate.unicode;
    const details = document.createElement("div");
    const name = document.createElement("strong");
    name.textContent = candidate.name;
    const command = document.createElement("code");
    command.textContent = candidate.latex;
    details.append(name, command);
    heading.append(glyph, details);
    const exampleLabel = document.createElement("p");
    exampleLabel.className = "help";
    exampleLabel.textContent = "Ejemplo en modo matemático:";
    const example = document.createElement("code");
    example.textContent = candidate.example_latex;
    const button = document.createElement("button");
    button.className = "secondary copy-command";
    button.type = "button";
    button.textContent = "Copiar " + candidate.latex;
    button.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(candidate.latex);
        copyStatus.textContent = "Copiado: " + candidate.latex;
      } catch (_error) {
        const selection = window.getSelection();
        const range = document.createRange();
        range.selectNodeContents(command);
        selection.removeAllRanges();
        selection.addRange(range);
        copyStatus.textContent = "Comando seleccionado. Presiona Ctrl+C para copiarlo.";
      }
    });
    card.append(heading, exampleLabel, example, button);
    candidates.append(card);
  }
}

predictButton.addEventListener("click", async () => {
  if (!hasDrawing) {
    showStatus("Dibuja un símbolo antes de consultar.", true);
    return;
  }
  busy = true;
  updateControls();
  candidates.replaceChildren();
  copyStatus.textContent = "";
  showStatus("Buscando coincidencias...");
  try {
    const response = await fetch("/predict", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: canvas.toDataURL("image/png") })
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "No se pudo consultar el símbolo.");
    displayCandidates(result);
    showStatus(result.message);
  } catch (error) {
    showStatus(error.message || "No se pudo conectar con la aplicación.", true);
  } finally {
    busy = false;
    updateControls();
  }
});

saveButton.addEventListener("click", async () => {
  if (!participant.value.trim()) {
    showStatus("Ingresa tu código de participante, por ejemplo P01.", true);
    participant.focus();
    return;
  }
  if (!hasDrawing) {
    showStatus("Dibuja el símbolo antes de guardarlo.", true);
    return;
  }
  const symbol = catalog.symbols.find(item => item.id === selector.value);
  const payload = {
    class_id: symbol.id,
    participant_id: participant.value.trim(),
    image: canvas.toDataURL("image/png"),
    input_method: inputMethod
  };
  busy = true;
  updateControls();
  showStatus("Guardando dibujo...");
  try {
    const response = await fetch("/upload", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "No se pudo guardar el dibujo.");
    participant.value = result.participant_id;
    try {
      localStorage.setItem("latex-participant", participant.value);
    } catch (_error) {}
    counts = result.counts;
    updateCounts();
    resetCanvas();
    chooseBalancedCategory();
    downloadLink.hidden = true;
    exportStatus.textContent = "Hay nuevos dibujos. Prepara los datos para actualizar la descarga.";
    showStatus("Dibujo de " + symbol.name + " guardado. Continúa con el siguiente símbolo.");
  } catch (error) {
    showStatus(error.message || "No se pudo conectar con la aplicación.", true);
  } finally {
    busy = false;
    updateControls();
  }
});

exportButton.addEventListener("click", async () => {
  busy = true;
  updateControls();
  downloadLink.hidden = true;
  exportStatus.textContent = "Preparando la colección...";
  try {
    const response = await fetch("/prepare", { method: "POST" });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "No se pudieron preparar los datos.");
    downloadLink.href = result.download_url;
    downloadLink.hidden = false;
    const metadata = result.metadata;
    exportStatus.textContent = metadata.n_samples + " muestras preparadas, de " +
      metadata.n_participants + " participante(s)." +
      (metadata.complete_collection ? " La meta de recolección está completa." :
        " Puedes probar la exportación; todavía falta completar la meta por categoría.");
  } catch (error) {
    exportStatus.textContent = error.message || "No se pudo conectar con la aplicación.";
  } finally {
    busy = false;
    updateControls();
  }
});

resetCanvas();
updateCounts();
chooseBalancedCategory();
