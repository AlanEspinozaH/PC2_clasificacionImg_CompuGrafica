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
  showStatus("Lienzo limpio. Dibuja el símbolo indicado.");
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
