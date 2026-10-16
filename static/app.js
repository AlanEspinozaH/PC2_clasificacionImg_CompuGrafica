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
const recognitionResult = document.getElementById("recognition-result");
const previewToggle = document.getElementById("preview-toggle");
const processedPreview = document.getElementById("processed-preview");
const processedPreviewImage = document.getElementById("processed-preview-image");
const unknownCaptureToggle = document.getElementById("unknown-capture-toggle");
const unknownCaptureForm = document.getElementById("unknown-capture-form");
const unknownParticipant = document.getElementById("unknown-participant-id");
const unknownKind = document.getElementById("unknown-kind");
const unknownSaveButton = document.getElementById("unknown-save-button");
const unknownSaveStatus = document.getElementById("unknown-save-status");
let modelStatus = JSON.parse(document.getElementById("model-data").textContent);
let mode = "collection";
let busy = false;
let hasDrawing = false;
let activePointer = null;
let lastPoint = null;
let inputMethod = "unknown";

try {
  participant.value = localStorage.getItem("latex-participant") || "";
  unknownParticipant.value = participant.value;
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
  unknownCaptureToggle.disabled = busy;
  unknownSaveButton.disabled = busy;
  unknownParticipant.disabled = busy;
  unknownKind.disabled = busy;
  previewToggle.disabled = busy;
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

function clearResult() {
  candidates.replaceChildren();
  copyStatus.textContent = "";
  recognitionResult.textContent = "";
  recognitionResult.classList.remove("rejected", "recognized");
  processedPreview.hidden = true;
  processedPreviewImage.removeAttribute("src");
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
  clearResult();
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
  clearResult();
  showStatus(mode === "collection" ? "Lienzo limpio. Dibuja el símbolo indicado." : "Lienzo limpio. Dibuja un símbolo del catálogo.");
});

async function changeMode(nextMode) {
  if (busy || mode === nextMode) return;
  // Evita guardar un dibujo de consulta bajo la categoría anterior de recolección.
  mode = nextMode;
  resetCanvas();
  clearResult();
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
  if (!result.recognized) {
    recognitionResult.className = "recognition-result rejected";
    recognitionResult.textContent = "Sin coincidencia confiable";
    return;
  }
  if (!Array.isArray(result.candidates) || result.candidates.length !== 3) {
    throw new Error("La respuesta del modelo no contiene las tres puntuaciones esperadas.");
  }
  const predictions = result.candidates;
  recognitionResult.className = "recognition-result recognized";
  const mainCandidate = result.candidates[0];
  recognitionResult.textContent =
    "Símbolo reconocido: " + mainCandidate.unicode + " — " + mainCandidate.name;

  predictions.forEach((candidate, index) => {
    const isPrincipal = index === 0;
    const card = document.createElement("article");
    card.className = isPrincipal ? "candidate candidate-principal" : "candidate";
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
    const example = document.createElement("p");
    example.className = "candidate-example";
    example.textContent = "Ejemplo: " + candidate.example_latex;
    heading.append(glyph, details);

    const score = candidate.score;
    if (typeof score !== "number" || !Number.isFinite(score) || score < 0 || score > 1) {
      throw new Error("La respuesta del modelo contiene una puntuación inválida.");
    }
    const scoreLabel = document.createElement("p");
    scoreLabel.className = "prediction-confidence";
    scoreLabel.textContent =
      (isPrincipal ? "Puntuación del modelo (Softmax): " : "Puntuación Softmax: ") +
      (score * 100).toFixed(1) + "%";
    const bar = document.createElement("progress");
    bar.className = "prediction-bar";
    bar.max = 1;
    bar.value = score;
    bar.setAttribute("aria-label", "Puntuación Softmax de " + candidate.name);

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
        copyStatus.textContent =
          "Comando seleccionado. Presiona Ctrl+C para copiarlo.";
      }
    });
    card.append(heading, example, scoreLabel, bar, button);
    candidates.append(card);
  });
}


predictButton.addEventListener("click", async () => {
  if (!hasDrawing) {
    showStatus("Dibuja un símbolo antes de consultar.", true);
    return;
  }
  busy = true;
  updateControls();
  clearResult();
  showStatus("Buscando coincidencias...");
  try {
    const response = await fetch("/predict", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image: canvas.toDataURL("image/png"),
        include_preview: previewToggle.checked
      })
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "No se pudo consultar el símbolo.");
    displayCandidates(result);
    if (result.processed_preview) {
      processedPreviewImage.src = result.processed_preview;
      processedPreview.hidden = false;
    }
    showStatus(result.message);
  } catch (error) {
    showStatus(error.message || "No se pudo conectar con la aplicación.", true);
  } finally {
    busy = false;
    updateControls();
  }
});

unknownCaptureToggle.addEventListener("click", () => {
  unknownCaptureForm.hidden = !unknownCaptureForm.hidden;
  unknownSaveStatus.textContent = "";
  if (!unknownCaptureForm.hidden && !unknownParticipant.value) {
    unknownParticipant.value = participant.value;
  }
});

unknownSaveButton.addEventListener("click", async () => {
  if (!hasDrawing) {
    unknownSaveStatus.textContent = "Dibuja el símbolo desconocido antes de guardarlo.";
    return;
  }
  busy = true;
  updateControls();
  unknownSaveStatus.textContent = "Guardando muestra separada...";
  try {
    const response = await fetch("/unknown", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image: canvas.toDataURL("image/png"),
        unknown_kind: unknownKind.value,
        participant_id: unknownParticipant.value.trim(),
        input_method: inputMethod
      })
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "No se pudo guardar la muestra.");
    unknownParticipant.value = result.participant_id;
    participant.value = result.participant_id;
    try {
      localStorage.setItem("latex-participant", result.participant_id);
    } catch (_error) {}
    const total = Object.values(result.counts).reduce((sum, count) => sum + count, 0);
    unknownSaveStatus.textContent =
      "Muestra guardada fuera de las seis clases. Total desconocido: " + total + ".";
    resetCanvas();
    clearResult();
  } catch (error) {
    unknownSaveStatus.textContent = error.message || "No se pudo conectar con la aplicación.";
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
    unknownParticipant.value = result.participant_id;
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
