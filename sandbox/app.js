"use strict";

const cells = [];
let token = "";
let executing = false;
const sessionStatus = document.querySelector("#session-status");

async function request(path, body) {
  const options = body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Sandbox-Token": token },
    body: JSON.stringify(body),
  };
  const response = await fetch(path, { ...options, cache: "no-store" });
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error("El servidor devolvió una respuesta no válida. Comprueba que sigue en marcha.");
  }
  if (!response.ok) throw new Error(data.error || `Error HTTP ${response.status}`);
  return data;
}

function updateCell(cell) {
  const dirty = cell.editor.value !== cell.savedSource;
  cell.node.classList.toggle("unsaved", dirty);
  cell.savedState.textContent = dirty ? "Cambios sin guardar" : cell.modified ? "Copia local guardada" : "Ejemplo original";
  cell.run.disabled = executing || cell.busy;
  cell.save.disabled = cell.busy || !dirty;
  cell.reset.disabled = cell.busy || (!dirty && !cell.modified);
  cell.editor.disabled = cell.busy;
  cell.nav.classList.toggle("has-edits", dirty || cell.modified);
  if (cell.lastRunSource !== undefined && !cell.busy) {
    const stale = cell.editor.value !== cell.lastRunSource;
    cell.output.classList.toggle("stale", stale);
    cell.outputNote.hidden = !stale && !cell.runNote;
    cell.outputNote.textContent = [stale ? "Esta salida corresponde a una versión anterior del código." : "", cell.runNote].filter(Boolean).join(" ");
  }
}

function updateButtons() {
  for (const cell of cells) updateCell(cell);
}

function updateLines(cell) {
  const count = cell.editor.value.split("\n").length;
  cell.lines.textContent = Array.from({ length: count }, (_, i) => i + 1).join("\n");
  cell.syncHighlight?.();
}

async function act(cell, action) {
  if (cell.busy || (action === "run" && executing)) return;
  if (action === "reset" && !window.confirm("¿Restaurar el ejemplo original? Se reemplazarán los cambios de esta celda.")) return;
  const source = cell.editor.value;
  cell.busy = true;
  cell.message.classList.remove("error");
  cell.message.textContent = action === "run" ? "Compilando y ejecutando…" : action === "save" ? "Guardando…" : "Restaurando…";
  if (action === "run") {
    executing = true;
    cell.status.textContent = "En ejecución";
    cell.status.dataset.state = "running";
    sessionStatus.textContent = `Ejecutando: ${cell.title}`;
  }
  updateButtons();
  try {
    const result = await request(`/api/cells/${encodeURIComponent(cell.id)}/${action}`, action === "reset" ? {} : { source });
    if (action === "run") {
      cell.savedSource = source;
      cell.modified = typeof result.modified === "boolean"
        ? result.modified
        : source !== cell.originalSource || cell.originalSource === undefined;
      cell.lastRunSource = source;
      const ok = result.exit_code === 0 && !result.timed_out;
      cell.status.textContent = result.timed_out ? "Tiempo agotado" : ok ? "Completado" : "Error";
      cell.status.dataset.state = ok ? "success" : "error";
      cell.nav.dataset.state = ok ? "success" : "error";
      const seconds = (result.duration_ms / 1000).toFixed(2);
      cell.meta.textContent = `${seconds} s · Código de salida: ${result.exit_code ?? "—"}`;
      cell.stdout.textContent = result.stdout || "";
      cell.stderr.textContent = result.stderr || "";
      cell.stdoutBlock.hidden = !result.stdout;
      cell.stderrBlock.hidden = !result.stderr;
      cell.placeholder.hidden = Boolean(result.stdout || result.stderr);
      cell.placeholder.textContent = "El programa terminó sin imprimir salida.";
      cell.runNote = [result.timed_out ? "Se alcanzó el límite de tiempo de ejecución." : "", result.output_truncated ? "La salida se ha truncado por alcanzar el límite de tamaño." : ""].filter(Boolean).join(" ");
      cell.message.textContent = ok ? "Ejecutado y guardado" : "Código guardado; revisa el resultado";
      sessionStatus.textContent = ok ? "Ejecución completada" : "Ejecución finalizada con errores";
    } else {
      cell.editor.value = result.source;
      cell.savedSource = result.source;
      cell.modified = result.modified;
      if (action === "reset") cell.originalSource = result.source;
      updateLines(cell);
      cell.message.textContent = action === "reset" ? "Ejemplo original restaurado" : "Cambios guardados";
    }
  } catch (error) {
    cell.message.textContent = error.message;
    cell.message.classList.add("error");
    if (action === "run") {
      cell.status.textContent = "Error de ejecución";
      cell.status.dataset.state = "error";
      sessionStatus.textContent = "No se pudo completar la ejecución";
    }
  } finally {
    cell.busy = false;
    if (action === "run") executing = false;
    updateButtons();
  }
}

function addCell(data, index) {
  const fragment = document.querySelector("#cell-template").content.cloneNode(true);
  const node = fragment.querySelector(".cell");
  const select = (selector) => node.querySelector(selector);
  node.id = `cell-${data.id}`;
  select(".cell-number").textContent = String(index + 1).padStart(2, "0");
  select("h2").textContent = data.title;
  select(".cell-description").textContent = data.description;
  const editor = select("textarea");
  editor.id = `editor-${data.id}`;
  editor.value = data.source;
  editor.rows = Math.max(8, Math.min(22, data.source.split("\n").length));
  const label = select("label");
  label.htmlFor = editor.id;
  label.textContent = `${data.id}.rs`;
  const nav = document.createElement("a");
  nav.href = `#${node.id}`;
  const number = document.createElement("span");
  number.className = "nav-number";
  number.textContent = String(index + 1).padStart(2, "0");
  const title = document.createElement("span");
  title.textContent = data.title;
  nav.append(number, title);
  nav.addEventListener("click", () => {
    for (const item of cells) item.nav.removeAttribute("aria-current");
    nav.setAttribute("aria-current", "location");
  });
  document.querySelector("#navigation").append(nav);
  const cell = {
    ...data, node, editor, nav, savedSource: data.source,
    originalSource: data.modified ? undefined : data.source, busy: false,
    lines: select(".line-numbers"), savedState: select(".saved-state"),
    run: select(".run-button"), save: select(".save-button"), reset: select(".reset-button"),
    message: select(".cell-message"), status: select(".cell-status"),
    output: select(".output"), meta: select(".output-meta"),
    stdout: select(".stdout"), stderr: select(".stderr"),
    stdoutBlock: select(".stdout-block"), stderrBlock: select(".stderr-block"),
    placeholder: select(".output-placeholder"), outputNote: select(".output-note"),
  };
  cell.run.addEventListener("click", () => act(cell, "run"));
  cell.save.addEventListener("click", () => act(cell, "save"));
  cell.reset.addEventListener("click", () => act(cell, "reset"));
  editor.addEventListener("input", () => {
    cell.message.textContent = "";
    updateLines(cell);
    updateCell(cell);
  });
  editor.addEventListener("scroll", () => { cell.lines.scrollTop = editor.scrollTop; });
  editor.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      act(cell, "run");
    }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
      event.preventDefault();
      act(cell, "save");
    }
  });
  cells.push(cell);
  updateLines(cell);
  updateCell(cell);
  document.querySelector("#notebook").append(fragment);
  cell.syncHighlight = window.RustEditor.attach(cell, request);
}

window.addEventListener("beforeunload", (event) => {
  if (executing || cells.some((cell) => cell.editor.value !== cell.savedSource)) {
    event.preventDefault();
    event.returnValue = "";
  }
});

async function loadNotebook() {
  try {
    const notebook = await request("/api/notebook");
    token = notebook.token;
    document.querySelector("#notebook-title").textContent = notebook.title;
    document.querySelector("#navigation").replaceChildren();
    notebook.cells.forEach(addCell);
    document.querySelector("#cell-count").textContent = `${cells.length} ejemplos · Celdas independientes`;
    sessionStatus.textContent = "Listo para experimentar";
  } catch (error) {
    const banner = document.querySelector("#connection-error");
    banner.textContent = `${error.message} Abre este cuaderno desde el servidor local y recarga la página.`;
    banner.hidden = false;
    sessionStatus.textContent = "Sin conexión";
  }
}

loadNotebook();
