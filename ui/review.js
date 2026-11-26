"use strict";

const fields = ["meaning_preserved", "grammatical", "harmful_change"];
const ui = {
  file: document.querySelector("#packet-file"),
  reviewer: document.querySelector("#reviewer-name"),
  workspace: document.querySelector("#review-workspace"),
  error: document.querySelector("#review-error"),
  position: document.querySelector("#review-position"),
  counts: document.querySelector("#review-counts"),
  original: document.querySelector("#original-prompt"),
  proposed: document.querySelector("#proposed-prompt"),
  form: document.querySelector("#judgment-form"),
  notes: document.querySelector("#review-notes"),
  previous: document.querySelector("#previous-prompt"),
  next: document.querySelector("#next-prompt"),
  export: document.querySelector("#export-review"),
};
let rows = [];
let current = 0;
let packetName = "review-packet";
let dirty = false;

function sameReviewer(left, right) {
  const normalize = value => value.normalize("NFKC").trim().replace(/\s+/g, " ").toLocaleLowerCase();
  return normalize(left) === normalize(right);
}

function showError(message) {
  ui.error.textContent = message;
  ui.error.hidden = false;
}

function validateRow(row, seen) {
  if (!row || typeof row !== "object" || Array.isArray(row)) throw new Error("Every packet line must be a JSON object.");
  if (typeof row.review_id !== "string" || !row.review_id || seen.has(row.review_id)) throw new Error("The packet has a missing or duplicate review ID.");
  seen.add(row.review_id);
  if (typeof row.original !== "string" || typeof row.suggested !== "string" || typeof row.doc_id !== "string") throw new Error("A prompt is missing its original or proposed text.");
  for (const field of fields) {
    if (row[field] !== null && typeof row[field] !== "boolean") throw new Error(`Invalid ${field} judgment in packet.`);
  }
  if (row.reviewer !== null && (typeof row.reviewer !== "string" || !row.reviewer.trim())) throw new Error("Invalid reviewer name in packet.");
  if (typeof row.notes !== "string") throw new Error("Invalid notes in packet.");
}

function complete(row) { return fields.every(field => typeof row[field] === "boolean"); }

function saveCurrent() {
  if (!rows.length) return;
  const row = rows[current];
  for (const field of fields) {
    const selected = ui.form.querySelector(`input[name="${field}"]:checked`);
    row[field] = selected ? (selected.value === "null" ? null : selected.value === "true") : null;
  }
  row.notes = ui.notes.value;
}

function render() {
  if (!rows.length) return;
  const row = rows[current];
  ui.position.textContent = `Prompt ${current + 1} of ${rows.length}`;
  ui.counts.textContent = `${rows.filter(complete).length} fully judged · ${rows.length - rows.filter(complete).length} pending or unsure`;
  ui.original.textContent = row.original;
  ui.proposed.textContent = row.suggested;
  for (const field of fields) {
    for (const radio of ui.form.querySelectorAll(`input[name="${field}"]`)) {
      radio.checked = row[field] === null ? radio.value === "null" : radio.value === String(row[field]);
    }
  }
  ui.notes.value = row.notes;
  ui.previous.disabled = current === 0;
  ui.next.disabled = current === rows.length - 1;
}

ui.file.addEventListener("change", async () => {
  ui.error.hidden = true;
  const file = ui.file.files && ui.file.files[0];
  if (!file) return;
  if (dirty && !window.confirm("Opening another packet will discard edits that have not been exported. Continue?")) {
    ui.file.value = "";
    return;
  }
  try {
    const lines = (await file.text()).split(/\r?\n/).filter(line => line.trim());
    if (!lines.length) throw new Error("The packet is empty.");
    const parsed = lines.map((line, index) => {
      try { return JSON.parse(line); }
      catch { throw new Error(`Line ${index + 1} is not valid JSON.`); }
    });
    const seen = new Set();
    for (const row of parsed) validateRow(row, seen);
    const priorNames = parsed.map(row => row.reviewer).filter(Boolean);
    if (priorNames.some(name => !sameReviewer(name, priorNames[0]))) throw new Error("This file mixes reviewer names. Open a single person's packet.");
    if (priorNames.length) ui.reviewer.value = priorNames[0];
    rows = parsed;
    current = 0;
    dirty = false;
    packetName = file.name.replace(/\.jsonl$/i, "");
    ui.workspace.hidden = false;
    render();
  } catch (error) {
    if (!rows.length) ui.workspace.hidden = true;
    showError(error.message);
  } finally {
    ui.file.value = "";
  }
});

ui.form.addEventListener("change", () => { saveCurrent(); dirty = true; render(); });
ui.notes.addEventListener("input", () => { saveCurrent(); dirty = true; });
ui.previous.addEventListener("click", () => { saveCurrent(); current -= 1; render(); });
ui.next.addEventListener("click", () => { saveCurrent(); current += 1; render(); });

ui.export.addEventListener("click", () => {
  ui.error.hidden = true;
  saveCurrent();
  const name = ui.reviewer.value.trim();
  if (!name) { showError("Enter your reviewer name before exporting."); return; }
  for (const row of rows) {
    if (row.reviewer && !sameReviewer(row.reviewer, name)) { showError("The packet belongs to another reviewer. Use a separate packet."); return; }
  }
  const exported = rows.map(row => ({
    ...row,
    reviewer: fields.some(field => row[field] !== null) ? name : null,
  }));
  const content = exported.map(row => JSON.stringify(row)).join("\n") + "\n";
  const url = URL.createObjectURL(new Blob([content], { type: "application/x-ndjson" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = `${packetName}-${name.replace(/[^a-z0-9_-]+/gi, "-") || "reviewer"}.jsonl`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  dirty = false;
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

window.addEventListener("beforeunload", event => {
  if (!dirty) return;
  event.preventDefault();
  event.returnValue = "";
});
