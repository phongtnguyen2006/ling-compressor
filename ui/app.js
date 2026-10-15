const state = { examples: [] };

const elements = {
  form: document.querySelector("#compress-form"),
  source: document.querySelector("#source-text"),
  ratio: document.querySelector("#ratio"),
  ratioOutput: document.querySelector("#ratio-output"),
  substitute: document.querySelector("#substitute"),
  modes: [...document.querySelectorAll('input[name="mode"]')],
  modeNote: document.querySelector("#mode-note"),
  button: document.querySelector("#compress-button"),
  characterCount: document.querySelector("#character-count"),
  exampleList: document.querySelector("#example-list"),
  error: document.querySelector("#error-message"),
  result: document.querySelector("#result"),
  tabs: [...document.querySelectorAll(".view-tab")],
  panels: [...document.querySelectorAll(".view-panel")],
};

function setView(view, updateHistory = true) {
  const nextView = view === "system" ? "system" : "workbench";
  elements.tabs.forEach(tab => {
    const selected = tab.dataset.view === nextView;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
  });
  elements.panels.forEach(panel => {
    panel.hidden = panel.id !== `${nextView}-view`;
  });
  if (updateHistory) {
    history.pushState({ view: nextView }, "", `#${nextView}`);
  }
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function escapeHtml(value) {
  return value.replace(/[&<>'"]/g, char => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[char]);
}

function annotateOriginal(text, deleted, vetoed) {
  const boundaries = new Set([0, text.length]);
  [...deleted, ...vetoed].forEach(item => {
    boundaries.add(item.span.start);
    boundaries.add(item.span.end);
  });
  const points = [...boundaries].sort((a, b) => a - b);
  let html = "";
  for (let index = 0; index < points.length - 1; index += 1) {
    const start = points[index];
    const end = points[index + 1];
    const chunk = escapeHtml(text.slice(start, end));
    const cut = deleted.find(item => item.span.start <= start && item.span.end >= end);
    const protectedSpan = vetoed.find(item => item.span.start <= start && item.span.end >= end);
    if (cut) {
      html += `<mark class="annotation-cut" title="Removed: ${escapeHtml(cut.reason)}">${chunk}</mark>`;
    } else if (protectedSpan) {
      html += `<mark class="annotation-protected" title="Protected: ${escapeHtml(protectedSpan.protect_class)}">${chunk}</mark>`;
    } else {
      html += chunk;
    }
  }
  return html;
}

function annotateCompressed(text, substitutions) {
  const replacements = Object.values(substitutions).filter(Boolean).sort((a, b) => b.length - a.length);
  if (!replacements.length) return escapeHtml(text);
  const escaped = replacements.map(value => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const pattern = new RegExp(`(${escaped.join("|")})`, "gi");
  return text.split(pattern).map(part => {
    const isReplacement = replacements.some(value => value.toLowerCase() === part.toLowerCase());
    return isReplacement ? `<mark class="annotation-substitution">${escapeHtml(part)}</mark>` : escapeHtml(part);
  }).join("");
}

function buildAudit(data) {
  const entries = [
    ...data.deleted.map(item => ({ kind: "Removed", text: item.text, detail: `${item.deprel} · ${item.score.toFixed(3)}` })),
    ...data.vetoed.map(item => ({ kind: "Protected", text: item.text, detail: item.protect_class })),
    ...Object.entries(data.substitutions).map(([from, to]) => ({ kind: "Shortened", text: `${from} → ${to}`, detail: "reversible" })),
  ];
  if (!entries.length) return '<p class="empty-audit">No edits were available at this setting.</p>';
  return entries.map(entry => `
    <div class="audit-item">
      <span class="audit-kind">${entry.kind}</span>
      <code>${escapeHtml(entry.text)}</code>
      <small>${escapeHtml(entry.detail)}</small>
    </div>`).join("");
}

function renderResult(data) {
  const hybrid = data.mode === "hybrid_sentinel";
  document.querySelector("#result-mode").textContent = hybrid
    ? "Guarded LLMLingua-2 · protected clauses are restored and checked"
    : "Tree pruning · each cut is recorded below";
  document.querySelector("#compressed-title").textContent = hybrid ? "Hybrid compressed" : "Compressed";
  document.querySelector("#original-title").textContent = hybrid ? "Original" : "Original, annotated";
  document.querySelector("#tokens-after").textContent = data.tokens_after;
  document.querySelector("#tokens-saved").textContent = data.saved_tokens;
  document.querySelector("#time-taken").textContent = Math.round(data.timings_ms.total);
  document.querySelector("#ruler-fill").style.width = `${Math.min(100, data.ratio * 100)}%`;
  document.querySelector("#original-count").textContent = `${data.tokens_before} tokens`;
  document.querySelector("#compressed-count").textContent = `${data.tokens_after} tokens · ${data.saved_percent}% saved`;
  document.querySelector("#annotated-text").innerHTML = hybrid
    ? escapeHtml(data.original)
    : annotateOriginal(data.original, data.deleted, data.vetoed);
  document.querySelector("#compressed-text").innerHTML = hybrid
    ? escapeHtml(data.compressed)
    : annotateCompressed(data.compressed, data.substitutions);
  document.querySelector("#audit-strip").hidden = hybrid;
  document.querySelector("#audit-details").hidden = hybrid;
  document.querySelector("#hybrid-report").hidden = !hybrid;
  if (hybrid) {
    const stats = data.hybrid_stats;
    document.querySelector("#masked-count").textContent = stats.masked_sents;
    document.querySelector("#clean-count").textContent = stats.clean_sents + stats.masked_sents;
    document.querySelector("#fallback-count").textContent = stats.fallback_sents + stats.validation_fallbacks;
    document.querySelector("#hybrid-validation").textContent = stats.validation_fallbacks
      ? `${stats.validation_fallbacks} model result(s) failed validation and used tree pruning instead.`
      : "Protected clauses were restored and checked. Model edits do not have a word-level audit or a full meaning guarantee.";
  } else {
    document.querySelector("#deletion-count").textContent = data.deleted.length;
    document.querySelector("#protected-count").textContent = data.vetoed.length;
    document.querySelector("#substitution-count").textContent = Object.keys(data.substitutions).length;
    document.querySelector("#audit-list").innerHTML = buildAudit(data);
  }
  elements.result.hidden = false;
}

function selectedMode() {
  return elements.modes.find(mode => mode.checked).value;
}

function updateMode() {
  const hybrid = selectedMode() === "hybrid_sentinel";
  elements.substitute.disabled = hybrid;
  elements.modeNote.textContent = hybrid
    ? "Experimental. First run loads LLMLingua-2; protected clauses are restored, while model edits have no word-level audit."
    : "Tree pruning shows each deletion and protection decision.";
}

async function compressText() {
  elements.error.hidden = true;
  elements.button.disabled = true;
  elements.button.textContent = selectedMode() === "hybrid_sentinel" ? "Loading or compressing…" : "Compressing…";
  try {
    const response = await fetch("/api/compress", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: elements.source.value,
        target_ratio: Number(elements.ratio.value) / 100,
        substitute: elements.substitute.checked,
        mode: selectedMode(),
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Compression failed.");
    renderResult(data);
  } catch (error) {
    elements.error.textContent = error.message;
    elements.error.hidden = false;
  } finally {
    elements.button.disabled = false;
    elements.button.textContent = "Compress text";
  }
}

function renderExamples(examples) {
  elements.exampleList.innerHTML = examples.map(example => `
    <button class="example-button" type="button" data-example-id="${escapeHtml(example.id)}">
      <strong>${escapeHtml(example.name)}</strong>
      <span>${escapeHtml(example.preview)}…</span>
    </button>`).join("");
  elements.exampleList.querySelectorAll("button").forEach(button => {
    button.addEventListener("click", () => {
      const example = state.examples.find(item => item.id === button.dataset.exampleId);
      elements.source.value = example.text;
      elements.modes.find(mode => mode.value === (example.mode || "conservative")).checked = true;
      updateMode();
      updateCharacterCount();
      document.querySelector("#workbench-title").scrollIntoView({ behavior: "smooth" });
      compressText();
    });
  });
}

async function loadExamples() {
  try {
    const response = await fetch("/api/examples");
    const data = await response.json();
    state.examples = data.examples;
    renderExamples(state.examples);
  } catch (_) {
    elements.exampleList.innerHTML = '<p class="loading-line">Examples could not be loaded.</p>';
  }
}

function updateCharacterCount() {
  const length = elements.source.value.length;
  elements.characterCount.textContent = `${length.toLocaleString()} character${length === 1 ? "" : "s"}`;
}

elements.form.addEventListener("submit", event => { event.preventDefault(); compressText(); });
elements.source.addEventListener("input", updateCharacterCount);
elements.ratio.addEventListener("input", () => { elements.ratioOutput.textContent = `${elements.ratio.value}%`; });
elements.modes.forEach(mode => mode.addEventListener("change", updateMode));
elements.tabs.forEach(tab => {
  tab.addEventListener("click", () => setView(tab.dataset.view));
  tab.addEventListener("keydown", event => {
    if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
    event.preventDefault();
    const currentIndex = elements.tabs.indexOf(tab);
    const delta = event.key === 'ArrowRight' ? 1 : -1;
    const next = elements.tabs[(currentIndex + delta + elements.tabs.length) % elements.tabs.length];
    next.focus();
    setView(next.dataset.view);
  });
});
window.addEventListener("popstate", () => setView(location.hash.slice(1), false));

updateCharacterCount();
updateMode();
loadExamples();
setView(location.hash.slice(1), false);
