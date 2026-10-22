const state = { examples: [] };

const elements = {
  form: document.querySelector("#compress-form"),
  source: document.querySelector("#source-text"),
  standardControls: document.querySelector("#standard-controls"),
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

  // Semantic settings elements
  semanticSettings: document.querySelector("#semantic-settings"),
  semanticArm: document.querySelector("#semantic-arm"),
  semanticArmCaption: document.querySelector("#semantic-arm-caption"),
  semanticEncoding: document.querySelector("#semantic-encoding"),
  semanticVectorSource: document.querySelector("#semantic-vector-source"),
  syntheticVectorWarning: document.querySelector("#synthetic-vector-warning"),
  semanticVectorPathGroup: document.querySelector("#semantic-vector-path-group"),
  semanticVectorPath: document.querySelector("#semantic-vector-path"),
  semanticPrice: document.querySelector("#semantic-price"),
  semanticMaxCandidates: document.querySelector("#semantic-max-candidates"),
  semanticSlotsGroup: document.querySelector("#semantic-slots-group"),
  semanticEmbeddingSlots: document.querySelector("#semantic-embedding-slots"),
  semanticWordnet: document.querySelector("#semantic-wordnet"),
  semanticSimilarityToggle: document.querySelector("#semantic-similarity-toggle"),
  semanticSimilarityControls: document.querySelector("#semantic-similarity-controls"),
  semanticMinSimilarity: document.querySelector("#semantic-min-similarity"),
  semanticMinSimilarityOutput: document.querySelector("#semantic-min-similarity-output"),
  semanticAdaptiveSimilarity: document.querySelector("#semantic-adaptive-similarity"),

  // Semantic result elements
  costSavedSummary: document.querySelector("#cost-saved-summary"),
  semanticBanner: document.querySelector("#semantic-banner"),
  reviewStatusBadge: document.querySelector("#review-status-badge"),
  tokenCounterBadge: document.querySelector("#token-counter-badge"),
  vectorLabelBadge: document.querySelector("#vector-label-badge"),
  configHashBadge: document.querySelector("#config-hash-badge"),
  costSavedBadge: document.querySelector("#cost-saved-badge"),
  semanticWarningsList: document.querySelector("#semantic-warnings-list"),
  deletionLabel: document.querySelector("#deletion-label"),
  protectedLabel: document.querySelector("#protected-label"),
  substitutionLabel: document.querySelector("#substitution-label"),
  auditSummaryLabel: document.querySelector("#audit-summary-label"),
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
  if (value == null) return "";
  return String(value).replace(/[&<>'"]/g, char => ({
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

function codepointOffsets(text) {
  const offsets = [0];
  let units = 0;
  for (const character of text) {
    units += character.length;
    offsets.push(units);
  }
  return offsets;
}

function annotateSemantic(original, compressed, edits, decisions) {
  // Python spans count Unicode code points; JavaScript slices count UTF-16 units.
  const offsets = codepointOffsets(original);
  const positionedEdits = edits.map(edit => ({
    ...edit, start: offsets[edit.span.start], end: offsets[edit.span.end],
  }));
  // Original annotations: edits as substitutions, and rejected decisions with protection as protected
  const origBoundaries = new Set([0, original.length]);
  positionedEdits.forEach(e => {
    origBoundaries.add(e.start);
    origBoundaries.add(e.end);
  });
  const protectedDecisions = (decisions || []).filter(
    d => d.decision === "rejected" && d.reason && d.reason.startsWith("protected")
  ).map(decision => ({
    ...decision, start: offsets[decision.span.start], end: offsets[decision.span.end],
  }));
  protectedDecisions.forEach(d => {
    origBoundaries.add(d.start);
    origBoundaries.add(d.end);
  });
  const origPoints = [...origBoundaries].sort((a, b) => a - b);
  let origHtml = "";
  for (let i = 0; i < origPoints.length - 1; i += 1) {
    const start = origPoints[i];
    const end = origPoints[i + 1];
    const chunk = escapeHtml(original.slice(start, end));
    const edit = positionedEdits.find(e => e.start <= start && e.end >= end);
    const prot = protectedDecisions.find(d => d.start <= start && d.end >= end);
    if (edit) {
      const simText = edit.similarity != null ? `, cos: ${edit.similarity.toFixed(3)}` : "";
      origHtml += `<mark class="annotation-substitution" title="Replaced with '${escapeHtml(edit.replacement)}' (${escapeHtml(edit.provenance)}${simText}, -${edit.token_delta} tok)">${chunk}</mark>`;
    } else if (prot) {
      origHtml += `<mark class="annotation-protected" title="Protected: ${escapeHtml(prot.reason)}">${chunk}</mark>`;
    } else {
      origHtml += chunk;
    }
  }

  // Compressed annotations: highlight the exact occurrence replaced in compressed output
  const sortedEdits = [...positionedEdits].sort((a, b) => a.start - b.start);
  let compHtml = "";
  let origCursor = 0;
  let compCursor = 0;
  for (const edit of sortedEdits) {
    const prefixLen = edit.start - origCursor;
    const prefix = compressed.slice(compCursor, compCursor + prefixLen);
    compHtml += escapeHtml(prefix);
    compCursor += prefixLen;
    origCursor = edit.end;

    const replLen = edit.replacement.length;
    const replText = compressed.slice(compCursor, compCursor + replLen);
    const sim = edit.similarity != null ? `, cos: ${edit.similarity.toFixed(3)}` : "";
    const delta = edit.token_delta != null ? `, -${edit.token_delta} tok` : "";
    compHtml += `<mark class="annotation-substitution" title="Substituted for '${escapeHtml(edit.original)}' (${escapeHtml(edit.provenance)}${sim}${delta})">${escapeHtml(replText)}</mark>`;
    compCursor += replLen;
  }
  compHtml += escapeHtml(compressed.slice(compCursor));

  return { origHtml, compHtml };
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

function buildSemanticAudit(data) {
  const edits = data.edits || [];
  const decisions = data.decisions || [];
  const rejected = decisions.filter(d => d.decision !== "selected");

  if (!edits.length && !rejected.length) {
    return '<p class="empty-audit">No candidates were evaluated for this text.</p>';
  }

  let html = "";

  if (edits.length) {
    html += `<div class="audit-section-header"><h4>Selected edits (${edits.length})</h4></div>`;
    html += edits.map(edit => {
      const sim = edit.similarity != null ? `cos: ${edit.similarity.toFixed(3)}` : "no vector score";
      const delta = edit.token_delta != null ? `-${edit.token_delta} tok` : "";
      return `
        <div class="audit-item audit-item-selected">
          <span class="audit-kind audit-kind-selected">Selected</span>
          <div class="audit-content">
            <div class="audit-terms">
              <code>${escapeHtml(edit.original)}</code>
              <span class="audit-arrow">→</span>
              <code class="audit-replacement">${escapeHtml(edit.replacement)}</code>
            </div>
            <div class="audit-context">
              <span class="context-label">Context:</span>
              <span class="context-before">${escapeHtml(edit.sentence_before)}</span>
              <span class="audit-arrow">→</span>
              <span class="context-after">${escapeHtml(edit.sentence_after)}</span>
            </div>
          </div>
          <div class="audit-meta">
            <small class="audit-detail">${escapeHtml(edit.provenance)} · ${sim} · ${delta}</small>
            <small class="audit-span">span ${edit.span.start}:${edit.span.end}</small>
          </div>
        </div>`;
    }).join("");
  }

  if (rejected.length) {
    html += `<div class="audit-section-header"><h4>Rejected candidates (${rejected.length})</h4></div>`;
    html += rejected.map(item => {
      const sim = item.similarity != null ? `cos: ${item.similarity.toFixed(3)}` : "no vector score";
      const delta = item.token_delta != null ? `${item.token_delta > 0 ? '-' + item.token_delta : item.token_delta} tok` : "0 tok";
      return `
        <div class="audit-item audit-item-rejected">
          <span class="audit-kind audit-kind-rejected">Rejected</span>
          <div class="audit-content">
            <div class="audit-terms">
              <code>${escapeHtml(item.original)}</code>
              <span class="audit-arrow">→</span>
              <code class="audit-candidate">${escapeHtml(item.replacement)}</code>
              <span class="reject-reason-badge">${escapeHtml(item.reason)}</span>
            </div>
          </div>
          <div class="audit-meta">
            <small class="audit-detail">${escapeHtml(item.provenance)} · ${sim} · ${delta}</small>
            <small class="audit-span">span ${item.span.start}:${item.span.end} · rank ${item.candidate_rank ?? 0}</small>
          </div>
        </div>`;
    }).join("");
  }

  return html;
}

function renderResult(data) {
  const mode = data.mode;
  const isHybrid = mode === "hybrid_sentinel";
  const isSemantic = mode === "semantic";

  if (isSemantic) {
    const armLabel = data.provider_metadata && data.provider_metadata.arm
      ? data.provider_metadata.arm.toUpperCase()
      : "B0";
    document.querySelector("#result-mode").textContent =
      `Semantic substitution · ${armLabel} · ${data.token_counter || 'tiktoken'} · review pending`;
  } else if (isHybrid) {
    document.querySelector("#result-mode").textContent =
      "Guarded LLMLingua-2 · protected clauses are restored and checked";
  } else {
    document.querySelector("#result-mode").textContent =
      "Tree pruning · each cut is recorded below";
  }

  document.querySelector("#compressed-title").textContent =
    isSemantic ? "Proposed output" : isHybrid ? "Hybrid compressed" : "Compressed";
  document.querySelector("#original-title").textContent =
    isHybrid ? "Original" : "Original, annotated";

  document.querySelector("#tokens-after").textContent = data.tokens_after;
  document.querySelector("#tokens-saved").textContent = data.saved_tokens;
  const totalMs = data.timings_ms ? (data.timings_ms.total != null ? data.timings_ms.total : 0) : 0;
  document.querySelector("#time-taken").textContent = Math.round(totalMs);
  document.querySelector("#ruler-fill").style.width = `${Math.min(100, Math.max(0, data.ratio * 100))}%`;
  document.querySelector("#original-count").textContent = `${data.tokens_before} tokens`;
  document.querySelector("#compressed-count").textContent = `${data.tokens_after} tokens · ${data.saved_percent}% saved`;

  // Cost saved summary in ruler
  if (elements.costSavedSummary) {
    if (isSemantic && data.cost_saved_usd != null) {
      elements.costSavedSummary.textContent = ` · $${data.cost_saved_usd.toFixed(6)} saved`;
    } else {
      elements.costSavedSummary.textContent = "";
    }
  }

  // Semantic metadata banner & warnings
  if (elements.semanticBanner) {
    elements.semanticBanner.hidden = !isSemantic;
    if (isSemantic) {
      if (elements.reviewStatusBadge) {
        elements.reviewStatusBadge.textContent = `Review: ${data.review_status || "pending"}`;
      }
      if (elements.tokenCounterBadge) {
        elements.tokenCounterBadge.textContent = `Tokenizer: ${data.token_counter || "tiktoken"}`;
      }
      if (elements.vectorLabelBadge) {
        elements.vectorLabelBadge.textContent = `Vectors: ${data.vector_label || "none"}`;
      }
      if (elements.configHashBadge) {
        elements.configHashBadge.textContent = `Config: ${(data.config_hash || "unknown").slice(0, 12)}`;
        elements.configHashBadge.title = data.config_hash || "Configuration hash unavailable";
      }
      if (elements.costSavedBadge) {
        const price = data.input_price_per_million != null ? data.input_price_per_million : 0;
        elements.costSavedBadge.textContent = `Est. saved: $${(data.cost_saved_usd || 0).toFixed(6)} (@ $${price.toFixed(2)}/1M)`;
      }
      if (elements.semanticWarningsList) {
        const warnings = data.warnings || [];
        if (warnings.length > 0) {
          elements.semanticWarningsList.innerHTML = warnings.map(w => `<div class="warning-callout">${escapeHtml(w)}</div>`).join("");
          elements.semanticWarningsList.hidden = false;
        } else {
          elements.semanticWarningsList.innerHTML = "";
          elements.semanticWarningsList.hidden = true;
        }
      }
    }
  }

  // Annotations
  if (isSemantic) {
    const { origHtml, compHtml } = annotateSemantic(data.original, data.compressed, data.edits || [], data.decisions || []);
    document.querySelector("#annotated-text").innerHTML = origHtml;
    document.querySelector("#compressed-text").innerHTML = compHtml;
  } else if (isHybrid) {
    document.querySelector("#annotated-text").innerHTML = escapeHtml(data.original);
    document.querySelector("#compressed-text").innerHTML = escapeHtml(data.compressed);
  } else {
    document.querySelector("#annotated-text").innerHTML = annotateOriginal(data.original, data.deleted || [], data.vetoed || []);
    document.querySelector("#compressed-text").innerHTML = annotateCompressed(data.compressed, data.substitutions || {});
  }

  // Audit and Reports
  document.querySelector("#hybrid-report").hidden = !isHybrid;
  document.querySelector("#audit-strip").hidden = isHybrid;
  document.querySelector("#audit-details").hidden = isHybrid;

  if (isHybrid) {
    const stats = data.hybrid_stats || { masked_sents: 0, clean_sents: 0, fallback_sents: 0, validation_fallbacks: 0 };
    document.querySelector("#masked-count").textContent = stats.masked_sents;
    document.querySelector("#clean-count").textContent = stats.clean_sents + stats.masked_sents;
    document.querySelector("#fallback-count").textContent = stats.fallback_sents + stats.validation_fallbacks;
    document.querySelector("#hybrid-validation").textContent = stats.validation_fallbacks
      ? `${stats.validation_fallbacks} model result(s) failed validation and used tree pruning instead.`
      : "Protected clauses were restored and checked. Model edits do not have a word-level audit or a full meaning guarantee.";
  } else if (isSemantic) {
    const editsCount = (data.edits || []).length;
    const decisions = data.decisions || [];
    const rejectedCount = decisions.filter(d => d.decision !== "selected").length;
    document.querySelector("#deletion-count").textContent = editsCount;
    if (elements.deletionLabel) elements.deletionLabel.textContent = "edits selected";
    document.querySelector("#protected-count").textContent = rejectedCount;
    if (elements.protectedLabel) elements.protectedLabel.textContent = "candidates rejected";
    document.querySelector("#substitution-count").textContent = `$${(data.cost_saved_usd || 0).toFixed(6)}`;
    if (elements.substitutionLabel) elements.substitutionLabel.textContent = "estimated input savings";
    if (elements.auditSummaryLabel) {
      elements.auditSummaryLabel.textContent = `Open the substitution audit (${editsCount} selected, ${rejectedCount} rejected)`;
    }
    document.querySelector("#audit-list").innerHTML = buildSemanticAudit(data);
  } else {
    document.querySelector("#deletion-count").textContent = (data.deleted || []).length;
    if (elements.deletionLabel) elements.deletionLabel.textContent = "cuts made";
    document.querySelector("#protected-count").textContent = (data.vetoed || []).length;
    if (elements.protectedLabel) elements.protectedLabel.textContent = "spans protected";
    document.querySelector("#substitution-count").textContent = Object.keys(data.substitutions || {}).length;
    if (elements.substitutionLabel) elements.substitutionLabel.textContent = "phrases shortened";
    if (elements.auditSummaryLabel) elements.auditSummaryLabel.textContent = "Open the audit trail";
    document.querySelector("#audit-list").innerHTML = buildAudit(data);
  }

  elements.result.hidden = false;
}

function selectedMode() {
  const checked = elements.modes.find(mode => mode.checked);
  return checked ? checked.value : "conservative";
}

function updateSemanticSettings() {
  if (!elements.semanticArm) return;
  const arm = elements.semanticArm.value;
  const isB0 = arm === "b0";
  const isB2 = arm === "b2";

  // Arm caption
  if (elements.semanticArmCaption) {
    if (isB0) {
      elements.semanticArmCaption.textContent = "B0 uses curated rules with optional WordNet. Vectors are not required.";
    } else if (arm === "b1") {
      elements.semanticArmCaption.textContent = "B1 ranks the baseline candidate pool using cosine similarity to local vectors.";
    } else {
      elements.semanticArmCaption.textContent = "B2 explores vector neighbors alongside the baseline candidate pool.";
    }
  }

  // Vector source availability
  if (elements.semanticVectorSource) {
    elements.semanticVectorSource.disabled = isB0;
  }
  const vectorSource = elements.semanticVectorSource ? elements.semanticVectorSource.value : "synthetic";
  const showLocalPath = !isB0 && vectorSource === "local";
  if (elements.semanticVectorPathGroup) {
    elements.semanticVectorPathGroup.hidden = !showLocalPath;
  }
  if (elements.syntheticVectorWarning) {
    elements.syntheticVectorWarning.hidden = isB0 || vectorSource !== "synthetic";
  }

  // Max candidates & embedding slots
  const maxCandidates = parseInt(elements.semanticMaxCandidates.value, 10) || 8;
  if (elements.semanticEmbeddingSlots) {
    elements.semanticEmbeddingSlots.max = String(Math.max(1, maxCandidates - 1));
    if (parseInt(elements.semanticEmbeddingSlots.value, 10) >= maxCandidates) {
      elements.semanticEmbeddingSlots.value = String(Math.max(1, maxCandidates - 1));
    }
    elements.semanticEmbeddingSlots.disabled = !isB2;
  }
  if (elements.semanticSlotsGroup) {
    elements.semanticSlotsGroup.style.opacity = isB2 ? "1" : "0.5";
  }

  // Similarity threshold toggle
  if (elements.semanticSimilarityToggle) {
    elements.semanticSimilarityToggle.disabled = !isB2;
    if (!isB2) elements.semanticSimilarityToggle.checked = false;
  }
  const similarityOn = isB2 && elements.semanticSimilarityToggle && elements.semanticSimilarityToggle.checked;
  if (elements.semanticSimilarityControls) {
    elements.semanticSimilarityControls.hidden = !similarityOn;
  }
  if (elements.semanticAdaptiveSimilarity) {
    elements.semanticAdaptiveSimilarity.disabled = !similarityOn;
    if (!similarityOn) {
      elements.semanticAdaptiveSimilarity.checked = false;
    }
  }
}

function updateMode() {
  const mode = selectedMode();
  const isHybrid = mode === "hybrid_sentinel";
  const isSemantic = mode === "semantic";

  // Hide ratio/substitute controls in semantic mode
  if (elements.standardControls) {
    elements.standardControls.hidden = isSemantic;
  }
  elements.substitute.disabled = isHybrid;

  // Toggle semantic settings visibility
  if (elements.semanticSettings) {
    elements.semanticSettings.hidden = !isSemantic;
    elements.semanticSettings.disabled = !isSemantic;
  }

  // Update mode note
  if (isSemantic) {
    elements.modeNote.textContent =
      "Offline lexical suggestions with local vectors and structural guards. Human review is required before using rewritten prompts.";
  } else if (isHybrid) {
    elements.modeNote.textContent =
      "Experimental. First run loads LLMLingua-2; protected clauses are restored, while model edits have no word-level audit.";
  } else {
    elements.modeNote.textContent =
      "Tree pruning shows each deletion and protection decision.";
  }

  if (isSemantic) {
    updateSemanticSettings();
  }
}

async function compressText() {
  elements.error.hidden = true;
  elements.button.disabled = true;
  const mode = selectedMode();
  elements.button.textContent =
    mode === "hybrid_sentinel"
      ? "Loading or compressing…"
      : mode === "semantic"
      ? "Finding suggestions…"
      : "Compressing…";

  try {
    let payload;
    if (mode === "semantic") {
      const arm = elements.semanticArm ? elements.semanticArm.value : "b0";
      const useWordnet = elements.semanticWordnet ? elements.semanticWordnet.checked : false;
      const vectorSource = elements.semanticVectorSource ? elements.semanticVectorSource.value : "synthetic";
      const vectorPath = elements.semanticVectorPath ? elements.semanticVectorPath.value.trim() : "";
      const encoding = elements.semanticEncoding ? elements.semanticEncoding.value : "o200k_base";
      const maxCandidates = parseInt(elements.semanticMaxCandidates.value, 10) || 8;
      const embeddingSlots = parseInt(elements.semanticEmbeddingSlots.value, 10) || 4;
      const similarityOn = elements.semanticSimilarityToggle && elements.semanticSimilarityToggle.checked;
      const minSimilarity = similarityOn ? parseFloat(elements.semanticMinSimilarity.value) : null;
      const adaptiveSimilarity = similarityOn && elements.semanticAdaptiveSimilarity.checked;
      const price = elements.semanticPrice ? parseFloat(elements.semanticPrice.value) || 0.0 : 0.0;

      payload = {
        text: elements.source.value,
        mode: "semantic",
        arm: arm,
        use_wordnet: useWordnet,
        vector_source: vectorSource,
        encoding: encoding,
        max_candidates: maxCandidates,
        embedding_slots: embeddingSlots,
        min_embedding_similarity: minSimilarity,
        adaptive_similarity: adaptiveSimilarity,
        input_price_per_million: Math.max(0, price),
      };

      if (arm !== "b0" && vectorSource === "local") {
        if (!vectorPath) {
          throw new Error("Enter a valid local vector file path or choose bundled synthetic vectors.");
        }
        payload.vector_path = vectorPath;
      }
    } else {
      payload = {
        text: elements.source.value,
        target_ratio: Number(elements.ratio.value) / 100,
        substitute: elements.substitute.checked,
        mode: mode,
      };
    }

    const response = await fetch("/api/compress", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
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
  elements.exampleList.innerHTML = examples.map(example => {
    const isSemantic = example.mode === "semantic";
    const isHybrid = example.mode === "hybrid_sentinel";
    const tagClass = isSemantic ? "tag-semantic" : isHybrid ? "tag-hybrid" : "tag-tree";
    const tagLabel = isSemantic
      ? `Semantic · ${(example.arm || "b0").toUpperCase()}`
      : isHybrid
      ? "Hybrid"
      : "Tree pruning";

    return `
      <button class="example-button" type="button" data-example-id="${escapeHtml(example.id)}">
        <span class="example-tag ${tagClass}">${escapeHtml(tagLabel)}</span>
        <strong>${escapeHtml(example.name)}</strong>
        <span>${escapeHtml(example.preview)}…</span>
      </button>`;
  }).join("");

  elements.exampleList.querySelectorAll("button").forEach(button => {
    button.addEventListener("click", () => {
      const example = state.examples.find(item => item.id === button.dataset.exampleId);
      if (!example) return;
      elements.source.value = example.text;
      const targetMode = example.mode || "conservative";
      const modeRadio = elements.modes.find(mode => mode.value === targetMode);
      if (modeRadio) modeRadio.checked = true;

      if (targetMode === "semantic") {
        if (example.arm && elements.semanticArm) {
          elements.semanticArm.value = example.arm;
        }
        if (example.vector_source && elements.semanticVectorSource) {
          elements.semanticVectorSource.value = example.vector_source;
        }
      }

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

if (elements.semanticArm) {
  elements.semanticArm.addEventListener("change", updateSemanticSettings);
}
if (elements.semanticVectorSource) {
  elements.semanticVectorSource.addEventListener("change", updateSemanticSettings);
}
if (elements.semanticMaxCandidates) {
  elements.semanticMaxCandidates.addEventListener("input", updateSemanticSettings);
}
if (elements.semanticSimilarityToggle) {
  elements.semanticSimilarityToggle.addEventListener("change", updateSemanticSettings);
}
if (elements.semanticMinSimilarity) {
  elements.semanticMinSimilarity.addEventListener("input", () => {
    if (elements.semanticMinSimilarityOutput) {
      elements.semanticMinSimilarityOutput.textContent = elements.semanticMinSimilarity.value;
    }
  });
}

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
