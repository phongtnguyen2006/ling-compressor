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
  semanticVectorLimitGroup: document.querySelector("#semantic-vector-limit-group"),
  semanticVectorMaxWords: document.querySelector("#semantic-vector-max-words"),
  semanticFullSizeWarning: document.querySelector("#semantic-full-size-warning"),
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
  vectorRowsBadge: document.querySelector("#vector-rows-badge"),
  configHashBadge: document.querySelector("#config-hash-badge"),
  costSavedBadge: document.querySelector("#cost-saved-badge"),
  semanticWarningsList: document.querySelector("#semantic-warnings-list"),
  deletionLabel: document.querySelector("#deletion-label"),
  protectedLabel: document.querySelector("#protected-label"),
  substitutionLabel: document.querySelector("#substitution-label"),
  auditSummaryLabel: document.querySelector("#audit-summary-label"),

  // Semantic reviewed template approval workflow elements
  semanticApprovalWorkflow: document.querySelector("#semantic-approval-workflow"),
  occurrencesList: document.querySelector("#occurrences-list"),
  approveAllButton: document.querySelector("#approve-all-button"),
  rejectAllButton: document.querySelector("#reject-all-button"),
  resetAllButton: document.querySelector("#reset-all-button"),
  approvalSummaryBadge: document.querySelector("#approval-summary-badge"),
  approvalRecountNotice: document.querySelector("#approval-recount-notice"),
  persistTemplateButton: document.querySelector("#persist-template-button"),
  copyApprovedButton: document.querySelector("#copy-approved-button"),
  exportTextButton: document.querySelector("#export-text-button"),
  exportAuditButton: document.querySelector("#export-audit-button"),
  approvalBannerMessage: document.querySelector("#approval-banner-message"),
};

const semanticReviewState = {
  active: false,
  originalText: "",
  configHash: "",
  encoding: "o200k_base",
  vectorIdentity: "none",
  occurrences: [],
  composedText: "",
  tokensBefore: 0,
  tokensAfter: 0,
  savedTokens: 0,
  savedPercent: 0,
  reviewStatus: "pending",
  savedTemplate: null,
  savedAudit: null,
  lastDecisions: [],
  activeRequestConfig: null,
  invalidated: false,
  reviewVersion: 0,
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
  const positionedEdits = (edits || []).map(edit => {
    const spanStart = edit.span.start != null ? edit.span.start : edit.span[0];
    const spanEnd = edit.span.end != null ? edit.span.end : edit.span[1];
    return {
      ...edit,
      start: offsets[spanStart],
      end: offsets[spanEnd],
    };
  });
  // Original annotations: edits with their review status, and rejected decisions with protection as protected
  const origBoundaries = new Set([0, original.length]);
  positionedEdits.forEach(e => {
    origBoundaries.add(e.start);
    origBoundaries.add(e.end);
  });
  const protectedDecisions = (decisions || []).filter(
    d => d.decision === "rejected" && d.reason && d.reason.startsWith("protected")
  ).map(decision => {
    const spanStart = decision.span.start != null ? decision.span.start : decision.span[0];
    const spanEnd = decision.span.end != null ? decision.span.end : decision.span[1];
    return {
      ...decision,
      start: offsets[spanStart],
      end: offsets[spanEnd],
    };
  });
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
      const status = edit.decision || "pending";
      const simText = edit.similarity != null ? `, cos: ${edit.similarity.toFixed(3)}` : "";
      const deltaText = edit.token_delta != null ? `, -${edit.token_delta} tok` : "";
      origHtml += `<mark class="annotation-substitution annotation-substitution-${escapeHtml(status)}" title="[${escapeHtml(status.toUpperCase())}] Replaced with '${escapeHtml(edit.replacement)}' (${escapeHtml(edit.provenance || '')}${simText}${deltaText})">${chunk}</mark>`;
    } else if (prot) {
      origHtml += `<mark class="annotation-protected" title="Protected: ${escapeHtml(prot.reason)}">${chunk}</mark>`;
    } else {
      origHtml += chunk;
    }
  }

  // Compressed annotations: highlight ONLY accepted substitutions in the composed output
  const acceptedEdits = positionedEdits.filter(e => e.decision === "accepted").sort((a, b) => a.start - b.start);
  let compHtml = "";
  let origCursor = 0;
  let compCursor = 0;
  for (const edit of acceptedEdits) {
    const prefixLen = edit.start - origCursor;
    const prefix = compressed.slice(compCursor, compCursor + prefixLen);
    compHtml += escapeHtml(prefix);
    compCursor += prefixLen;
    origCursor = edit.end;

    const replLen = edit.replacement.length;
    const replText = compressed.slice(compCursor, compCursor + replLen);
    const sim = edit.similarity != null ? `, cos: ${edit.similarity.toFixed(3)}` : "";
    const delta = edit.token_delta != null ? `, -${edit.token_delta} tok` : "";
    compHtml += `<mark class="annotation-substitution annotation-substitution-accepted" title="Accepted: Substituted for '${escapeHtml(edit.original)}' (${escapeHtml(edit.provenance || '')}${sim}${delta})">${escapeHtml(replText)}</mark>`;
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

function renderOccurrencesList() {
  if (!elements.occurrencesList) return;
  if (!semanticReviewState.occurrences.length) {
    elements.occurrencesList.innerHTML = '<p class="empty-audit">No substitutions were proposed for this text.</p>';
    if (elements.approvalSummaryBadge) {
      elements.approvalSummaryBadge.textContent = "0 proposed";
      elements.approvalSummaryBadge.className = "meta-badge badge-pending";
    }
    if (elements.persistTemplateButton) elements.persistTemplateButton.disabled = true;
    return;
  }

  elements.occurrencesList.innerHTML = semanticReviewState.occurrences.map((occ, idx) => {
    const status = occ.decision || "pending";
    const sim = occ.similarity != null ? `cos: ${occ.similarity.toFixed(3)}` : "no vector score";
    const delta = occ.token_delta != null ? `-${occ.token_delta} tok` : "";
    const spanStart = occ.span.start != null ? occ.span.start : occ.span[0];
    const spanEnd = occ.span.end != null ? occ.span.end : occ.span[1];

    return `
      <div class="occurrence-card occurrence-${escapeHtml(status)}" data-index="${idx}">
        <div class="occurrence-main">
          <div class="occurrence-terms">
            <code class="term-original">${escapeHtml(occ.original)}</code>
            <span class="term-arrow">→</span>
            <code class="term-replacement">${escapeHtml(occ.replacement)}</code>
            <span class="occurrence-badge badge-${escapeHtml(status)}">${escapeHtml(status)}</span>
          </div>
          <div class="occurrence-sentence">
            <span class="sentence-before">${escapeHtml(occ.sentence_before || '')}</span>
            <span class="term-arrow">→</span>
            <span class="sentence-after">${escapeHtml(occ.sentence_after || '')}</span>
          </div>
          <div class="occurrence-meta">
            <small>${escapeHtml(occ.provenance || "lexical")} · ${escapeHtml(sim)} · ${escapeHtml(delta)} · span [${spanStart}:${spanEnd}]</small>
          </div>
        </div>
        <div class="occurrence-buttons">
          <button type="button" class="btn-occ ${status === 'accepted' ? 'active-accepted' : ''}" data-action="accept" data-index="${idx}">Accept</button>
          <button type="button" class="btn-occ ${status === 'rejected' ? 'active-rejected' : ''}" data-action="reject" data-index="${idx}">Reject</button>
          <button type="button" class="btn-occ ${status === 'pending' ? 'active-pending' : ''}" data-action="reset" data-index="${idx}">Reset</button>
        </div>
      </div>`;
  }).join("");

  elements.occurrencesList.querySelectorAll(".btn-occ").forEach(btn => {
    btn.addEventListener("click", () => {
      const idx = parseInt(btn.dataset.index, 10);
      const action = btn.dataset.action;
      if (action === "accept") {
        semanticReviewState.occurrences[idx].decision = "accepted";
      } else if (action === "reject") {
        semanticReviewState.occurrences[idx].decision = "rejected";
      } else if (action === "reset") {
        semanticReviewState.occurrences[idx].decision = "pending";
      }
      updateApprovalPreview();
    });
  });
}

let previewRequestId = 0;

function invalidateStaleReviewState(reason) {
  semanticReviewState.reviewVersion += 1;
  semanticReviewState.invalidated = true;
  semanticReviewState.savedTemplate = null;
  semanticReviewState.savedAudit = null;
  if (elements.persistTemplateButton) elements.persistTemplateButton.disabled = true;
  if (elements.copyApprovedButton) elements.copyApprovedButton.disabled = true;
  if (elements.exportTextButton) elements.exportTextButton.disabled = true;
  if (elements.exportAuditButton) elements.exportAuditButton.disabled = true;
  if (elements.approvalBannerMessage) {
    elements.approvalBannerMessage.textContent = reason || "Source text was edited after compression. Run compression again to review suggestions.";
    elements.approvalBannerMessage.className = "approval-banner-message warning";
    elements.approvalBannerMessage.hidden = false;
  }
}

async function updateApprovalPreview() {
  if (!semanticReviewState.active || semanticReviewState.invalidated) return;
  semanticReviewState.reviewVersion += 1;

  // Clear any existing saved template and audit on decision change
  semanticReviewState.savedTemplate = null;
  semanticReviewState.savedAudit = null;
  if (elements.copyApprovedButton) elements.copyApprovedButton.disabled = true;
  if (elements.exportTextButton) elements.exportTextButton.disabled = true;
  if (elements.exportAuditButton) elements.exportAuditButton.disabled = true;

  if (elements.source.value !== semanticReviewState.originalText) {
    invalidateStaleReviewState("Source text was edited after compression. Run compression again to review suggestions.");
    return;
  }

  const currentReqId = ++previewRequestId;
  try {
    const payload = {
      text: semanticReviewState.originalText,
      decisions: semanticReviewState.occurrences,
      config_hash: semanticReviewState.configHash,
      encoding: semanticReviewState.encoding,
      vector_identity: semanticReviewState.vectorIdentity,
    };
    const response = await fetch("/api/semantic/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();

    // Prevent race conditions: discard superseded response
    if (currentReqId !== previewRequestId) return;

    if (elements.source.value !== semanticReviewState.originalText) {
      invalidateStaleReviewState("Source text was edited after compression. Run compression again to review suggestions.");
      return;
    }

    if (!response.ok) {
      throw new Error(data.error || "Preview calculation failed.");
    }

    semanticReviewState.composedText = data.composed_text;
    semanticReviewState.tokensBefore = data.tokens_before;
    semanticReviewState.tokensAfter = data.tokens_after;
    semanticReviewState.savedTokens = data.saved_tokens;
    semanticReviewState.savedPercent = data.saved_percent;
    semanticReviewState.reviewStatus = data.review_status;
    const acceptedCount = data.accepted_count ?? semanticReviewState.occurrences.filter(o => o.decision === "accepted").length;
    const rejectedCount = data.rejected_count ?? semanticReviewState.occurrences.filter(o => o.decision === "rejected").length;
    const pendingCount = data.pending_count ?? semanticReviewState.occurrences.filter(o => o.decision === "pending").length;

    if (!data.is_valid) {
      if (elements.approvalBannerMessage) {
        elements.approvalBannerMessage.textContent = `Invalidated: ${data.invalidation_reason}`;
        elements.approvalBannerMessage.className = "approval-banner-message error";
        elements.approvalBannerMessage.hidden = false;
      }
      if (elements.reviewStatusBadge) {
        elements.reviewStatusBadge.textContent = "Review: INVALIDATED";
        elements.reviewStatusBadge.className = "meta-badge badge-invalidated";
      }
    } else {
      if (elements.approvalBannerMessage) {
        elements.approvalBannerMessage.hidden = true;
      }
      if (elements.reviewStatusBadge) {
        elements.reviewStatusBadge.textContent = `Review: ${data.review_status.toUpperCase()}`;
        elements.reviewStatusBadge.className = `meta-badge badge-${data.review_status}`;
      }
    }

    document.querySelector("#tokens-after").textContent = data.tokens_after;
    document.querySelector("#tokens-saved").textContent = data.saved_tokens;
    const ratio = data.tokens_before > 0 ? data.tokens_after / data.tokens_before : 1;
    document.querySelector("#ruler-fill").style.width = `${Math.min(100, Math.max(0, ratio * 100))}%`;
    document.querySelector("#compressed-count").textContent = `${data.tokens_after} tokens · ${data.saved_percent}% saved`;

    if (!data.is_valid) {
      document.querySelector("#compressed-title").textContent = "Review invalidated · original text retained";
    } else if (pendingCount > 0) {
      document.querySelector("#compressed-title").textContent =
        `Review pending (${acceptedCount} of ${semanticReviewState.occurrences.length} approved · ${pendingCount} pending)`;
    } else if (acceptedCount > 0 && data.saved_tokens > 0) {
      document.querySelector("#compressed-title").textContent =
        `Candidate approved (${acceptedCount} of ${semanticReviewState.occurrences.length} approved · unpersisted draft)`;
    } else {
      document.querySelector("#compressed-title").textContent =
        `Review rejected (0 of ${semanticReviewState.occurrences.length} approved)`;
    }

    if (elements.approvalSummaryBadge) {
      elements.approvalSummaryBadge.textContent =
        `${acceptedCount} of ${semanticReviewState.occurrences.length} approved (${pendingCount} pending, ${rejectedCount} rejected)`;
      elements.approvalSummaryBadge.className = `meta-badge badge-${data.review_status}`;
    }
    if (elements.approvalRecountNotice) {
      elements.approvalRecountNotice.textContent =
        `Recounted under ${semanticReviewState.encoding}: ${data.tokens_before} → ${data.tokens_after} tokens (-${data.saved_tokens} tok, ${data.saved_percent}%)`;
    }

    if (elements.costSavedSummary) {
      const price = elements.semanticPrice ? parseFloat(elements.semanticPrice.value) || 0.0 : 0.0;
      const costSaved = data.saved_tokens * price / 1_000_000;
      elements.costSavedSummary.textContent = ` · $${costSaved.toFixed(6)} saved`;
    }

    const { origHtml, compHtml } = annotateSemantic(
      semanticReviewState.originalText,
      semanticReviewState.composedText,
      semanticReviewState.occurrences,
      semanticReviewState.lastDecisions || []
    );
    document.querySelector("#annotated-text").innerHTML = origHtml;
    document.querySelector("#compressed-text").innerHTML = compHtml;

    renderOccurrencesList();

    // Save button requirement: no pending, >=1 accepted, positive savings, is_valid, source unchanged
    const canPersist = (
      data.is_valid &&
      pendingCount === 0 &&
      acceptedCount >= 1 &&
      data.saved_tokens > 0 &&
      elements.source.value === semanticReviewState.originalText
    );
    if (elements.persistTemplateButton) {
      elements.persistTemplateButton.disabled = !canPersist;
    }
    // Copy and export stay disabled until explicitly persisted
    if (elements.copyApprovedButton) elements.copyApprovedButton.disabled = true;
    if (elements.exportTextButton) elements.exportTextButton.disabled = true;
    if (elements.exportAuditButton) elements.exportAuditButton.disabled = true;
  } catch (err) {
    if (currentReqId === previewRequestId && elements.approvalBannerMessage) {
      elements.approvalBannerMessage.textContent = err.message;
      elements.approvalBannerMessage.className = "approval-banner-message error";
      elements.approvalBannerMessage.hidden = false;
    }
  }
}

async function persistApprovedTemplate() {
  if (!semanticReviewState.active || semanticReviewState.invalidated) return;
  const savingVersion = semanticReviewState.reviewVersion;
  if (elements.source.value !== semanticReviewState.originalText) {
    invalidateStaleReviewState("Source text was edited after compression. Run compression again before saving.");
    return;
  }
  elements.persistTemplateButton.disabled = true;
  elements.persistTemplateButton.textContent = "Saving template…";
  try {
    const payload = {
      text: semanticReviewState.originalText,
      decisions: semanticReviewState.occurrences,
      reviewer: "local-user",
      ...(semanticReviewState.activeRequestConfig || {}),
      config_hash: semanticReviewState.configHash,
      vector_identity: semanticReviewState.vectorIdentity,
      metadata: {
        price_per_million: elements.semanticPrice ? parseFloat(elements.semanticPrice.value) || 0.0 : 0.0,
      },
    };
    const response = await fetch("/api/semantic/approve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Failed to persist approved template.");
    }
    if (semanticReviewState.invalidated || savingVersion !== semanticReviewState.reviewVersion || elements.source.value !== semanticReviewState.originalText) {
      invalidateStaleReviewState("Source text or semantic settings changed while saving. Re-run compression before exporting.");
      return;
    }
    semanticReviewState.savedTemplate = data.template;
    semanticReviewState.savedAudit = data.audit;

    if (elements.approvalBannerMessage) {
      elements.approvalBannerMessage.innerHTML = `
        <strong>Template persisted locally:</strong> <code>${escapeHtml(data.template.template_id)}</code><br>
        Bound to content hash <code>${escapeHtml(data.template.binding.original_hash.slice(0, 16))}…</code> ·
        config hash <code>${escapeHtml(data.template.binding.config_hash.slice(0, 16))}…</code> ·
        tokenizer <code>${escapeHtml(data.template.binding.tokenizer_name)}</code> ·
        vectors <code>${escapeHtml(data.template.binding.vector_identity)}</code>.
      `;
      elements.approvalBannerMessage.className = "approval-banner-message success";
      elements.approvalBannerMessage.hidden = false;
    }
    document.querySelector("#compressed-title").textContent =
      `Persisted approved template (${data.template.template_id})`;
    if (elements.copyApprovedButton) elements.copyApprovedButton.disabled = false;
    if (elements.exportTextButton) elements.exportTextButton.disabled = false;
    if (elements.exportAuditButton) elements.exportAuditButton.disabled = false;
  } catch (err) {
    if (elements.approvalBannerMessage) {
      elements.approvalBannerMessage.textContent = err.message;
      elements.approvalBannerMessage.className = "approval-banner-message error";
      elements.approvalBannerMessage.hidden = false;
    }
  } finally {
    elements.persistTemplateButton.textContent = "Save Approved Template";
    const hasPending = semanticReviewState.occurrences.some(o => o.decision === "pending");
    const hasAccepted = semanticReviewState.occurrences.some(o => o.decision === "accepted");
    elements.persistTemplateButton.disabled = semanticReviewState.invalidated || hasPending || !hasAccepted || elements.source.value !== semanticReviewState.originalText;
  }
}

function downloadFile(filename, content, mimeType = "text/plain") {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function exportApprovedText() {
  if (!semanticReviewState.savedTemplate) {
    alert("Please save the approved template before exporting.");
    return;
  }
  const tplId = semanticReviewState.savedTemplate.template_id;
  downloadFile(`${tplId}-approved.txt`, semanticReviewState.savedTemplate.composed_text, "text/plain;charset=utf-8");
}

function exportAuditJson() {
  if (!semanticReviewState.savedAudit) {
    alert("Please save the approved template before exporting audit records.");
    return;
  }
  const tplId = semanticReviewState.savedTemplate ? semanticReviewState.savedTemplate.template_id : "template";
  downloadFile(`${tplId}-audit.json`, JSON.stringify(semanticReviewState.savedAudit, null, 2), "application/json;charset=utf-8");
}

async function copyApprovedText() {
  if (!semanticReviewState.savedTemplate) return;
  try {
    await navigator.clipboard.writeText(semanticReviewState.savedTemplate.composed_text);
    const originalText = elements.copyApprovedButton.textContent;
    elements.copyApprovedButton.textContent = "Copied!";
    setTimeout(() => { elements.copyApprovedButton.textContent = originalText; }, 2000);
  } catch (_) {
    const textarea = document.createElement("textarea");
    textarea.value = semanticReviewState.savedTemplate.composed_text;
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand("copy");
    document.body.removeChild(textarea);
    const originalText = elements.copyApprovedButton.textContent;
    elements.copyApprovedButton.textContent = "Copied!";
    setTimeout(() => { elements.copyApprovedButton.textContent = originalText; }, 2000);
  }
}

function renderResult(data, requestPayload = null) {
  // Ignore any preview response started for the preceding compression result.
  previewRequestId += 1;
  const mode = data.mode;
  const isHybrid = mode === "hybrid_sentinel";
  const isSemantic = mode === "semantic";

  if (isSemantic) {
    const armLabel = data.provider_metadata && data.provider_metadata.arm
      ? data.provider_metadata.arm.toUpperCase()
      : "B0";
    document.querySelector("#result-mode").textContent =
      `Semantic substitution · ${armLabel} · ${data.token_counter || 'tiktoken'} · review pending`;

    semanticReviewState.active = true;
    semanticReviewState.originalText = data.original;
    semanticReviewState.configHash = data.config_hash || "";
    semanticReviewState.encoding = (data.token_counter || (elements.semanticEncoding ? elements.semanticEncoding.value : "o200k_base")).replace(/^tiktoken:/, "");
    semanticReviewState.vectorIdentity = data.vector_label || "none";
    semanticReviewState.lastDecisions = data.decisions || [];
    semanticReviewState.occurrences = (data.occurrences || (data.edits || []).map(edit => ({
      span: edit.span,
      original: edit.original,
      replacement: edit.replacement,
      decision: "pending",
      provenance: edit.provenance,
      similarity: edit.similarity,
      token_delta: edit.token_delta,
      pos: edit.pos,
      sentence_before: edit.sentence_before,
      sentence_after: edit.sentence_after,
    })));
    semanticReviewState.composedText = data.original;
    semanticReviewState.tokensBefore = data.tokens_before;
    semanticReviewState.tokensAfter = data.tokens_before;
    semanticReviewState.savedTokens = 0;
    semanticReviewState.savedPercent = 0.0;
    semanticReviewState.activeRequestConfig = requestPayload ? {
      arm: requestPayload.arm,
      use_wordnet: requestPayload.use_wordnet,
      vector_source: requestPayload.vector_source,
      ...(requestPayload.vector_path ? { vector_path: requestPayload.vector_path } : {}),
      vector_max_words: requestPayload.vector_max_words,
      min_embedding_similarity: requestPayload.min_embedding_similarity,
      adaptive_similarity: requestPayload.adaptive_similarity,
      max_candidates: requestPayload.max_candidates,
      embedding_slots: requestPayload.embedding_slots,
      encoding: requestPayload.encoding,
    } : null;
    semanticReviewState.invalidated = false;
    semanticReviewState.reviewVersion += 1;
    semanticReviewState.reviewStatus = "pending";
    semanticReviewState.savedTemplate = null;
    semanticReviewState.savedAudit = null;

    if (elements.semanticApprovalWorkflow) {
      elements.semanticApprovalWorkflow.hidden = false;
    }
    if (elements.approvalBannerMessage) {
      elements.approvalBannerMessage.hidden = true;
      elements.approvalBannerMessage.textContent = "";
    }
    if (elements.persistTemplateButton) elements.persistTemplateButton.disabled = true;
    if (elements.copyApprovedButton) elements.copyApprovedButton.disabled = true;
    if (elements.exportTextButton) elements.exportTextButton.disabled = true;
    if (elements.exportAuditButton) elements.exportAuditButton.disabled = true;

    renderOccurrencesList();

    if (elements.approvalSummaryBadge) {
      elements.approvalSummaryBadge.textContent =
        `0 of ${semanticReviewState.occurrences.length} approved (${semanticReviewState.occurrences.length} pending)`;
      elements.approvalSummaryBadge.className = "meta-badge badge-pending";
    }
    if (elements.approvalRecountNotice) {
      elements.approvalRecountNotice.textContent =
        `Recounted (${semanticReviewState.encoding}): ${data.tokens_before} tokens (0 approved · review pending)`;
    }

    document.querySelector("#compressed-title").textContent =
      `Approved output (0 of ${semanticReviewState.occurrences.length} approved · review pending)`;
    document.querySelector("#original-title").textContent = "Original, annotated";
    document.querySelector("#tokens-after").textContent = data.tokens_before;
    document.querySelector("#tokens-saved").textContent = "0";
    document.querySelector("#ruler-fill").style.width = "100%";
    document.querySelector("#original-count").textContent = `${data.tokens_before} tokens`;
    document.querySelector("#compressed-count").textContent = `${data.tokens_before} tokens · 0.0% saved`;

    if (elements.costSavedSummary) {
      elements.costSavedSummary.textContent = " · gross estimate only (review pending, default unchanged)";
    }
  } else {
    semanticReviewState.active = false;
    if (elements.semanticApprovalWorkflow) {
      elements.semanticApprovalWorkflow.hidden = true;
    }
    if (isHybrid) {
      document.querySelector("#result-mode").textContent =
        "Guarded LLMLingua-2 · protected clauses are restored and checked";
      document.querySelector("#compressed-title").textContent = "Hybrid compressed";
      document.querySelector("#original-title").textContent = "Original";
    } else {
      document.querySelector("#result-mode").textContent =
        "Tree pruning · each cut is recorded below";
      document.querySelector("#compressed-title").textContent = "Compressed";
      document.querySelector("#original-title").textContent = "Original, annotated";
    }

    document.querySelector("#tokens-after").textContent = data.tokens_after;
    document.querySelector("#tokens-saved").textContent = data.saved_tokens;
    document.querySelector("#ruler-fill").style.width = `${Math.min(100, Math.max(0, data.ratio * 100))}%`;
    document.querySelector("#original-count").textContent = `${data.tokens_before} tokens`;
    document.querySelector("#compressed-count").textContent = `${data.tokens_after} tokens · ${data.saved_percent}% saved`;
    if (elements.costSavedSummary) {
      elements.costSavedSummary.textContent = "";
    }
  }

  const totalMs = data.timings_ms ? (data.timings_ms.total != null ? data.timings_ms.total : 0) : 0;
  document.querySelector("#time-taken").textContent = Math.round(totalMs);

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
      const vm = (data.provider_metadata && data.provider_metadata.vectors) || null;
      const isSynthetic = (data.vector_label && data.vector_label.toLowerCase().includes("synthetic")) || (data.provider_metadata && data.provider_metadata.vector_source === "synthetic");
      const armUpper = (data.provider_metadata && data.provider_metadata.arm ? data.provider_metadata.arm : "b0").toUpperCase();

      if (elements.vectorLabelBadge) {
        if (armUpper === "B0") {
          elements.vectorLabelBadge.textContent = "Vectors: none";
          elements.vectorLabelBadge.title = "B0 uses curated rules only. Vectors not loaded.";
        } else if (isSynthetic) {
          elements.vectorLabelBadge.textContent = "Vectors: Bundled synthetic fixture (software fixture, not trained embeddings)";
          elements.vectorLabelBadge.title = data.vector_label || "Bundled synthetic software fixture";
        } else {
          elements.vectorLabelBadge.textContent = `Vectors: ${data.vector_label || "custom local file"}`;
          elements.vectorLabelBadge.title = data.vector_label || "";
        }
      }

      if (elements.vectorRowsBadge) {
        if (vm && armUpper !== "B0") {
          const retained = vm.retained_rows.toLocaleString();
          const source = vm.source_rows.toLocaleString();
          const cap = vm.max_words ? (vm.max_words >= 1000 ? `${vm.max_words / 1000}k` : vm.max_words) : "none";
          const fixtureTag = isSynthetic ? "Synthetic fixture · " : "";
          elements.vectorRowsBadge.textContent = `${fixtureTag}${retained} of ${source} rows retained (cap: ${cap})`;
          elements.vectorRowsBadge.hidden = false;
          if (vm.retained_rows >= 400000) {
            elements.vectorRowsBadge.classList.add("badge-warning");
            elements.vectorRowsBadge.title = "400k full retention selected: cold load may incur higher initial latency and memory usage.";
          } else {
            elements.vectorRowsBadge.classList.remove("badge-warning");
            elements.vectorRowsBadge.title = "";
          }
        } else {
          elements.vectorRowsBadge.hidden = true;
        }
      }
      if (elements.configHashBadge) {
        elements.configHashBadge.textContent = `Config: ${(data.config_hash || "unknown").slice(0, 12)}`;
        elements.configHashBadge.title = data.config_hash || "Configuration hash unavailable";
      }
      if (elements.costSavedBadge) {
        const price = data.input_price_per_million != null ? data.input_price_per_million : 0;
        elements.costSavedBadge.textContent = `Gross input est.: $${(data.cost_saved_usd || 0).toFixed(6)} (@ $${price.toFixed(2)}/1M · unreviewed proposal)`;
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
    const { origHtml, compHtml } = annotateSemantic(
      data.original,
      semanticReviewState.composedText,
      semanticReviewState.occurrences,
      data.decisions || []
    );
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
  if (elements.semanticVectorLimitGroup) {
    elements.semanticVectorLimitGroup.hidden = !showLocalPath;
  }
  const retentionElem = elements.semanticVectorMaxWords;
  if (retentionElem) {
    retentionElem.disabled = isB0;
  }
  if (elements.semanticFullSizeWarning) {
    elements.semanticFullSizeWarning.hidden = !showLocalPath || (retentionElem && retentionElem.value !== "400000");
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

  if (semanticReviewState.active) {
    invalidateStaleReviewState("Semantic configuration changed. Re-run compression to generate suggestions under the new settings.");
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
      const retentionElem = elements.semanticVectorMaxWords;
      const vectorMaxWords = retentionElem ? Number(retentionElem.value) : 50000;
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
        vector_max_words: vectorMaxWords,
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
    renderResult(data, payload);
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
        const retentionElem = elements.semanticVectorMaxWords;
        if (retentionElem) {
          retentionElem.value = String(example.vector_max_words || 50000);
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
    state.examples = data.examples.filter(example => example.mode === "semantic");
    renderExamples(state.examples);
  } catch (_) {
    elements.exampleList.innerHTML = '<p class="loading-line">Examples could not be loaded.</p>';
  }
}

function updateCharacterCount() {
  const length = elements.source.value.length;
  elements.characterCount.textContent = `${length.toLocaleString()} character${length === 1 ? "" : "s"}`;
  if (semanticReviewState.active && elements.source.value !== semanticReviewState.originalText) {
    invalidateStaleReviewState("Input text was modified: template bindings are invalidated. Unchanged text preserved. Re-run compression to review new suggestions.");
    if (elements.reviewStatusBadge) {
      elements.reviewStatusBadge.textContent = "Review: INVALIDATED (STALE CONTENT)";
      elements.reviewStatusBadge.className = "meta-badge badge-invalidated";
    }
  }
}

elements.form.addEventListener("submit", event => { event.preventDefault(); compressText(); });
elements.source.addEventListener("input", updateCharacterCount);
elements.ratio.addEventListener("input", () => { elements.ratioOutput.textContent = `${elements.ratio.value}%`; });
elements.modes.forEach(mode => mode.addEventListener("change", updateMode));

if (elements.approveAllButton) {
  elements.approveAllButton.addEventListener("click", () => {
    semanticReviewState.occurrences.forEach(occ => { occ.decision = "accepted"; });
    updateApprovalPreview();
  });
}
if (elements.rejectAllButton) {
  elements.rejectAllButton.addEventListener("click", () => {
    semanticReviewState.occurrences.forEach(occ => { occ.decision = "rejected"; });
    updateApprovalPreview();
  });
}
if (elements.resetAllButton) {
  elements.resetAllButton.addEventListener("click", () => {
    semanticReviewState.occurrences.forEach(occ => { occ.decision = "pending"; });
    updateApprovalPreview();
  });
}
if (elements.persistTemplateButton) {
  elements.persistTemplateButton.addEventListener("click", persistApprovedTemplate);
}
if (elements.copyApprovedButton) {
  elements.copyApprovedButton.addEventListener("click", copyApprovedText);
}
if (elements.exportTextButton) {
  elements.exportTextButton.addEventListener("click", exportApprovedText);
}
if (elements.exportAuditButton) {
  elements.exportAuditButton.addEventListener("click", exportAuditJson);
}

if (elements.semanticArm) {
  elements.semanticArm.addEventListener("change", updateSemanticSettings);
}
if (elements.semanticVectorSource) {
  elements.semanticVectorSource.addEventListener("change", updateSemanticSettings);
}
const retentionElem = elements.semanticVectorMaxWords;
if (retentionElem) {
  retentionElem.addEventListener("change", updateSemanticSettings);
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
    if (semanticReviewState.active) {
      invalidateStaleReviewState("Semantic settings changed. Re-run compression before saving.");
    }
  });
}
[
  elements.semanticEncoding,
  elements.semanticVectorPath,
  elements.semanticVectorMaxWords,
  elements.semanticEmbeddingSlots,
  elements.semanticWordnet,
  elements.semanticAdaptiveSimilarity,
].filter(Boolean).forEach(control => {
  control.addEventListener("input", () => {
    if (semanticReviewState.active) {
      invalidateStaleReviewState("Semantic settings changed. Re-run compression before saving.");
    }
  });
  control.addEventListener("change", () => {
    if (semanticReviewState.active) {
      invalidateStaleReviewState("Semantic settings changed. Re-run compression before saving.");
    }
  });
});

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
