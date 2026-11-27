const state = { examples: [], resultMode: null };

const TREE_SEMANTIC_SAMPLE = {
  id: "tree_semantic_combined",
  name: "Combined tree + semantic · B2",
  text:
    "The committee, after a long and unusually detailed discussion, approved the plan. " +
    "These projects necessitate careful planning. " +
    "No expenditure may exceed $5,000 without written approval.",
  preview:
    "Two-stage opt-in demo: prune optional modifiers while protecting the $5,000 cap, then propose unreviewed lexical edits on the pruned text",
  mode: "tree_semantic",
  target_ratio: 0.85,
  arm: "b2",
  vector_source: "synthetic",
  vector_max_words: 50000,
};

const elements = {
  form: document.querySelector("#compress-form"),
  source: document.querySelector("#source-text"),
  standardControls: document.querySelector("#standard-controls"),
  ratio: document.querySelector("#ratio"),
  ratioOutput: document.querySelector("#ratio-output"),
  substitute: document.querySelector("#substitute"),
  substituteControl: document.querySelector("#substitute-control"),
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
  semanticSettingsLegend: document.querySelector("#semantic-settings-legend"),
  semanticWarningCopy: document.querySelector("#semantic-warning-copy"),
  semanticArm: document.querySelector("#semantic-arm"),
  semanticArmCaption: document.querySelector("#semantic-arm-caption"),
  semanticEncoding: document.querySelector("#semantic-encoding"),
  semanticEncodingCaption: document.querySelector("#semantic-encoding-caption"),
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

  // Semantic & combined result elements
  costSavedSummary: document.querySelector("#cost-saved-summary"),
  unreviewedCombinedBanner: document.querySelector("#unreviewed-combined-banner"),
  stageBreakdown: document.querySelector("#stage-breakdown"),
  stageBeforeTokens: document.querySelector("#stage-before-tokens"),
  stageBeforeDetail: document.querySelector("#stage-before-detail"),
  stageTreeTokens: document.querySelector("#stage-tree-tokens"),
  stageTreeDetail: document.querySelector("#stage-tree-detail"),
  stageFinalTokens: document.querySelector("#stage-final-tokens"),
  stageFinalDetail: document.querySelector("#stage-final-detail"),
  semanticBanner: document.querySelector("#semantic-banner"),
  reviewStatusBadge: document.querySelector("#review-status-badge"),
  tokenCounterBadge: document.querySelector("#token-counter-badge"),
  vectorLabelBadge: document.querySelector("#vector-label-badge"),
  vectorRowsBadge: document.querySelector("#vector-rows-badge"),
  configHashBadge: document.querySelector("#config-hash-badge"),
  costSavedBadge: document.querySelector("#cost-saved-badge"),
  semanticWarningsList: document.querySelector("#semantic-warnings-list"),
  textCompare: document.querySelector("#text-compare"),
  prunedColumn: document.querySelector("#pruned-column"),
  prunedTitle: document.querySelector("#pruned-title"),
  prunedCount: document.querySelector("#pruned-count"),
  prunedText: document.querySelector("#pruned-text"),
  auditStrip: document.querySelector("#audit-strip"),
  deletionLabel: document.querySelector("#deletion-label"),
  protectedLabel: document.querySelector("#protected-label"),
  substitutionLabel: document.querySelector("#substitution-label"),
  rejectedAuditCell: document.querySelector("#rejected-audit-cell"),
  rejectedCount: document.querySelector("#rejected-count"),
  rejectedLabel: document.querySelector("#rejected-label"),
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

function spanBounds(span) {
  if (!span) return { start: 0, end: 0 };
  const start = span.start != null ? span.start : span[0];
  const end = span.end != null ? span.end : span[1];
  return { start: start ?? 0, end: end ?? 0 };
}

function annotateOriginal(text, deleted, vetoed) {
  const offsets = codepointOffsets(text);
  const maxIndex = offsets.length - 1;
  const clampOffset = cp => offsets[Math.min(maxIndex, Math.max(0, cp))];
  const positionedDeleted = (deleted || []).map(item => {
    const raw = spanBounds(item.span);
    return { ...item, start: clampOffset(raw.start), end: clampOffset(raw.end) };
  });
  const positionedVetoed = (vetoed || []).map(item => {
    const raw = spanBounds(item.span);
    return { ...item, start: clampOffset(raw.start), end: clampOffset(raw.end) };
  });
  const boundaries = new Set([0, text.length]);
  [...positionedDeleted, ...positionedVetoed].forEach(item => {
    boundaries.add(item.start);
    boundaries.add(item.end);
  });
  const points = [...boundaries].sort((a, b) => a - b);
  let html = "";
  for (let index = 0; index < points.length - 1; index += 1) {
    const start = points[index];
    const end = points[index + 1];
    const chunk = escapeHtml(text.slice(start, end));
    const cut = positionedDeleted.find(item => item.start <= start && item.end >= end);
    const protectedSpan = positionedVetoed.find(item => item.start <= start && item.end >= end);
    if (cut) {
      html += `<mark class="annotation-cut" title="Removed: ${escapeHtml(cut.reason || cut.deprel || "")}">${chunk}</mark>`;
    } else if (protectedSpan) {
      html += `<mark class="annotation-protected" title="Protected: ${escapeHtml(protectedSpan.protect_class || "")}">${chunk}</mark>`;
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

function annotateCombinedStages(prunedText, finalProposalText, edits, decisions, treeSubstitutions = {}) {
  const offsets = codepointOffsets(prunedText);
  const maxIndex = offsets.length - 1;
  const clampOffset = cp => offsets[Math.min(maxIndex, Math.max(0, cp))];
  const semanticBackedOff = finalProposalText === prunedText;

  const positionedEdits = (edits || []).map(edit => {
    const raw = spanBounds(edit.span);
    return {
      ...edit,
      start: clampOffset(raw.start),
      end: clampOffset(raw.end),
    };
  }).sort((a, b) => a.start - b.start);

  const protectedDecisions = (decisions || []).filter(
    d => d.decision === "rejected" && d.reason && d.reason.startsWith("protected")
  ).map(decision => {
    const raw = spanBounds(decision.span);
    return {
      ...decision,
      start: clampOffset(raw.start),
      end: clampOffset(raw.end),
    };
  });

  const prunedBoundaries = new Set([0, prunedText.length]);
  if (!semanticBackedOff) {
    positionedEdits.forEach(e => {
      prunedBoundaries.add(e.start);
      prunedBoundaries.add(e.end);
    });
  }
  protectedDecisions.forEach(d => {
    prunedBoundaries.add(d.start);
    prunedBoundaries.add(d.end);
  });
  const prunedPoints = [...prunedBoundaries].sort((a, b) => a - b);

  let prunedHtml = "";
  if (prunedPoints.length <= 2 && Object.keys(treeSubstitutions || {}).length > 0) {
    prunedHtml = annotateCompressed(prunedText, treeSubstitutions);
  } else {
    for (let i = 0; i < prunedPoints.length - 1; i += 1) {
      const start = prunedPoints[i];
      const end = prunedPoints[i + 1];
      const chunk = escapeHtml(prunedText.slice(start, end));
      const edit = !semanticBackedOff ? positionedEdits.find(e => e.start <= start && e.end >= end) : null;
      const prot = protectedDecisions.find(d => d.start <= start && d.end >= end);
      if (edit) {
        const simText = edit.similarity != null ? `, cos: ${edit.similarity.toFixed(3)}` : "";
        const deltaText = edit.token_delta != null ? `, -${edit.token_delta} tok` : "";
        prunedHtml += `<mark class="annotation-substitution annotation-substitution-pending" title="[STAGE 2 PROPOSAL · UNREVIEWED] '${escapeHtml(edit.original)}' → '${escapeHtml(edit.replacement)}' (${escapeHtml(edit.provenance || "")}${simText}${deltaText})">${chunk}</mark>`;
      } else if (prot) {
        prunedHtml += `<mark class="annotation-protected" title="Protected in semantic stage: ${escapeHtml(prot.reason)}">${chunk}</mark>`;
      } else {
        prunedHtml += chunk;
      }
    }
  }

  if (semanticBackedOff || !positionedEdits.length) {
    return { prunedHtml, finalHtml: escapeHtml(finalProposalText) };
  }

  // Highlight proposed lexical replacements in finalProposalText, falling back cleanly if offsets diverge.
  let finalHtml = "";
  let prunedCursor = 0;
  let finalCursor = 0;
  for (const edit of positionedEdits) {
    const prefixLen = edit.start - prunedCursor;
    if (prefixLen < 0) {
      return { prunedHtml, finalHtml: escapeHtml(finalProposalText) };
    }
    const expectedPrefix = prunedText.slice(prunedCursor, edit.start);
    const actualPrefix = finalProposalText.slice(finalCursor, finalCursor + prefixLen);
    const replacement = String(edit.replacement ?? "");
    const actualRepl = finalProposalText.slice(finalCursor + prefixLen, finalCursor + prefixLen + replacement.length);
    if (expectedPrefix !== actualPrefix || actualRepl !== replacement) {
      return { prunedHtml, finalHtml: escapeHtml(finalProposalText) };
    }
    finalHtml += escapeHtml(actualPrefix);
    finalCursor += prefixLen;
    prunedCursor = edit.end;

    const sim = edit.similarity != null ? `, cos: ${edit.similarity.toFixed(3)}` : "";
    const delta = edit.token_delta != null ? `, -${edit.token_delta} tok` : "";
    finalHtml += `<mark class="annotation-substitution annotation-substitution-pending" title="Unreviewed proposal: substituted for '${escapeHtml(edit.original)}' (${escapeHtml(edit.provenance || "")}${sim}${delta})">${escapeHtml(actualRepl)}</mark>`;
    finalCursor += replacement.length;
  }
  finalHtml += escapeHtml(finalProposalText.slice(finalCursor));

  return { prunedHtml, finalHtml };
}

function buildAudit(data) {
  const deleted = data.deleted || [];
  const vetoed = data.vetoed || [];
  const substitutions = data.substitutions || {};
  const entries = [
    ...deleted.map(item => ({
      kind: "Removed",
      text: item.text,
      detail: `${item.deprel || "subtree"}${typeof item.score === "number" ? ` · ${item.score.toFixed(3)}` : ""}${item.reason ? ` · ${item.reason}` : ""}`,
    })),
    ...vetoed.map(item => ({ kind: "Protected", text: item.text, detail: item.protect_class })),
    ...Object.entries(substitutions).map(([from, to]) => ({ kind: "Shortened", text: `${from} → ${to}`, detail: "reversible" })),
  ];
  if (!entries.length) return '<p class="empty-audit">No edits were available at this setting.</p>';
  return entries.map(entry => `
    <div class="audit-item">
      <span class="audit-kind">${entry.kind}</span>
      <code>${escapeHtml(entry.text)}</code>
      <small>${escapeHtml(entry.detail)}</small>
    </div>`).join("");
}

function buildSemanticAudit(data, options = {}) {
  const edits = data.edits || [];
  const decisions = data.decisions || [];
  const rejected = decisions.filter(d => d.decision !== "selected");
  const selectedKind = options.unreviewedProposal ? "Proposed" : "Selected";
  const selectedHeading = options.unreviewedProposal
    ? `Proposed lexical edits (${edits.length} · unreviewed)`
    : `Selected edits (${edits.length})`;
  const spanLabelPrefix = options.prunedSpan ? "pruned span" : "span";

  if (!edits.length && !rejected.length) {
    return '<p class="empty-audit">No candidates were evaluated for this text.</p>';
  }

  let html = "";
  if (options.backoffNotice) {
    html += `<div class="empty-audit">${escapeHtml(options.backoffNotice)}</div>`;
  }

  if (edits.length) {
    html += `<div class="audit-section-header"><h4>${escapeHtml(selectedHeading)}</h4></div>`;
    html += edits.map(edit => {
      const sim = edit.similarity != null ? `cos: ${edit.similarity.toFixed(3)}` : "no vector score";
      const delta = edit.token_delta != null ? `-${edit.token_delta} tok` : "";
      const bounds = spanBounds(edit.span);
      return `
        <div class="audit-item audit-item-selected">
          <span class="audit-kind audit-kind-selected">${escapeHtml(selectedKind)}</span>
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
            <small class="audit-span">${spanLabelPrefix} ${bounds.start}:${bounds.end}</small>
          </div>
        </div>`;
    }).join("");
  }

  if (rejected.length) {
    html += `<div class="audit-section-header"><h4>Rejected candidates (${rejected.length})</h4></div>`;
    html += rejected.map(item => {
      const sim = item.similarity != null ? `cos: ${item.similarity.toFixed(3)}` : "no vector score";
      const delta = item.token_delta != null ? `${item.token_delta > 0 ? '-' + item.token_delta : item.token_delta} tok` : "0 tok";
      const bounds = spanBounds(item.span);
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
            <small class="audit-span">${spanLabelPrefix} ${bounds.start}:${bounds.end} · rank ${item.candidate_rank ?? 0}</small>
          </div>
        </div>`;
    }).join("");
  }

  return html;
}

function buildCombinedAudit(treeData, semanticData, stageStats) {
  const deletedCount = (treeData.deleted || []).length;
  const vetoedCount = (treeData.vetoed || []).length;
  const treeSubCount = Object.keys(treeData.substitutions || {}).length;
  const editsCount = (semanticData.edits || []).length;
  const rejectedCount = (semanticData.decisions || []).filter(d => d.decision !== "selected").length;
  const backoffNotice =
    stageStats.semanticBackedOff && editsCount > 0
      ? "Full-text token recount found no net token savings from Stage 2 proposals; final output retains Stage 1 pruned text."
      : "";

  const treeHeader = `
    <div class="audit-stage-divider">
      <h4>Stage 1 · Tree deletion &amp; protection audit</h4>
      <small>${deletedCount} cut${deletedCount === 1 ? "" : "s"} · ${vetoedCount} protected${treeSubCount ? ` · ${treeSubCount} shortened` : ""} · -${stageStats.treeSavedTokens} tok</small>
    </div>`;

  const semanticHeader = `
    <div class="audit-stage-divider">
      <h4>Stage 2 · Semantic edit &amp; rejection audit on pruned text (unreviewed)</h4>
      <small>${editsCount} proposed · ${rejectedCount} rejected · -${stageStats.semanticSavedTokens} tok · template persistence disabled</small>
    </div>`;

  return (
    treeHeader +
    buildAudit(treeData) +
    semanticHeader +
    buildSemanticAudit(semanticData, {
      unreviewedProposal: true,
      prunedSpan: true,
      backoffNotice,
    })
  );
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
let compressRequestId = 0;
let inFlightCompressId = 0;

function captureWorkbenchSnapshot() {
  return {
    mode: selectedMode(),
    source: elements.source ? elements.source.value : "",
    ratio: elements.ratio ? String(elements.ratio.value) : "",
    substitute: Boolean(elements.substitute && elements.substitute.checked),
    arm: elements.semanticArm ? elements.semanticArm.value : "",
    encoding: elements.semanticEncoding ? elements.semanticEncoding.value : "",
    vectorSource: elements.semanticVectorSource ? elements.semanticVectorSource.value : "",
    vectorPath: elements.semanticVectorPath ? elements.semanticVectorPath.value.trim() : "",
    vectorMaxWords: elements.semanticVectorMaxWords ? String(elements.semanticVectorMaxWords.value) : "",
    price: elements.semanticPrice ? String(elements.semanticPrice.value) : "",
    maxCandidates: elements.semanticMaxCandidates ? String(elements.semanticMaxCandidates.value) : "",
    embeddingSlots: elements.semanticEmbeddingSlots ? String(elements.semanticEmbeddingSlots.value) : "",
    useWordnet: Boolean(elements.semanticWordnet && elements.semanticWordnet.checked),
    similarityToggle: Boolean(elements.semanticSimilarityToggle && elements.semanticSimilarityToggle.checked),
    minSimilarity: elements.semanticMinSimilarity ? String(elements.semanticMinSimilarity.value) : "",
    adaptiveSimilarity: Boolean(elements.semanticAdaptiveSimilarity && elements.semanticAdaptiveSimilarity.checked),
  };
}

function matchesWorkbenchSnapshot(snapshot) {
  if (!snapshot) return false;
  const current = captureWorkbenchSnapshot();
  return Object.keys(snapshot).every(key => snapshot[key] === current[key]);
}

function invalidateInFlightCompression() {
  compressRequestId += 1;
  previewRequestId += 1;
  if (inFlightCompressId !== 0) {
    inFlightCompressId = 0;
    if (elements.button) {
      elements.button.disabled = false;
      elements.button.textContent = "Compress text";
    }
  }
}

function isCurrentCompressionRequest(requestId, requestSnapshot) {
  return (
    requestId === compressRequestId &&
    requestId === inFlightCompressId &&
    matchesWorkbenchSnapshot(requestSnapshot)
  );
}

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
  if (selectedMode() !== "semantic" || !semanticReviewState.active || semanticReviewState.invalidated) return;
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

    // Prevent race conditions: discard superseded response or mode switch away from standalone semantic
    if (
      currentReqId !== previewRequestId ||
      selectedMode() !== "semantic" ||
      !semanticReviewState.active ||
      semanticReviewState.invalidated
    ) {
      return;
    }

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
    if (
      currentReqId === previewRequestId &&
      selectedMode() === "semantic" &&
      semanticReviewState.active &&
      elements.approvalBannerMessage
    ) {
      elements.approvalBannerMessage.textContent = err.message;
      elements.approvalBannerMessage.className = "approval-banner-message error";
      elements.approvalBannerMessage.hidden = false;
    }
  }
}

async function persistApprovedTemplate() {
  if (selectedMode() !== "semantic" || !semanticReviewState.active || semanticReviewState.invalidated) return;
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
    if (
      selectedMode() !== "semantic" ||
      !semanticReviewState.active ||
      semanticReviewState.invalidated ||
      savingVersion !== semanticReviewState.reviewVersion ||
      elements.source.value !== semanticReviewState.originalText
    ) {
      if (selectedMode() === "semantic" && semanticReviewState.active) {
        invalidateStaleReviewState("Source text or semantic settings changed while saving. Re-run compression before exporting.");
      }
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
    if (selectedMode() === "semantic" && semanticReviewState.active && elements.approvalBannerMessage) {
      elements.approvalBannerMessage.textContent = err.message;
      elements.approvalBannerMessage.className = "approval-banner-message error";
      elements.approvalBannerMessage.hidden = false;
    }
  } finally {
    elements.persistTemplateButton.textContent = "Save Approved Template";
    const hasPending = semanticReviewState.occurrences.some(o => o.decision === "pending");
    const hasAccepted = semanticReviewState.occurrences.some(o => o.decision === "accepted");
    elements.persistTemplateButton.disabled =
      selectedMode() !== "semantic" ||
      !semanticReviewState.active ||
      semanticReviewState.invalidated ||
      hasPending ||
      !hasAccepted ||
      elements.source.value !== semanticReviewState.originalText;
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

function populateSemanticMetadataBanner(semanticData, options = {}) {
  if (!elements.semanticBanner) return;
  const isCombined = Boolean(options.isCombined);
  const treeData = options.treeData || null;
  const totalSavedTokens = options.totalSavedTokens ?? semanticData.saved_tokens ?? 0;

  if (elements.reviewStatusBadge) {
    if (isCombined) {
      elements.reviewStatusBadge.textContent = "Review: UNREVIEWED (NO TEMPLATE PERSISTENCE)";
      elements.reviewStatusBadge.className = "meta-badge badge-pending";
    } else {
      const status = semanticData.review_status || "pending";
      elements.reviewStatusBadge.textContent = `Review: ${status}`;
      elements.reviewStatusBadge.className = "meta-badge badge-pending";
    }
  }
  if (elements.tokenCounterBadge) {
    const counterLabel =
      semanticData.token_counter || (treeData && treeData.token_counter) || "tiktoken";
    elements.tokenCounterBadge.textContent = `Tokenizer: ${counterLabel}`;
  }

  const providerMeta = semanticData.provider_metadata || {};
  const vm = providerMeta.vectors || semanticData.vector_size || null;
  const vectorLabel = semanticData.vector_label || "none";
  const isSynthetic =
    vectorLabel.toLowerCase().includes("synthetic") ||
    providerMeta.vector_source === "synthetic";
  const armUpper = (providerMeta.arm ? providerMeta.arm : "b0").toUpperCase();

  if (elements.vectorLabelBadge) {
    if (armUpper === "B0") {
      elements.vectorLabelBadge.textContent = "Vectors: none";
      elements.vectorLabelBadge.title = "B0 uses curated rules only. Vectors not loaded.";
    } else if (isSynthetic) {
      elements.vectorLabelBadge.textContent =
        "Vectors: Bundled synthetic fixture (software fixture, not trained embeddings)";
      elements.vectorLabelBadge.title = vectorLabel || "Bundled synthetic software fixture";
    } else {
      elements.vectorLabelBadge.textContent = `Vectors: ${vectorLabel || "custom local file"}`;
      elements.vectorLabelBadge.title = vectorLabel || "";
    }
  }

  if (elements.vectorRowsBadge) {
    if (vm && armUpper !== "B0" && vm.retained_rows != null && vm.source_rows != null) {
      const retained = vm.retained_rows.toLocaleString();
      const source = vm.source_rows.toLocaleString();
      const maxWords = providerMeta.vectors && providerMeta.vectors.max_words != null
        ? providerMeta.vectors.max_words
        : vm.max_words;
      const cap = maxWords ? (maxWords >= 1000 ? `${maxWords / 1000}k` : maxWords) : "none";
      const fixtureTag = isSynthetic ? "Synthetic fixture · " : "";
      elements.vectorRowsBadge.textContent = `${fixtureTag}${retained} of ${source} rows retained (cap: ${cap})`;
      elements.vectorRowsBadge.hidden = false;
      if (vm.retained_rows >= 400000) {
        elements.vectorRowsBadge.classList.add("badge-warning");
        elements.vectorRowsBadge.title =
          "400k full retention selected: cold load may incur higher initial latency and memory usage.";
      } else {
        elements.vectorRowsBadge.classList.remove("badge-warning");
        elements.vectorRowsBadge.title = "";
      }
    } else {
      elements.vectorRowsBadge.hidden = true;
    }
  }

  if (elements.configHashBadge) {
    if (isCombined && treeData) {
      const treeHash = (treeData.config_hash || "unknown").slice(0, 8);
      const semHash = (semanticData.config_hash || "unknown").slice(0, 8);
      elements.configHashBadge.textContent = `Config: tree ${treeHash} · sem ${semHash}`;
      elements.configHashBadge.title = `Tree config: ${treeData.config_hash || "unknown"} | Semantic config: ${semanticData.config_hash || "unknown"}`;
    } else {
      elements.configHashBadge.textContent = `Config: ${(semanticData.config_hash || "unknown").slice(0, 12)}`;
      elements.configHashBadge.title = semanticData.config_hash || "Configuration hash unavailable";
    }
  }

  if (elements.costSavedBadge) {
    const price =
      options.inputPricePerMillion != null
        ? options.inputPricePerMillion
        : semanticData.input_price_per_million != null
        ? semanticData.input_price_per_million
        : 0;
    const costSaved = isCombined
      ? (options.totalCostSavedUsd != null ? options.totalCostSavedUsd : (totalSavedTokens * price) / 1_000_000)
      : (semanticData.cost_saved_usd || 0);
    if (isCombined && !(price > 0)) {
      elements.costSavedBadge.hidden = true;
      elements.costSavedBadge.textContent = "";
    } else {
      elements.costSavedBadge.hidden = false;
      elements.costSavedBadge.textContent = `Gross input est.: $${costSaved.toFixed(6)} (@ $${price.toFixed(2)}/1M · unreviewed proposal)`;
    }
  }

  if (elements.semanticWarningsList) {
    const rawWarnings = [
      ...((treeData && treeData.warnings) || []),
      ...(semanticData.warnings || []),
      ...(options.extraWarnings || []),
    ];
    const warnings = [...new Set(rawWarnings.filter(Boolean))];
    if (warnings.length > 0) {
      elements.semanticWarningsList.innerHTML = warnings
        .map(w => `<div class="warning-callout">${escapeHtml(w)}</div>`)
        .join("");
      elements.semanticWarningsList.hidden = false;
    } else {
      elements.semanticWarningsList.innerHTML = "";
      elements.semanticWarningsList.hidden = true;
    }
  }
}

function renderResult(data, requestPayload = null) {
  // Ignore any preview response started for the preceding compression result.
  previewRequestId += 1;
  const activeUiMode = selectedMode();
  const mode = data.mode;
  state.resultMode = mode;
  if (
    (requestPayload && requestPayload.mode && requestPayload.mode !== activeUiMode) ||
    (mode && mode !== activeUiMode)
  ) {
    if (activeUiMode !== "semantic") {
      semanticReviewState.active = false;
      if (elements.semanticApprovalWorkflow) {
        elements.semanticApprovalWorkflow.hidden = true;
      }
      if (elements.persistTemplateButton) elements.persistTemplateButton.disabled = true;
    }
    return;
  }
  const isHybrid = mode === "hybrid_sentinel";
  const isSemantic = mode === "semantic";
  const isTreeSemantic = mode === "tree_semantic";

  if (elements.unreviewedCombinedBanner) {
    elements.unreviewedCombinedBanner.hidden = !isTreeSemantic;
  }
  if (elements.stageBreakdown) {
    elements.stageBreakdown.hidden = !isTreeSemantic;
  }
  if (elements.prunedColumn) {
    elements.prunedColumn.hidden = !isTreeSemantic;
  }
  if (elements.textCompare) {
    elements.textCompare.classList.toggle("three-stage", isTreeSemantic);
  }
  if (elements.rejectedAuditCell) {
    elements.rejectedAuditCell.hidden = !isTreeSemantic;
  }
  if (elements.auditStrip) {
    elements.auditStrip.classList.toggle("four-stat", isTreeSemantic);
  }

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
  } else if (isTreeSemantic) {
    // Approved-template persistence is explicitly disabled in combined tree_semantic mode.
    semanticReviewState.active = false;
    semanticReviewState.savedTemplate = null;
    semanticReviewState.savedAudit = null;
    if (elements.semanticApprovalWorkflow) {
      elements.semanticApprovalWorkflow.hidden = true;
    }
    if (elements.persistTemplateButton) elements.persistTemplateButton.disabled = true;
    if (elements.copyApprovedButton) elements.copyApprovedButton.disabled = true;
    if (elements.exportTextButton) elements.exportTextButton.disabled = true;
    if (elements.exportAuditButton) elements.exportAuditButton.disabled = true;

    const treeData = data.tree || {};
    const semanticData = data.semantic || {};
    const originalText = data.original != null ? data.original : (treeData.original || "");
    const prunedText = data.pruned != null ? data.pruned : (treeData.compressed != null ? treeData.compressed : (semanticData.original || ""));
    const finalText = data.compressed != null ? data.compressed : (semanticData.compressed != null ? semanticData.compressed : prunedText);

    const tokensBefore = data.tokens_before ?? treeData.tokens_before ?? 0;
    const tokensAfterTree = data.tokens_after_tree ?? treeData.tokens_after ?? semanticData.tokens_before ?? tokensBefore;
    const tokensAfter = data.tokens_after ?? semanticData.tokens_after ?? tokensAfterTree;
    const treeSavedTokens = data.tree_saved_tokens ?? (tokensBefore - tokensAfterTree);
    const semanticSavedTokens = data.semantic_saved_tokens ?? (tokensAfterTree - tokensAfter);
    const savedTokens = data.saved_tokens ?? (tokensBefore - tokensAfter);
    const ratio = data.ratio != null ? data.ratio : (tokensBefore > 0 ? tokensAfter / tokensBefore : 1);
    const savedPercent = data.saved_percent != null
      ? data.saved_percent
      : (tokensBefore > 0 ? Number(((savedTokens / tokensBefore) * 100).toFixed(1)) : 0);

    const armLabel = (
      (semanticData.provider_metadata && semanticData.provider_metadata.arm) ||
      (requestPayload && requestPayload.arm) ||
      "b0"
    ).toUpperCase();
    const counterLabel = semanticData.token_counter || treeData.token_counter || "tiktoken";
    const targetRatioPct = Math.round(
      ((data.target_ratio != null ? data.target_ratio : (requestPayload && requestPayload.target_ratio != null ? requestPayload.target_ratio : 0.6)) * 100)
    );

    document.querySelector("#result-mode").textContent =
      `Combined tree pruning (${targetRatioPct}% target) + semantic substitution (${armLabel}) · ${counterLabel} · unreviewed proposal`;

    document.querySelector("#tokens-after").textContent = tokensAfter;
    document.querySelector("#tokens-saved").textContent = savedTokens;
    document.querySelector("#ruler-fill").style.width = `${Math.min(100, Math.max(0, ratio * 100))}%`;

    const inputPrice =
      data.input_price_per_million != null
        ? data.input_price_per_million
        : requestPayload && requestPayload.input_price_per_million != null
        ? requestPayload.input_price_per_million
        : semanticData.input_price_per_million != null
        ? semanticData.input_price_per_million
        : 0;
    const totalCostSavedUsd = (savedTokens * inputPrice) / 1_000_000;

    if (elements.costSavedSummary) {
      elements.costSavedSummary.textContent =
        inputPrice > 0
          ? ` · $${totalCostSavedUsd.toFixed(6)} gross est. (unreviewed proposal)`
          : " · unreviewed combined proposal";
    }

    if (elements.stageBeforeTokens) elements.stageBeforeTokens.textContent = `${tokensBefore} tokens`;
    if (elements.stageBeforeDetail) {
      elements.stageBeforeDetail.textContent = `input · ${counterLabel}`;
    }
    if (elements.stageTreeTokens) elements.stageTreeTokens.textContent = `${tokensAfterTree} tokens`;
    if (elements.stageTreeDetail) {
      const treePct = tokensBefore > 0 ? ((treeSavedTokens / tokensBefore) * 100).toFixed(1) : "0.0";
      const treeMs = data.timings_ms && data.timings_ms.tree != null ? ` · ${Math.round(data.timings_ms.tree)} ms` : "";
      elements.stageTreeDetail.textContent = `-${treeSavedTokens} tok (${treePct}%) · target ${targetRatioPct}%${treeMs}`;
    }
    if (elements.stageFinalTokens) elements.stageFinalTokens.textContent = `${tokensAfter} tokens`;
    if (elements.stageFinalDetail) {
      const semMs = data.timings_ms && data.timings_ms.semantic != null ? ` · ${Math.round(data.timings_ms.semantic)} ms` : "";
      elements.stageFinalDetail.textContent =
        `-${semanticSavedTokens} lexical · -${savedTokens} total (${savedPercent}% saved · unreviewed)${semMs}`;
    }

    document.querySelector("#original-title").textContent = "Original, tree-annotated";
    document.querySelector("#original-count").textContent = `${tokensBefore} tokens`;
    if (elements.prunedTitle) elements.prunedTitle.textContent = "Middle · Pruned text";
    if (elements.prunedCount) {
      elements.prunedCount.textContent = `${tokensAfterTree} tokens · -${treeSavedTokens} tree`;
    }
    document.querySelector("#compressed-title").textContent = "Final lexical proposal (unreviewed)";
    document.querySelector("#compressed-count").textContent =
      `${tokensAfter} tokens · -${semanticSavedTokens} lexical · ${savedPercent}% total`;

    document.querySelector("#annotated-text").innerHTML = annotateOriginal(
      originalText,
      treeData.deleted || [],
      treeData.vetoed || []
    );
    const { prunedHtml, finalHtml } = annotateCombinedStages(
      prunedText,
      finalText,
      semanticData.edits || [],
      semanticData.decisions || [],
      treeData.substitutions || {}
    );
    if (elements.prunedText) elements.prunedText.innerHTML = prunedHtml;
    document.querySelector("#compressed-text").innerHTML = finalHtml;

    if (elements.semanticBanner) {
      elements.semanticBanner.hidden = false;
      populateSemanticMetadataBanner(semanticData, {
        isCombined: true,
        treeData,
        totalSavedTokens: savedTokens,
        inputPricePerMillion: inputPrice,
        totalCostSavedUsd,
        extraWarnings: data.warnings || [],
      });
    }

    const treeCutsCount = (treeData.deleted || []).length;
    const treeVetoedCount = (treeData.vetoed || []).length;
    const semanticEditsCount = (semanticData.edits || []).length;
    const semanticRejectedCount = (semanticData.decisions || []).filter(d => d.decision !== "selected").length;

    document.querySelector("#deletion-count").textContent = treeCutsCount;
    if (elements.deletionLabel) elements.deletionLabel.textContent = "tree cuts made";
    document.querySelector("#protected-count").textContent = treeVetoedCount;
    if (elements.protectedLabel) elements.protectedLabel.textContent = "tree spans protected";
    document.querySelector("#substitution-count").textContent = semanticEditsCount;
    if (elements.substitutionLabel) elements.substitutionLabel.textContent = "lexical proposals (unreviewed)";
    if (elements.rejectedCount) elements.rejectedCount.textContent = semanticRejectedCount;
    if (elements.rejectedLabel) elements.rejectedLabel.textContent = "lexical candidates rejected";

    if (elements.auditSummaryLabel) {
      elements.auditSummaryLabel.textContent =
        `Combined audit trail (${treeCutsCount} tree cuts, ${treeVetoedCount} protected · ${semanticEditsCount} lexical proposed, ${semanticRejectedCount} rejected)`;
    }
    document.querySelector("#audit-list").innerHTML = buildCombinedAudit(treeData, semanticData, {
      treeSavedTokens,
      semanticSavedTokens,
      semanticBackedOff: finalText === prunedText,
    });
    document.querySelector("#audit-details").open = true;
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

  const totalMs = data.timings_ms
    ? (data.timings_ms.total != null
      ? data.timings_ms.total
      : ((data.tree && data.tree.timings_ms && data.tree.timings_ms.total) || 0) +
        ((data.semantic && data.semantic.timings_ms && data.semantic.timings_ms.total) || 0))
    : 0;
  document.querySelector("#time-taken").textContent = Math.round(totalMs);

  // Semantic metadata banner & warnings for standalone semantic mode
  if (!isTreeSemantic && elements.semanticBanner) {
    elements.semanticBanner.hidden = !isSemantic;
    if (isSemantic) {
      populateSemanticMetadataBanner(data, { isCombined: false });
    }
  }

  // Annotations for standalone modes
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
  } else if (!isTreeSemantic) {
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
  } else if (!isTreeSemantic) {
    document.querySelector("#deletion-count").textContent = (data.deleted || []).length;
    if (elements.deletionLabel) elements.deletionLabel.textContent = "cuts made";
    document.querySelector("#protected-count").textContent = (data.vetoed || []).length;
    if (elements.protectedLabel) elements.protectedLabel.textContent = "spans protected";
    document.querySelector("#substitution-count").textContent = Object.keys(data.substitutions || {}).length;
    if (elements.substitutionLabel) elements.substitutionLabel.textContent = "phrases shortened";
    if (elements.auditSummaryLabel) elements.auditSummaryLabel.textContent = "Open the audit trail";
    document.querySelector("#audit-list").innerHTML = buildAudit(data);
  }

  if (selectedMode() !== "semantic") {
    semanticReviewState.active = false;
    if (elements.semanticApprovalWorkflow) {
      elements.semanticApprovalWorkflow.hidden = true;
    }
    if (elements.persistTemplateButton) elements.persistTemplateButton.disabled = true;
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
  invalidateInFlightCompression();
}

function updateMode() {
  const mode = selectedMode();
  if (state.resultMode && state.resultMode !== mode) {
    elements.result.hidden = true;
    state.resultMode = null;
  }
  const isHybrid = mode === "hybrid_sentinel";
  const isSemantic = mode === "semantic";
  const isTreeSemantic = mode === "tree_semantic";
  const usesSemanticControls = isSemantic || isTreeSemantic;

  // Show target_ratio slider in tree, hybrid, and combined tree_semantic modes; hide only in standalone semantic mode.
  if (elements.standardControls) {
    elements.standardControls.hidden = isSemantic;
  }
  elements.substitute.disabled = isHybrid || isTreeSemantic;
  if (elements.substituteControl) {
    elements.substituteControl.hidden = isTreeSemantic;
  }

  // Toggle semantic settings visibility for semantic and tree_semantic modes.
  if (elements.semanticSettings) {
    elements.semanticSettings.hidden = !usesSemanticControls;
    elements.semanticSettings.disabled = !usesSemanticControls;
  }
  if (elements.semanticSettingsLegend) {
    elements.semanticSettingsLegend.textContent = isTreeSemantic
      ? "Stage 2 · Semantic substitution settings (run on pruned text)"
      : "Semantic substitution settings";
  }
  if (elements.semanticWarningCopy) {
    elements.semanticWarningCopy.textContent = isTreeSemantic
      ? "Two-stage experimental mode: deterministic tree pruning runs first at the target ratio, then offline lexical proposals run on the pruned text. Combined output is unreviewed and approved-template persistence is disabled."
      : "Offline lexical suggestions with local assets. Token savings, cosine similarity, and grammar guards do not establish meaning preservation; human review is required before using rewritten prompts.";
  }

  // Lock tokenizer to o200k_base in combined tree_semantic mode so both stages share one counter.
  if (elements.semanticEncoding) {
    if (isTreeSemantic) {
      elements.semanticEncoding.value = "o200k_base";
      elements.semanticEncoding.disabled = true;
    } else {
      elements.semanticEncoding.disabled = false;
    }
  }
  if (elements.semanticEncodingCaption) {
    elements.semanticEncodingCaption.textContent = isTreeSemantic
      ? "Pinned to o200k_base so tree pruning and semantic stages share one recount."
      : "Tokenizer used for full-text recount.";
  }

  // Never keep approved-template persistence active outside standalone semantic mode.
  if (!isSemantic) {
    previewRequestId += 1;
    if (semanticReviewState.active) {
      semanticReviewState.active = false;
      semanticReviewState.savedTemplate = null;
      semanticReviewState.savedAudit = null;
    }
    if (elements.semanticApprovalWorkflow) {
      elements.semanticApprovalWorkflow.hidden = true;
    }
    if (elements.persistTemplateButton) elements.persistTemplateButton.disabled = true;
    if (elements.copyApprovedButton) elements.copyApprovedButton.disabled = true;
    if (elements.exportTextButton) elements.exportTextButton.disabled = true;
    if (elements.exportAuditButton) elements.exportAuditButton.disabled = true;
  }

  // Update mode note
  if (isTreeSemantic) {
    elements.modeNote.textContent =
      "Opt-in combined run: tree pruning cuts subtrees at the target ratio, then offline lexical proposals run on the pruned text. Combined output is unreviewed and cannot be saved as an approved template.";
  } else if (isSemantic) {
    elements.modeNote.textContent =
      "Offline lexical suggestions with local vectors and structural guards. Human review is required before using rewritten prompts.";
  } else if (isHybrid) {
    elements.modeNote.textContent =
      "Experimental. First run loads LLMLingua-2; protected clauses are restored, while model edits have no word-level audit.";
  } else {
    elements.modeNote.textContent =
      "Tree pruning shows each deletion and protection decision.";
  }

  if (usesSemanticControls) {
    updateSemanticSettings();
  }
}

function buildSemanticSettingsPayload() {
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
  const adaptiveSimilarity = Boolean(similarityOn && elements.semanticAdaptiveSimilarity.checked);
  const rawPrice = elements.semanticPrice ? parseFloat(elements.semanticPrice.value) : 0.0;
  const price = Number.isFinite(rawPrice) ? Math.max(0, rawPrice) : 0.0;

  const settings = {
    arm,
    use_wordnet: useWordnet,
    vector_source: vectorSource,
    vector_max_words: vectorMaxWords,
    encoding,
    max_candidates: maxCandidates,
    embedding_slots: embeddingSlots,
    min_embedding_similarity: minSimilarity,
    adaptive_similarity: adaptiveSimilarity,
    input_price_per_million: price,
  };

  if (arm !== "b0" && vectorSource === "local") {
    if (!vectorPath) {
      throw new Error("Enter a valid local vector file path or choose bundled synthetic vectors.");
    }
    settings.vector_path = vectorPath;
  }

  return settings;
}

async function compressText() {
  const requestId = ++compressRequestId;
  inFlightCompressId = requestId;
  previewRequestId += 1;
  const requestSnapshot = captureWorkbenchSnapshot();

  elements.error.hidden = true;
  elements.button.disabled = true;
  const mode = requestSnapshot.mode;
  elements.button.textContent =
    mode === "hybrid_sentinel"
      ? "Loading or compressing…"
      : mode === "semantic"
      ? "Finding suggestions…"
      : mode === "tree_semantic"
      ? "Pruning + proposing…"
      : "Compressing…";

  try {
    let payload;
    if (mode === "semantic") {
      payload = {
        text: requestSnapshot.source,
        mode: "semantic",
        ...buildSemanticSettingsPayload(),
      };
    } else if (mode === "tree_semantic") {
      payload = {
        text: requestSnapshot.source,
        mode: "tree_semantic",
        target_ratio: Number(requestSnapshot.ratio) / 100,
        ...buildSemanticSettingsPayload(),
      };
    } else {
      payload = {
        text: requestSnapshot.source,
        target_ratio: Number(requestSnapshot.ratio) / 100,
        substitute: requestSnapshot.substitute,
        mode: mode,
      };
    }

    const response = await fetch("/api/compress", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!isCurrentCompressionRequest(requestId, requestSnapshot)) {
      return;
    }
    if (!response.ok) throw new Error(data.error || "Compression failed.");
    renderResult(data, payload);
  } catch (error) {
    if (!isCurrentCompressionRequest(requestId, requestSnapshot)) {
      return;
    }
    elements.error.textContent = error.message;
    elements.error.hidden = false;
  } finally {
    if (requestId === inFlightCompressId) {
      inFlightCompressId = 0;
      elements.button.disabled = false;
      elements.button.textContent = "Compress text";
    }
  }
}

function renderExamples(examples) {
  elements.exampleList.innerHTML = examples.map(example => {
    const isTreeSemantic = example.mode === "tree_semantic";
    const isSemantic = example.mode === "semantic";
    const isHybrid = example.mode === "hybrid_sentinel";
    const tagClass = isTreeSemantic
      ? "tag-combined"
      : isSemantic
      ? "tag-semantic"
      : isHybrid
      ? "tag-hybrid"
      : "tag-tree";
    const tagLabel = isTreeSemantic
      ? `Tree + Semantic · ${(example.arm || "b0").toUpperCase()}`
      : isSemantic
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
      invalidateInFlightCompression();
      elements.source.value = example.text;
      const targetMode = example.mode || "conservative";
      elements.modes.forEach(mode => {
        mode.checked = mode.value === targetMode;
      });

      if (example.target_ratio != null && elements.ratio) {
        const pct = Math.min(100, Math.max(10, Math.round(Number(example.target_ratio) * 100 / 5) * 5));
        elements.ratio.value = String(pct);
        if (elements.ratioOutput) {
          elements.ratioOutput.textContent = `${pct}%`;
        }
      }

      if (targetMode === "semantic" || targetMode === "tree_semantic") {
        if (example.arm && elements.semanticArm) {
          elements.semanticArm.value = example.arm;
        }
        if (elements.semanticVectorSource) {
          elements.semanticVectorSource.value = example.vector_source || "synthetic";
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

function withCombinedSample(examples) {
  const list = Array.isArray(examples) ? [...examples] : [];
  if (!list.some(item => item && item.mode === "tree_semantic")) {
    list.push(TREE_SEMANTIC_SAMPLE);
  }
  return list;
}

async function loadExamples() {
  try {
    const response = await fetch("/api/examples");
    const data = await response.json();
    state.examples = withCombinedSample(data.examples);
    renderExamples(state.examples);
  } catch (_) {
    state.examples = [TREE_SEMANTIC_SAMPLE];
    renderExamples(state.examples);
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
elements.source.addEventListener("input", () => {
  invalidateInFlightCompression();
  updateCharacterCount();
});
elements.ratio.addEventListener("input", () => {
  elements.ratioOutput.textContent = `${elements.ratio.value}%`;
  invalidateInFlightCompression();
});
elements.ratio.addEventListener("change", () => {
  elements.ratioOutput.textContent = `${elements.ratio.value}%`;
  invalidateInFlightCompression();
});
if (elements.substitute) {
  elements.substitute.addEventListener("change", invalidateInFlightCompression);
}
elements.modes.forEach(mode => mode.addEventListener("change", () => {
  invalidateInFlightCompression();
  updateMode();
}));

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
  const handleMinSimilarityChange = () => {
    if (elements.semanticMinSimilarityOutput) {
      elements.semanticMinSimilarityOutput.textContent = elements.semanticMinSimilarity.value;
    }
    invalidateInFlightCompression();
    if (semanticReviewState.active) {
      invalidateStaleReviewState("Semantic settings changed. Re-run compression before saving.");
    }
  };
  elements.semanticMinSimilarity.addEventListener("input", handleMinSimilarityChange);
  elements.semanticMinSimilarity.addEventListener("change", handleMinSimilarityChange);
}
if (elements.semanticPrice) {
  elements.semanticPrice.addEventListener("input", invalidateInFlightCompression);
  elements.semanticPrice.addEventListener("change", invalidateInFlightCompression);
}
[
  elements.semanticEncoding,
  elements.semanticVectorPath,
  elements.semanticVectorMaxWords,
  elements.semanticEmbeddingSlots,
  elements.semanticWordnet,
  elements.semanticAdaptiveSimilarity,
].filter(Boolean).forEach(control => {
  const handleSemanticControlChange = () => {
    invalidateInFlightCompression();
    if (semanticReviewState.active) {
      invalidateStaleReviewState("Semantic settings changed. Re-run compression before saving.");
    }
  };
  control.addEventListener("input", handleSemanticControlChange);
  control.addEventListener("change", handleSemanticControlChange);
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
