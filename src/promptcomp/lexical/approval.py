"""Explicit review and approval workflow for lexical substitutions.

Review-first production workflow: unreviewed suggestions do not establish
semantic equivalence or production quality. A template is persisted only via
explicit human review action and is strictly bound to original content hash,
configuration hash, tokenizer identity, and vector provenance.

Any content or configuration mismatch returns unchanged text with clear
invalidation, preventing automated live rewriting or false production quality claims.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Sequence

from ..protect import (
    _NEGATION_RE,
    _NUMBER_RE,
    extract_defined_terms,
    load_defined_terms,
    load_protect_list,
)
from ..segment import segment
from ..tokens import TokenCounter
from ..types import Span
from .protection import (
    additional_defined_terms,
    frozen_ranges,
    normalize_protected_spans,
    overlaps,
)
from .tokenization import local_counter

APPROVAL_SCHEMA_VERSION = "lexical-approval-1"
WORKFLOW_DISCLAIMER = (
    "Review-first production workflow: unreviewed suggestions do not establish "
    "semantic equivalence or production quality. Human review is required before "
    "using rewritten prompts. Persisted templates are bound to exact content, "
    "configuration, tokenizer, and vector identity hashes."
)


def compute_content_hash(text: str) -> str:
    """Compute the deterministic SHA-256 hex digest of UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class OccurrenceReview:
    """Explicit review state for a single occurrence substitution."""

    span: Span
    original: str
    replacement: str
    decision: str = "pending"  # "pending" | "accepted" | "rejected"
    provenance: str = ""
    similarity: float | None = None
    token_delta: int | None = None
    pos: str | None = None
    sentence_before: str = ""
    sentence_after: str = ""
    notes: str = ""

    def __post_init__(self):
        if self.decision not in {"pending", "accepted", "rejected"}:
            raise ValueError(f"Invalid decision '{self.decision}'; must be pending, accepted, or rejected.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "span": {"start": self.span.start, "end": self.span.end},
            "original": self.original,
            "replacement": self.replacement,
            "decision": self.decision,
            "provenance": self.provenance,
            "similarity": self.similarity,
            "token_delta": self.token_delta,
            "pos": self.pos,
            "sentence_before": self.sentence_before,
            "sentence_after": self.sentence_after,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OccurrenceReview:
        raw_span = data["span"]
        if isinstance(raw_span, Span):
            span = raw_span
        elif isinstance(raw_span, dict):
            span = Span(int(raw_span["start"]), int(raw_span["end"]))
        elif isinstance(raw_span, (list, tuple)) and len(raw_span) == 2:
            span = Span(int(raw_span[0]), int(raw_span[1]))
        else:
            raise ValueError(f"Invalid span format: {raw_span!r}")

        return cls(
            span=span,
            original=str(data["original"]),
            replacement=str(data["replacement"]),
            decision=str(data.get("decision", "pending")),
            provenance=str(data.get("provenance", "")),
            similarity=float(data["similarity"]) if data.get("similarity") is not None else None,
            token_delta=int(data["token_delta"]) if data.get("token_delta") is not None else None,
            pos=data.get("pos"),
            sentence_before=str(data.get("sentence_before", "")),
            sentence_after=str(data.get("sentence_after", "")),
            notes=str(data.get("notes", "")),
        )


@dataclass(frozen=True)
class TemplateBinding:
    """Cryptographic and metadata binding for an approved template."""

    original_hash: str
    config_hash: str
    tokenizer_name: str
    vector_identity: str

    def to_dict(self) -> dict[str, str]:
        return {
            "original_hash": self.original_hash,
            "config_hash": self.config_hash,
            "tokenizer_name": self.tokenizer_name,
            "vector_identity": self.vector_identity,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemplateBinding:
        return cls(
            original_hash=str(data["original_hash"]),
            config_hash=str(data["config_hash"]),
            tokenizer_name=str(data["tokenizer_name"]),
            vector_identity=str(data["vector_identity"]),
        )


@dataclass(frozen=True)
class ApprovedTemplate:
    """An immutable, auditable approved template bound to content and config hashes."""

    template_id: str
    original_text: str
    binding: TemplateBinding
    decisions: tuple[OccurrenceReview, ...]
    composed_text: str
    tokens_before: int
    tokens_after: int
    tokens_saved: int
    review_status: str  # "approved" | "rejected" | "partial" | "pending"
    reviewer: str
    created_at: str
    updated_at: str
    metadata: dict[str, Any] = field(default_factory=dict)
    disclaimer: str = WORKFLOW_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": APPROVAL_SCHEMA_VERSION,
            "template_id": self.template_id,
            "original_text": self.original_text,
            "binding": self.binding.to_dict(),
            "decisions": [d.to_dict() for d in self.decisions],
            "composed_text": self.composed_text,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "tokens_saved": self.tokens_saved,
            "review_status": self.review_status,
            "reviewer": self.reviewer,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
            "disclaimer": self.disclaimer,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ApprovedTemplate:
        binding = TemplateBinding.from_dict(data["binding"])
        decisions = tuple(OccurrenceReview.from_dict(d) for d in data["decisions"])
        return cls(
            template_id=str(data["template_id"]),
            original_text=str(data["original_text"]),
            binding=binding,
            decisions=decisions,
            composed_text=str(data["composed_text"]),
            tokens_before=int(data["tokens_before"]),
            tokens_after=int(data["tokens_after"]),
            tokens_saved=int(data["tokens_saved"]),
            review_status=str(data["review_status"]),
            reviewer=str(data.get("reviewer", "local-user")),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            metadata=dict(data.get("metadata", {})),
            disclaimer=str(data.get("disclaimer", WORKFLOW_DISCLAIMER)),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> ApprovedTemplate:
        return cls.from_dict(json.loads(s))

    def to_audit_dict(self) -> dict[str, Any]:
        """Export a comprehensive audit dictionary suitable for compliance and tracking."""
        accepted = [d for d in self.decisions if d.decision == "accepted"]
        rejected = [d for d in self.decisions if d.decision == "rejected"]
        pending = [d for d in self.decisions if d.decision == "pending"]
        return {
            "schema_version": APPROVAL_SCHEMA_VERSION,
            "workflow": "review_first_production_workflow",
            "production_quality_claim": False,
            "disclaimer": self.disclaimer,
            "template_id": self.template_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "reviewer": self.reviewer,
            "review_status": self.review_status,
            "binding": self.binding.to_dict(),
            "original_text": self.original_text,
            "composed_text": self.composed_text,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "tokens_saved": self.tokens_saved,
            "token_reduction_ratio": round(self.tokens_after / self.tokens_before, 4) if self.tokens_before else 1.0,
            "summary": {
                "total_occurrences": len(self.decisions),
                "accepted_count": len(accepted),
                "rejected_count": len(rejected),
                "pending_count": len(pending),
            },
            "occurrences": [d.to_dict() for d in self.decisions],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ValidationResult:
    """The result of validating and attempting to apply an approved template."""

    is_valid: bool
    status: str  # "valid" | "invalidated"
    text: str  # Unchanged input text if invalidated; composed text if valid
    invalidation_reason: str | None = None
    tokens_before: int | None = None
    tokens_after: int | None = None
    tokens_saved: int | None = None
    binding: TemplateBinding | None = None
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "status": self.status,
            "text": self.text,
            "invalidation_reason": self.invalidation_reason,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "tokens_saved": self.tokens_saved,
            "binding": self.binding.to_dict() if self.binding else None,
            "warnings": list(self.warnings),
        }


def compose_reviewed_text(original: str, decisions: Sequence[OccurrenceReview]) -> str:
    """Compose text applying ONLY accepted occurrences, validating boundaries and overlap."""
    accepted = [d for d in decisions if d.decision == "accepted"]
    if not accepted:
        return original

    # Sort accepted occurrences by span start
    sorted_accepted = sorted(accepted, key=lambda d: (d.span.start, d.span.end))

    parts = []
    cursor = 0
    for item in sorted_accepted:
        if not (0 <= item.span.start <= item.span.end <= len(original)):
            raise ValueError(
                f"Occurrence span ({item.span.start}, {item.span.end}) is out of bounds for text of length {len(original)}."
            )
        if item.span.start < cursor:
            raise ValueError(
                f"Overlapping occurrence spans detected at offset {item.span.start} (cursor at {cursor})."
            )
        actual_source = original[item.span.start : item.span.end]
        if actual_source != item.original:
            raise ValueError(
                f"Occurrence source mismatch at span [{item.span.start}:{item.span.end}]: "
                f"expected {item.original!r}, found {actual_source!r}."
            )
        parts.append(original[cursor : item.span.start])
        parts.append(item.replacement)
        cursor = item.span.end

    parts.append(original[cursor:])
    return "".join(parts)


def find_protected_conflicts(
    text: str,
    occurrences: Sequence[OccurrenceReview],
    explicit_protected_spans: Sequence[Span | tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    """Identify any accepted occurrences that overlap with protected regions."""
    accepted = [d for d in occurrences if d.decision == "accepted"]
    if not accepted:
        return []

    explicit = (
        normalize_protected_spans(text, explicit_protected_spans)
        if explicit_protected_spans
        else ()
    )

    yaml_terms, _ = load_defined_terms()
    defined = frozenset(set(yaml_terms) | extract_defined_terms(text) | additional_defined_terms(text))
    blocks = segment(text)
    frozen = list(frozen_ranges(text, blocks, explicit, defined))

    # Add numeric literals as protected spans
    for m in _NUMBER_RE.finditer(text):
        frozen.append(type("FrozenItem", (), {"span": Span(m.start(), m.end()), "reason": "numeric_literal"})())

    # Add negation words as protected spans
    for m in _NEGATION_RE.finditer(text):
        frozen.append(type("FrozenItem", (), {"span": Span(m.start(), m.end()), "reason": "negation"})())

    # Add defined terms as protected spans
    for term in defined:
        if term:
            pattern = re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)", re.IGNORECASE)
            for m in pattern.finditer(text):
                frozen.append(type("FrozenItem", (), {"span": Span(m.start(), m.end()), "reason": f"defined_term_{term}"})())

    protect_list = load_protect_list()
    low = text.lower()
    for words in protect_list.single_terms.values():
        for w in words:
            for m in re.finditer(r"\b" + re.escape(w) + r"\b", low):
                frozen.append(type("FrozenItem", (), {"span": Span(m.start(), m.end()), "reason": f"protected_term_{w}"})())
    for phrases in protect_list.phrase_terms.values():
        for p in phrases:
            if p:
                start = 0
                while True:
                    idx = low.find(p, start)
                    if idx == -1:
                        break
                    frozen.append(type("FrozenItem", (), {"span": Span(idx, idx + len(p)), "reason": f"protected_phrase_{p}"})())
                    start = idx + len(p)

    conflicts = []
    for occ in accepted:
        # Check against frozen ranges
        for item in frozen:
            if overlaps(occ.span, item.span):
                conflicts.append({
                    "occurrence": occ.to_dict(),
                    "protected_span": {"start": item.span.start, "end": item.span.end},
                    "reason": f"protected_{item.reason}",
                })
                break
    return conflicts


def preview_approval(
    original_text: str,
    decisions: Sequence[OccurrenceReview | dict[str, Any]],
    *,
    config_hash: str,
    tokenizer_name: str,
    vector_identity: str,
    counter: TokenCounter | None = None,
    explicit_protected_spans: Sequence[Span | tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Preview composed text and complete-text recount under explicit user decisions."""
    parsed_decisions = [
        d if isinstance(d, OccurrenceReview) else OccurrenceReview.from_dict(d)
        for d in decisions
    ]

    if counter is None:
        counter = local_counter(tokenizer_name)

    conflicts = find_protected_conflicts(original_text, parsed_decisions, explicit_protected_spans)
    if conflicts:
        reason = f"Accepted occurrence overlaps with protected span: {conflicts[0]['reason']}"
        tokens_orig = counter.count(original_text)
        return {
            "original_text": original_text,
            "composed_text": original_text,
            "tokens_before": tokens_orig,
            "tokens_after": tokens_orig,
            "saved_tokens": 0,
            "saved_percent": 0.0,
            "review_status": "invalidated",
            "is_valid": False,
            "invalidation_reason": reason,
            "decisions": [d.to_dict() for d in parsed_decisions],
            "binding": {
                "original_hash": compute_content_hash(original_text),
                "config_hash": config_hash,
                "tokenizer_name": tokenizer_name,
                "vector_identity": vector_identity,
            },
        }

    try:
        composed = compose_reviewed_text(original_text, parsed_decisions)
    except ValueError as exc:
        tokens_orig = counter.count(original_text)
        return {
            "original_text": original_text,
            "composed_text": original_text,
            "tokens_before": tokens_orig,
            "tokens_after": tokens_orig,
            "saved_tokens": 0,
            "saved_percent": 0.0,
            "review_status": "invalidated",
            "is_valid": False,
            "invalidation_reason": str(exc),
            "decisions": [d.to_dict() for d in parsed_decisions],
            "binding": {
                "original_hash": compute_content_hash(original_text),
                "config_hash": config_hash,
                "tokenizer_name": tokenizer_name,
                "vector_identity": vector_identity,
            },
        }

    tokens_before = counter.count(original_text)
    tokens_after = counter.count(composed)
    saved = tokens_before - tokens_after
    saved_percent = round(100 * saved / tokens_before, 1) if tokens_before > 0 else 0.0

    accepted_count = sum(1 for d in parsed_decisions if d.decision == "accepted")
    rejected_count = sum(1 for d in parsed_decisions if d.decision == "rejected")
    pending_count = sum(1 for d in parsed_decisions if d.decision == "pending")

    if not parsed_decisions:
        review_status = "pending"
    elif pending_count > 0:
        review_status = "pending"
    elif accepted_count > 0 and saved > 0:
        review_status = "approved"
    else:
        review_status = "rejected"

    return {
        "original_text": original_text,
        "composed_text": composed,
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "saved_tokens": saved,
        "saved_percent": saved_percent,
        "review_status": review_status,
        "is_valid": True,
        "invalidation_reason": None,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "pending_count": pending_count,
        "decisions": [d.to_dict() for d in parsed_decisions],
        "binding": {
            "original_hash": compute_content_hash(original_text),
            "config_hash": config_hash,
            "tokenizer_name": tokenizer_name,
            "vector_identity": vector_identity,
        },
    }


def create_approved_template(
    original_text: str,
    decisions: Sequence[OccurrenceReview | dict[str, Any]],
    *,
    config_hash: str,
    tokenizer_name: str,
    vector_identity: str,
    counter: TokenCounter | None = None,
    reviewer: str = "local-user",
    template_id: str | None = None,
    explicit_protected_spans: Sequence[Span | tuple[int, int]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ApprovedTemplate:
    """Create an immutable ApprovedTemplate bound to content, config, and tokenizer identities."""
    parsed_decisions = tuple(
        d if isinstance(d, OccurrenceReview) else OccurrenceReview.from_dict(d)
        for d in decisions
    )

    if counter is None:
        counter = local_counter(tokenizer_name)

    conflicts = find_protected_conflicts(original_text, parsed_decisions, explicit_protected_spans)
    if conflicts:
        raise ValueError(
            f"Cannot approve template: accepted occurrence conflicts with protected span: {conflicts[0]['reason']}"
        )

    composed = compose_reviewed_text(original_text, parsed_decisions)
    tokens_before = counter.count(original_text)
    tokens_after = counter.count(composed)
    saved = tokens_before - tokens_after

    accepted_count = sum(1 for d in parsed_decisions if d.decision == "accepted")
    pending_count = sum(1 for d in parsed_decisions if d.decision == "pending")

    # Fail-closed invariant 1: every decision must be explicitly accepted or rejected
    if pending_count > 0:
        raise ValueError(
            f"Cannot approve template: {pending_count} occurrence(s) remain pending review. "
            "All candidate decisions must be explicitly accepted or rejected before saving."
        )

    # Fail-closed invariant 2: at least one accepted occurrence
    if accepted_count == 0:
        raise ValueError(
            "Cannot approve template: zero occurrences were accepted. "
            "An approved compression template requires at least one accepted substitution."
        )

    # Fail-closed invariant 3: positive full-text token reduction
    if saved <= 0:
        raise ValueError(
            f"Cannot approve template: token savings must be positive (saved {saved} tokens). "
            "An approved compression template requires measured token reduction."
        )

    review_status = "approved"

    original_hash = compute_content_hash(original_text)
    binding = TemplateBinding(
        original_hash=original_hash,
        config_hash=config_hash,
        tokenizer_name=tokenizer_name,
        vector_identity=vector_identity,
    )

    now = datetime.now(timezone.utc).isoformat()
    if not template_id:
        template_id = f"tpl-{original_hash[:12]}-{int(datetime.now(timezone.utc).timestamp())}"

    return ApprovedTemplate(
        template_id=template_id,
        original_text=original_text,
        binding=binding,
        decisions=parsed_decisions,
        composed_text=composed,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        tokens_saved=saved,
        review_status=review_status,
        reviewer=reviewer,
        created_at=now,
        updated_at=now,
        metadata=dict(metadata or {}),
        disclaimer=WORKFLOW_DISCLAIMER,
    )


def validate_and_apply(
    template: ApprovedTemplate | dict[str, Any],
    text: str,
    *,
    config_hash: str,
    tokenizer_name: str,
    vector_identity: str,
    counter: TokenCounter | None = None,
    explicit_protected_spans: Sequence[Span | tuple[int, int]] | None = None,
) -> ValidationResult:
    """Validate template against text, config, tokenizer, and vector identity.

    CRITICAL CONTRACT: Any mismatch or violation returns unchanged `text` and
    a clear invalidation reason, never an altered prompt or false quality guarantee.
    """
    if isinstance(template, dict):
        template = ApprovedTemplate.from_dict(template)

    # 0. Strict review status check: fail closed on any non-approved or tampered status
    if template.review_status != "approved":
        return ValidationResult(
            is_valid=False,
            status="invalidated",
            text=text,
            invalidation_reason=(
                f"Invalid template review status: '{template.review_status}'. "
                "Only explicitly approved templates (review_status='approved') may be applied."
            ),
        )

    # Invariant: positive token reduction required
    if template.tokens_saved <= 0:
        return ValidationResult(
            is_valid=False,
            status="invalidated",
            text=text,
            invalidation_reason=f"Approved template has non-positive token savings ({template.tokens_saved}).",
        )

    if template.tokens_saved != template.tokens_before - template.tokens_after:
        return ValidationResult(
            is_valid=False,
            status="invalidated",
            text=text,
            invalidation_reason="Template token savings do not match the before/after token counts.",
        )

    # Invariant: all decisions must be accepted or rejected, no pending decisions allowed
    if any(d.decision not in {"accepted", "rejected"} or d.decision == "pending" for d in template.decisions):
        return ValidationResult(
            is_valid=False,
            status="invalidated",
            text=text,
            invalidation_reason="Template contains unapproved or pending occurrence decisions.",
        )

    # Invariant: at least one accepted occurrence
    if not any(d.decision == "accepted" for d in template.decisions):
        return ValidationResult(
            is_valid=False,
            status="invalidated",
            text=text,
            invalidation_reason="Approved template must contain at least one accepted substitution.",
        )

    # 1. Exact original text & content hash binding
    current_hash = compute_content_hash(text)
    if current_hash != template.binding.original_hash:
        return ValidationResult(
            is_valid=False,
            status="invalidated",
            text=text,
            invalidation_reason=(
                f"Content hash mismatch: input text hash ({current_hash[:12]}...) "
                f"does not match bound hash ({template.binding.original_hash[:12]}...)."
            ),
        )

    if text != template.original_text:
        return ValidationResult(
            is_valid=False,
            status="invalidated",
            text=text,
            invalidation_reason="Exact original text mismatch despite matching hash length.",
        )

    # 2. Config hash binding
    if config_hash != template.binding.config_hash:
        return ValidationResult(
            is_valid=False,
            status="invalidated",
            text=text,
            invalidation_reason=(
                f"Configuration hash mismatch: active configuration ({config_hash[:12]}...) "
                f"does not match template configuration ({template.binding.config_hash[:12]}...)."
            ),
        )

    # 3. Tokenizer identity
    if tokenizer_name != template.binding.tokenizer_name and (
        tokenizer_name.removeprefix("tiktoken:") != template.binding.tokenizer_name.removeprefix("tiktoken:")
    ):
        return ValidationResult(
            is_valid=False,
            status="invalidated",
            text=text,
            invalidation_reason=(
                f"Tokenizer mismatch: active tokenizer '{tokenizer_name}' "
                f"does not match template tokenizer '{template.binding.tokenizer_name}'."
            ),
        )

    # 4. Vector identity
    if vector_identity != template.binding.vector_identity:
        return ValidationResult(
            is_valid=False,
            status="invalidated",
            text=text,
            invalidation_reason=(
                f"Vector identity mismatch: active vectors '{vector_identity}' "
                f"do not match template vectors '{template.binding.vector_identity}'."
            ),
        )

    # 5. Protected spans check
    conflicts = find_protected_conflicts(text, template.decisions, explicit_protected_spans)
    if conflicts:
        return ValidationResult(
            is_valid=False,
            status="invalidated",
            text=text,
            invalidation_reason=f"Protected span conflict: {conflicts[0]['reason']}.",
        )

    # 6. Occurrence spans, source text and compose check
    try:
        recomposed = compose_reviewed_text(text, template.decisions)
    except ValueError as exc:
        return ValidationResult(
            is_valid=False,
            status="invalidated",
            text=text,
            invalidation_reason=f"Occurrence span validation error: {exc}",
        )

    if recomposed != template.composed_text:
        return ValidationResult(
            is_valid=False,
            status="invalidated",
            text=text,
            invalidation_reason="Composed text mismatch between template record and recomposed output.",
        )

    # 7. Full-text token count recount check
    if counter is not None:
        actual_before = counter.count(text)
        actual_after = counter.count(recomposed)
        if actual_before != template.tokens_before or actual_after != template.tokens_after:
            return ValidationResult(
                is_valid=False,
                status="invalidated",
                text=text,
                invalidation_reason=(
                    f"Full-text token recount mismatch: actual count is {actual_before}->{actual_after}, "
                    f"but template record specifies {template.tokens_before}->{template.tokens_after}."
                ),
            )

    return ValidationResult(
        is_valid=True,
        status="valid",
        text=template.composed_text,
        invalidation_reason=None,
        tokens_before=template.tokens_before,
        tokens_after=template.tokens_after,
        tokens_saved=template.tokens_saved,
        binding=template.binding,
    )


class TemplateStore:
    """Local repository for approved prompt templates."""

    def __init__(self, root_dir: str | Path | None = None):
        if root_dir is None:
            root_dir = Path(".cache") / "approved_templates"
        self.root_dir = Path(root_dir)

    def save(self, template: ApprovedTemplate, *, overwrite: bool = False) -> Path:
        """Persist an approved template only via explicit user invocation.

        Approval records are immutable by default: collisions raise FileExistsError.
        """
        self.root_dir.mkdir(parents=True, exist_ok=True)
        # Sanitize filename
        safe_id = re.sub(r"[^\w.-]", "_", template.template_id)
        target = self.root_dir / f"{safe_id}.json"
        if overwrite:
            raise ValueError("Approval records are immutable and cannot be overwritten.")
        temp = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.root_dir,
                                             prefix=f".{safe_id}.", suffix=".tmp", delete=False) as handle:
                temp = Path(handle.name)
                handle.write(template.to_json())
                handle.flush()
                os.fsync(handle.fileno())
            # A hard link publishes the complete file atomically and fails if another
            # request already published the same ID.
            try:
                os.link(temp, target)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"Approved template '{template.template_id}' already exists. "
                    "Approval records are immutable; collisions are rejected."
                ) from exc
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)
        return target

    def get(self, template_id: str) -> ApprovedTemplate | None:
        """Retrieve a persisted template by template ID."""
        safe_id = re.sub(r"[^\w.-]", "_", template_id.removesuffix(".json"))
        target = self.root_dir / f"{safe_id}.json"
        if not target.is_file():
            return None
        try:
            return ApprovedTemplate.from_json(target.read_text(encoding="utf-8"))
        except Exception:
            return None

    def list_all(self) -> list[ApprovedTemplate]:
        """List all locally saved templates sorted by creation timestamp descending."""
        if not self.root_dir.is_dir():
            return []
        templates = []
        for path in sorted(self.root_dir.glob("*.json")):
            try:
                templates.append(ApprovedTemplate.from_json(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        templates.sort(key=lambda t: t.created_at, reverse=True)
        return templates

    def delete(self, template_id: str) -> bool:
        """Delete a persisted template file."""
        safe_id = re.sub(r"[^\w.-]", "_", template_id.removesuffix(".json"))
        target = self.root_dir / f"{safe_id}.json"
        if target.is_file():
            target.unlink()
            return True
        return False
