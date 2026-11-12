"""Tests for review-first lexical approval, template binding, and invalidation."""
from __future__ import annotations

import dataclasses
import hashlib
import http.client
import json
from threading import Thread

import pytest

from promptcomp.lexical.approval import (
    APPROVAL_SCHEMA_VERSION,
    OccurrenceReview,
    TemplateStore,
    WORKFLOW_DISCLAIMER,
    compose_reviewed_text,
    create_approved_template,
    preview_approval,
    validate_and_apply,
)
from promptcomp.lexical.tokenization import local_counter
from promptcomp.types import Span
import scripts.ui as ui_server


@pytest.fixture
def counter():
    return local_counter("o200k_base")


@pytest.fixture
def sample_text():
    return "The committee met in order to discuss the project and in order to finalize plans."


def test_default_review_pending_state(counter):
    text = "They worked in order to finish."
    span = Span(12, 23)
    occ = OccurrenceReview(span=span, original="in order to", replacement="to", decision="pending")
    assert occ.decision == "pending"

    # Preview allows pending occurrences and reports review_status='pending' with 0 savings
    preview = preview_approval(
        text,
        [occ],
        config_hash="cfg-123",
        tokenizer_name="o200k_base",
        vector_identity="synthetic",
        counter=counter,
    )
    assert preview["review_status"] == "pending"
    assert preview["composed_text"] == text
    assert preview["saved_tokens"] == 0
    assert preview["is_valid"] is True

    # Fail-closed invariant: create_approved_template MUST reject pending decisions
    with pytest.raises(ValueError, match="remain pending review"):
        create_approved_template(
            text,
            [occ],
            config_hash="cfg-123",
            tokenizer_name="o200k_base",
            vector_identity="synthetic",
            counter=counter,
        )


def test_explicit_approval_composes_text_and_recounts(sample_text, counter):
    # Spans for "in order to"
    span1 = Span(18, 29)
    span2 = Span(54, 65)
    assert sample_text[span1.start : span1.end] == "in order to"
    assert sample_text[span2.start : span2.end] == "in order to"

    occ1 = OccurrenceReview(span=span1, original="in order to", replacement="to", decision="accepted")
    occ2 = OccurrenceReview(span=span2, original="in order to", replacement="to", decision="accepted")

    template = create_approved_template(
        sample_text,
        [occ1, occ2],
        config_hash="test-config-hash-1",
        tokenizer_name="o200k_base",
        vector_identity="synthetic",
        counter=counter,
        reviewer="auditor-alice",
    )

    expected_composed = "The committee met to discuss the project and to finalize plans."
    assert template.composed_text == expected_composed
    assert template.review_status == "approved"
    assert template.tokens_before == counter.count(sample_text)
    assert template.tokens_after == counter.count(expected_composed)
    assert template.tokens_saved > 0
    assert template.tokens_saved == template.tokens_before - template.tokens_after
    assert template.reviewer == "auditor-alice"
    assert template.disclaimer == WORKFLOW_DISCLAIMER


def test_partial_review_applies_only_accepted_occurrences(sample_text, counter):
    span1 = Span(18, 29)
    span2 = Span(54, 65)

    # Accept first occurrence, explicitly reject second occurrence (no pending)
    occ1 = OccurrenceReview(span=span1, original="in order to", replacement="to", decision="accepted")
    occ2 = OccurrenceReview(span=span2, original="in order to", replacement="to", decision="rejected")

    template = create_approved_template(
        sample_text,
        [occ1, occ2],
        config_hash="cfg-partial",
        tokenizer_name="o200k_base",
        vector_identity="synthetic",
        counter=counter,
    )

    expected_composed = "The committee met to discuss the project and in order to finalize plans."
    assert template.composed_text == expected_composed
    assert template.review_status == "approved"
    assert template.tokens_after == counter.count(expected_composed)
    assert template.tokens_saved == template.tokens_before - template.tokens_after
    assert template.tokens_saved > 0


def test_pending_occurrences_reject_template_creation(sample_text, counter):
    span1 = Span(18, 29)
    span2 = Span(54, 65)

    # One accepted, one pending -> must be rejected
    occ1 = OccurrenceReview(span=span1, original="in order to", replacement="to", decision="accepted")
    occ2 = OccurrenceReview(span=span2, original="in order to", replacement="to", decision="pending")

    with pytest.raises(ValueError, match="remain pending review"):
        create_approved_template(
            sample_text,
            [occ1, occ2],
            config_hash="cfg-pending",
            tokenizer_name="o200k_base",
            vector_identity="synthetic",
            counter=counter,
        )


def test_full_rejection_cannot_create_approved_template(sample_text, counter):
    span1 = Span(18, 29)
    span2 = Span(54, 65)

    occ1 = OccurrenceReview(span=span1, original="in order to", replacement="to", decision="rejected")
    occ2 = OccurrenceReview(span=span2, original="in order to", replacement="to", decision="rejected")

    # Preview on all rejected returns review_status="rejected", 0 saved, unchanged text
    preview = preview_approval(
        sample_text,
        [occ1, occ2],
        config_hash="cfg-rejected",
        tokenizer_name="o200k_base",
        vector_identity="synthetic",
        counter=counter,
    )
    assert preview["review_status"] == "rejected"
    assert preview["composed_text"] == sample_text
    assert preview["saved_tokens"] == 0

    # Fail closed: saving an approved compression template requires at least one accepted edit
    with pytest.raises(ValueError, match="zero occurrences were accepted"):
        create_approved_template(
            sample_text,
            [occ1, occ2],
            config_hash="cfg-rejected",
            tokenizer_name="o200k_base",
            vector_identity="synthetic",
            counter=counter,
        )


def test_zero_or_negative_token_savings_rejected(counter):
    text = "The quick brown fox."
    # Suppose an occurrence replaces "quick" with "speedy" (no token saving)
    span = Span(4, 9)
    occ = OccurrenceReview(span=span, original="quick", replacement="speedy", decision="accepted")

    with pytest.raises(ValueError, match="token savings must be positive"):
        create_approved_template(
            text,
            [occ],
            config_hash="cfg-nosave",
            tokenizer_name="o200k_base",
            vector_identity="none",
            counter=counter,
        )


def test_validate_and_apply_rejects_non_approved_or_tampered_status(sample_text, counter):
    span = Span(18, 29)
    occ = OccurrenceReview(span=span, original="in order to", replacement="to", decision="accepted")
    template = create_approved_template(
        sample_text,
        [occ],
        config_hash="cfg-valid",
        tokenizer_name="o200k_base",
        vector_identity="synthetic",
        counter=counter,
    )

    for bad_status in ("partial", "rejected", "pending", "tampered", "invalid"):
        tampered = dataclasses.replace(template, review_status=bad_status)
        res = validate_and_apply(
            tampered,
            sample_text,
            config_hash="cfg-valid",
            tokenizer_name="o200k_base",
            vector_identity="synthetic",
            counter=counter,
        )
        assert not res.is_valid
        assert res.status == "invalidated"
        assert res.text == sample_text  # CRITICAL: fallback to unchanged input text!
        assert "Invalid template review status" in (res.invalidation_reason or "")


def test_stale_content_invalidation(sample_text, counter):
    span1 = Span(18, 29)
    occ1 = OccurrenceReview(span=span1, original="in order to", replacement="to", decision="accepted")
    template = create_approved_template(
        sample_text,
        [occ1],
        config_hash="cfg-valid",
        tokenizer_name="o200k_base",
        vector_identity="synthetic",
        counter=counter,
    )

    # Altering original text by 1 character
    tampered_text = sample_text.replace("plans", "goals")
    res = validate_and_apply(
        template,
        tampered_text,
        config_hash="cfg-valid",
        tokenizer_name="o200k_base",
        vector_identity="synthetic",
        counter=counter,
    )

    assert not res.is_valid
    assert res.status == "invalidated"
    assert res.text == tampered_text  # CRITICAL: returns unchanged input text
    assert "Content hash mismatch" in (res.invalidation_reason or "")


def test_stale_config_hash_invalidation(sample_text, counter):
    span1 = Span(18, 29)
    occ1 = OccurrenceReview(span=span1, original="in order to", replacement="to", decision="accepted")
    template = create_approved_template(
        sample_text,
        [occ1],
        config_hash="cfg-valid",
        tokenizer_name="o200k_base",
        vector_identity="synthetic",
        counter=counter,
    )

    # Passing modified or stale config hash
    res = validate_and_apply(
        template,
        sample_text,
        config_hash="cfg-modified",
        tokenizer_name="o200k_base",
        vector_identity="synthetic",
        counter=counter,
    )

    assert not res.is_valid
    assert res.status == "invalidated"
    assert res.text == sample_text  # Unchanged
    assert "Configuration hash mismatch" in (res.invalidation_reason or "")


def test_tokenizer_mismatch_invalidation(sample_text, counter):
    span1 = Span(18, 29)
    occ1 = OccurrenceReview(span=span1, original="in order to", replacement="to", decision="accepted")
    template = create_approved_template(
        sample_text,
        [occ1],
        config_hash="cfg-valid",
        tokenizer_name="o200k_base",
        vector_identity="synthetic",
        counter=counter,
    )

    # Validating against a different tokenizer identity
    res = validate_and_apply(
        template,
        sample_text,
        config_hash="cfg-valid",
        tokenizer_name="cl100k_base",
        vector_identity="synthetic",
        counter=counter,
    )

    assert not res.is_valid
    assert res.status == "invalidated"
    assert res.text == sample_text
    assert "Tokenizer mismatch" in (res.invalidation_reason or "")


def test_vector_identity_mismatch_invalidation(sample_text, counter):
    span1 = Span(18, 29)
    occ1 = OccurrenceReview(span=span1, original="in order to", replacement="to", decision="accepted")
    template = create_approved_template(
        sample_text,
        [occ1],
        config_hash="cfg-valid",
        tokenizer_name="o200k_base",
        vector_identity="synthetic",
        counter=counter,
    )

    # Validating against different vector identity
    res = validate_and_apply(
        template,
        sample_text,
        config_hash="cfg-valid",
        tokenizer_name="o200k_base",
        vector_identity="/data/glove.6B.100d.txt",
        counter=counter,
    )

    assert not res.is_valid
    assert res.status == "invalidated"
    assert res.text == sample_text
    assert "Vector identity mismatch" in (res.invalidation_reason or "")


def test_protected_spans_prevent_approval_and_application(counter):
    text = "The cost exceeds $5,000 for this order."
    # $5,000 is numeric / protected. Suppose an edit tries to rewrite '$5,000'
    dollar_start = text.index("$5,000")
    dollar_end = dollar_start + len("$5,000")
    dollar_span = Span(dollar_start, dollar_end)

    occ = OccurrenceReview(span=dollar_span, original="$5,000", replacement="5k", decision="accepted")

    # Creating approved template should raise ValueError due to conflict with protected region
    with pytest.raises(ValueError, match="conflicts with protected span"):
        create_approved_template(
            text,
            [occ],
            config_hash="cfg-num",
            tokenizer_name="o200k_base",
            vector_identity="none",
            counter=counter,
        )

    # Preview approval should report invalidation
    preview = preview_approval(
        text,
        [occ],
        config_hash="cfg-num",
        tokenizer_name="o200k_base",
        vector_identity="none",
        counter=counter,
    )
    assert not preview["is_valid"]
    assert preview["review_status"] == "invalidated"
    assert "protected" in preview["invalidation_reason"].lower()
    assert preview["composed_text"] == text


def test_overlapping_occurrence_spans_invalidation(counter):
    text = "A quick brown fox jumps over the lazy dog."
    # Overlapping spans: [2:13] ("quick brown") and [8:17] ("brown fox")
    occ1 = OccurrenceReview(span=Span(2, 13), original="quick brown", replacement="fast brown", decision="accepted")
    occ2 = OccurrenceReview(span=Span(8, 17), original="brown fox", replacement="brown dog", decision="accepted")

    with pytest.raises(ValueError, match="Overlapping occurrence spans"):
        create_approved_template(
            text,
            [occ1, occ2],
            config_hash="cfg-overlap",
            tokenizer_name="o200k_base",
            vector_identity="none",
            counter=counter,
        )


def test_tampered_or_stale_span_text_invalidation(sample_text, counter):
    # Span points to [0:3] ("The"), but claim original is "XYZ"
    occ = OccurrenceReview(span=Span(0, 3), original="XYZ", replacement="A", decision="accepted")
    with pytest.raises(ValueError, match="Occurrence source mismatch"):
        compose_reviewed_text(sample_text, [occ])


def test_token_recount_mismatch_invalidation(sample_text, counter):
    span = Span(18, 29)
    occ = OccurrenceReview(span=span, original="in order to", replacement="to", decision="accepted")
    template = create_approved_template(
        sample_text,
        [occ],
        config_hash="cfg-recount",
        tokenizer_name="o200k_base",
        vector_identity="synthetic",
        counter=counter,
    )

    # 1. Arithmetic mismatch between tokens_saved and before/after difference
    arithmetic_tampered = dataclasses.replace(template, tokens_after=template.tokens_after + 1)
    res_arith = validate_and_apply(
        arithmetic_tampered,
        sample_text,
        config_hash="cfg-recount",
        tokenizer_name="o200k_base",
        vector_identity="synthetic",
        counter=counter,
    )
    assert not res_arith.is_valid
    assert res_arith.status == "invalidated"
    assert "token savings do not match" in (res_arith.invalidation_reason or "").lower()

    # 2. Recount mismatch where internal counts disagree with actual full-text token counter
    recount_tampered = dataclasses.replace(
        template,
        tokens_before=template.tokens_before + 5,
        tokens_saved=template.tokens_saved + 5,
    )
    res_recount = validate_and_apply(
        recount_tampered,
        sample_text,
        config_hash="cfg-recount",
        tokenizer_name="o200k_base",
        vector_identity="synthetic",
        counter=counter,
    )
    assert not res_recount.is_valid
    assert res_recount.status == "invalidated"
    assert res_recount.text == sample_text
    assert "recount mismatch" in (res_recount.invalidation_reason or "").lower()


def test_template_store_persistence_and_collision_rejection(tmp_path, sample_text, counter):
    store = TemplateStore(tmp_path / "templates")
    span = Span(18, 29)
    occ = OccurrenceReview(span=span, original="in order to", replacement="to", decision="accepted")

    template = create_approved_template(
        sample_text,
        [occ],
        config_hash="cfg-store",
        tokenizer_name="o200k_base",
        vector_identity="synthetic",
        counter=counter,
        reviewer="auditor-bob",
    )

    # Creating template does NOT persist it automatically
    assert list(store.root_dir.glob("*.json")) == []

    # Explicit save persists it
    saved_path = store.save(template)
    assert saved_path.is_file()
    assert len(store.list_all()) == 1

    # Invariant: Approval records are immutable; collisions are rejected
    with pytest.raises(FileExistsError, match="Approval records are immutable"):
        store.save(template)

    # Retrieve and verify round-trip
    loaded = store.get(template.template_id)
    assert loaded is not None
    assert loaded.template_id == template.template_id
    assert loaded.original_text == template.original_text
    assert loaded.composed_text == template.composed_text
    assert loaded.binding.original_hash == template.binding.original_hash
    assert loaded.binding.config_hash == template.binding.config_hash
    assert loaded.tokens_saved == template.tokens_saved
    assert loaded.reviewer == "auditor-bob"

    # Delete
    assert store.delete(template.template_id)
    assert store.get(template.template_id) is None


def test_audit_export_contract_and_disclaimer(sample_text, counter):
    span1 = Span(18, 29)
    span2 = Span(54, 65)
    occ1 = OccurrenceReview(span=span1, original="in order to", replacement="to", decision="accepted")
    occ2 = OccurrenceReview(span=span2, original="in order to", replacement="to", decision="rejected")

    template = create_approved_template(
        sample_text,
        [occ1, occ2],
        config_hash="cfg-audit",
        tokenizer_name="o200k_base",
        vector_identity="synthetic",
        counter=counter,
        reviewer="auditor-eve",
    )

    audit = template.to_audit_dict()
    assert audit["schema_version"] == APPROVAL_SCHEMA_VERSION
    assert audit["workflow"] == "review_first_production_workflow"
    assert audit["production_quality_claim"] is False
    assert audit["disclaimer"] == WORKFLOW_DISCLAIMER
    assert audit["reviewer"] == "auditor-eve"
    assert audit["review_status"] == "approved"
    assert audit["summary"]["total_occurrences"] == 2
    assert audit["summary"]["accepted_count"] == 1
    assert audit["summary"]["rejected_count"] == 1
    assert audit["summary"]["pending_count"] == 0
    assert audit["binding"]["original_hash"] == hashlib.sha256(sample_text.encode("utf-8")).hexdigest()
    assert audit["binding"]["config_hash"] == "cfg-audit"


def test_ui_endpoints_preview_approve_validate_and_templates(tmp_path, monkeypatch):
    test_store = TemplateStore(tmp_path / "templates")
    monkeypatch.setattr(ui_server, "_TEMPLATE_STORE", test_store)

    server = ui_server.ThreadingHTTPServer(("127.0.0.1", 0), ui_server.WorkbenchHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    conn = http.client.HTTPConnection(*server.server_address, timeout=10)

    try:
        text = "They worked in order to finish the assignment."
        # Spans for "in order to": [12:23]
        decisions = [
            {
                "span": {"start": 12, "end": 23},
                "original": "in order to",
                "replacement": "to",
                "decision": "pending",
                "provenance": "curated",
            }
        ]

        # 1. Preview with pending decision (returns unchanged text, 0 saved)
        conn.request(
            "POST",
            "/api/semantic/preview",
            body=json.dumps({
                "text": text,
                "decisions": decisions,
                "config_hash": "cfg-ui-test",
                "encoding": "o200k_base",
                "vector_identity": "none",
            }),
            headers={"Content-Type": "application/json"},
        )
        res = conn.getresponse()
        assert res.status == 200
        data = json.loads(res.read())
        assert data["review_status"] == "pending"
        assert data["composed_text"] == text
        assert data["saved_tokens"] == 0

        # 2. Preview with accepted decision (returns composed text and savings)
        decisions[0]["decision"] = "accepted"
        conn.request(
            "POST",
            "/api/semantic/preview",
            body=json.dumps({
                "text": text,
                "decisions": decisions,
                "config_hash": "cfg-ui-test",
                "encoding": "o200k_base",
                "vector_identity": "none",
            }),
            headers={"Content-Type": "application/json"},
        )
        res = conn.getresponse()
        assert res.status == 200
        data = json.loads(res.read())
        assert data["review_status"] == "approved"
        assert data["composed_text"] == "They worked to finish the assignment."
        assert data["saved_tokens"] > 0

        # 3. Server verification: client-invented arbitrary substitution rejected
        bad_decisions = [
            {
                "span": {"start": 35, "end": 45},
                "original": "assignment",
                "replacement": "work",
                "decision": "accepted",
            }
        ]
        conn.request(
            "POST",
            "/api/semantic/approve",
            body=json.dumps({
                "text": text,
                "decisions": bad_decisions,
                "arm": "b0",
                "encoding": "o200k_base",
            }),
            headers={"Content-Type": "application/json"},
        )
        res = conn.getresponse()
        assert res.status == 400
        bad_err = json.loads(res.read())
        assert "Unauthorized substitution" in bad_err["error"]

        # 4. Explicit Approve & Persist with authentic server-verified candidate
        conn.request(
            "POST",
            "/api/semantic/approve",
            body=json.dumps({
                "text": text,
                "decisions": decisions,
                "arm": "b0",
                "encoding": "o200k_base",
                "reviewer": "ui-user",
                "template_id": "test-fixed-template-id",
            }),
            headers={"Content-Type": "application/json"},
        )
        res = conn.getresponse()
        assert res.status == 200
        approved_data = json.loads(res.read())
        assert approved_data["status"] == "saved"
        template = approved_data["template"]
        assert template["composed_text"] == "They worked to finish the assignment."
        assert template["review_status"] == "approved"
        template_id = template["template_id"]
        server_config_hash = template["binding"]["config_hash"]
        server_vector_identity = template["binding"]["vector_identity"]

        # 5. Collision rejection: trying to save duplicate template_id must fail
        conn.request(
            "POST",
            "/api/semantic/approve",
            body=json.dumps({
                "text": text,
                "decisions": decisions,
                "arm": "b0",
                "encoding": "o200k_base",
                "template_id": "test-fixed-template-id",
            }),
            headers={"Content-Type": "application/json"},
        )
        res = conn.getresponse()
        assert res.status == 400
        coll_err = json.loads(res.read())
        assert "already exists" in coll_err["error"]

        # 6. List templates
        conn.request("GET", "/api/semantic/templates")
        res = conn.getresponse()
        assert res.status == 200
        list_data = json.loads(res.read())
        assert any(t["template_id"] == template_id for t in list_data["templates"])

        # 7. Fail-closed: Validate without active configuration (hash-only replay) must fail
        conn.request(
            "POST",
            "/api/semantic/validate",
            body=json.dumps({
                "template_id": template_id,
                "text": text,
                "config_hash": server_config_hash,
                "encoding": "o200k_base",
                "vector_identity": server_vector_identity,
                # Omitting 'arm'
            }),
            headers={"Content-Type": "application/json"},
        )
        res = conn.getresponse()
        assert res.status == 400
        hash_only_err = json.loads(res.read())
        assert "Hash-only replay is rejected" in hash_only_err["error"]

        # 8. Validate valid template with active configuration recomputation
        conn.request(
            "POST",
            "/api/semantic/validate",
            body=json.dumps({
                "template_id": template_id,
                "text": text,
                "arm": "b0",
                "encoding": "o200k_base",
            }),
            headers={"Content-Type": "application/json"},
        )
        res = conn.getresponse()
        assert res.status == 200
        val_data = json.loads(res.read())
        assert val_data["is_valid"] is True
        assert val_data["status"] == "valid"
        assert val_data["text"] == "They worked to finish the assignment."

        # 9. Validate with modified content (stale content invalidation -> fallback to unchanged text)
        conn.request(
            "POST",
            "/api/semantic/validate",
            body=json.dumps({
                "template_id": template_id,
                "text": "They worked hard in order to finish the assignment.",
                "arm": "b0",
                "encoding": "o200k_base",
            }),
            headers={"Content-Type": "application/json"},
        )
        res = conn.getresponse()
        assert res.status == 200
        val_data = json.loads(res.read())
        assert val_data["is_valid"] is False
        assert val_data["status"] == "invalidated"
        assert val_data["text"] == "They worked hard in order to finish the assignment."
        assert (
            "no longer produces" in val_data["invalidation_reason"]
            or "Content hash mismatch" in val_data["invalidation_reason"]
        )

        # 10. Validate with fabricated template object not in store (must fail closed)
        conn.request(
            "POST",
            "/api/semantic/validate",
            body=json.dumps({
                "template_id": "nonexistent-template-id",
                "text": text,
                "arm": "b0",
                "encoding": "o200k_base",
            }),
            headers={"Content-Type": "application/json"},
        )
        res = conn.getresponse()
        assert res.status == 400
        not_found_err = json.loads(res.read())
        assert "not found in store" in not_found_err["error"]

        # 11. Validate with tampered template object where fields disagree with store
        tampered_obj = dict(template)
        tampered_obj["composed_text"] = "Hacked text"
        conn.request(
            "POST",
            "/api/semantic/validate",
            body=json.dumps({
                "template_id": template_id,
                "template": tampered_obj,
                "text": text,
                "arm": "b0",
                "encoding": "o200k_base",
            }),
            headers={"Content-Type": "application/json"},
        )
        res = conn.getresponse()
        assert res.status == 400
        tamper_obj_err = json.loads(res.read())
        assert (
            "Validation accepts a saved template_id and active semantic settings only" in tamper_obj_err["error"]
            or "does not match stored approved template" in tamper_obj_err["error"]
        )

    finally:
        conn.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
def test_approve_rejects_missing_candidate_decisions(tmp_path, monkeypatch):
    test_store = TemplateStore(tmp_path / "templates")
    monkeypatch.setattr(ui_server, "_TEMPLATE_STORE", test_store)

    server = ui_server.ThreadingHTTPServer(("127.0.0.1", 0), ui_server.WorkbenchHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    conn = http.client.HTTPConnection(*server.server_address, timeout=10)

    try:
        # Text with 2 sentences, each having an "in order to" suggestion
        text = "They worked in order to finish. We planned in order to succeed."
        # Call /api/compress to see server edits
        conn.request(
            "POST",
            "/api/compress",
            body=json.dumps({"text": text, "mode": "semantic", "arm": "b0", "encoding": "o200k_base"}),
            headers={"Content-Type": "application/json"},
        )
        res = conn.getresponse()
        assert res.status == 200
        comp_data = json.loads(res.read())
        edits = comp_data["occurrences"]
        assert len(edits) == 2

        # Provide decision for only the first edit (omitting the second edit)
        partial_decisions = [
            {
                "span": edits[0]["span"],
                "original": edits[0]["original"],
                "replacement": edits[0]["replacement"],
                "decision": "accepted",
            }
        ]
        conn.request(
            "POST",
            "/api/semantic/approve",
            body=json.dumps({
                "text": text,
                "decisions": partial_decisions,
                "arm": "b0",
                "encoding": "o200k_base",
            }),
            headers={"Content-Type": "application/json"},
        )
        res = conn.getresponse()
        assert res.status == 400
        err = json.loads(res.read())
        assert "needs exactly one approval decision; some are missing" in err["error"]

    finally:
        conn.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def test_approve_rejects_duplicate_candidate_decisions(tmp_path, monkeypatch):
    test_store = TemplateStore(tmp_path / "templates")
    monkeypatch.setattr(ui_server, "_TEMPLATE_STORE", test_store)

    server = ui_server.ThreadingHTTPServer(("127.0.0.1", 0), ui_server.WorkbenchHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    conn = http.client.HTTPConnection(*server.server_address, timeout=10)

    try:
        text = "They worked in order to finish the assignment."
        decisions = [
            {
                "span": {"start": 12, "end": 23},
                "original": "in order to",
                "replacement": "to",
                "decision": "accepted",
            },
            {
                "span": {"start": 12, "end": 23},
                "original": "in order to",
                "replacement": "to",
                "decision": "rejected",
            },
        ]
        conn.request(
            "POST",
            "/api/semantic/approve",
            body=json.dumps({
                "text": text,
                "decisions": decisions,
                "arm": "b0",
                "encoding": "o200k_base",
            }),
            headers={"Content-Type": "application/json"},
        )
        res = conn.getresponse()
        assert res.status == 400
        err = json.loads(res.read())
        assert "Duplicate approval decision" in err["error"]

    finally:
        conn.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def test_approve_canonicalizes_metadata_ignoring_forged_fields(tmp_path, monkeypatch):
    test_store = TemplateStore(tmp_path / "templates")
    monkeypatch.setattr(ui_server, "_TEMPLATE_STORE", test_store)

    server = ui_server.ThreadingHTTPServer(("127.0.0.1", 0), ui_server.WorkbenchHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    conn = http.client.HTTPConnection(*server.server_address, timeout=10)

    try:
        text = "They worked in order to finish the assignment."
        # Submitting fabricated provenance and altered metadata
        forged_decision = [
            {
                "span": {"start": 12, "end": 23},
                "original": "in order to",
                "replacement": "to",
                "decision": "accepted",
                "provenance": "forged_malicious_provenance",
                "similarity": 0.999999,
                "token_delta": 9999,
                "sentence_before": "fabricated before",
                "sentence_after": "fabricated after",
            }
        ]
        conn.request(
            "POST",
            "/api/semantic/approve",
            body=json.dumps({
                "text": text,
                "decisions": forged_decision,
                "arm": "b0",
                "encoding": "o200k_base",
                "template_id": "tpl-canonical-test",
            }),
            headers={"Content-Type": "application/json"},
        )
        res = conn.getresponse()
        assert res.status == 200
        data = json.loads(res.read())
        template = data["template"]
        saved_decision = template["decisions"][0]

        # Invariant: Server canonicalized metadata from real engine edit, ignoring client forgery
        assert saved_decision["provenance"] == "curated"
        assert saved_decision["provenance"] != "forged_malicious_provenance"
        assert saved_decision["token_delta"] != 9999
        assert saved_decision["sentence_before"] != "fabricated before"

    finally:
        conn.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def test_atomic_template_store_collision_prevention(tmp_path, sample_text, counter):
    store = TemplateStore(tmp_path / "templates")
    span = Span(18, 29)
    occ = OccurrenceReview(span=span, original="in order to", replacement="to", decision="accepted")

    template = create_approved_template(
        sample_text,
        [occ],
        config_hash="cfg-atomic",
        tokenizer_name="o200k_base",
        vector_identity="synthetic",
        counter=counter,
        template_id="tpl-atomic-race",
    )

    # First save succeeds
    path = store.save(template)
    assert path.is_file()

    # Second save attempt must raise FileExistsError atomically
    with pytest.raises(FileExistsError, match="Approval records are immutable"):
        store.save(template)


def test_approve_rejects_candidate_not_in_server_selected_edits(tmp_path, monkeypatch):
    test_store = TemplateStore(tmp_path / "templates")
    monkeypatch.setattr(ui_server, "_TEMPLATE_STORE", test_store)

    server = ui_server.ThreadingHTTPServer(("127.0.0.1", 0), ui_server.WorkbenchHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    conn = http.client.HTTPConnection(*server.server_address, timeout=10)

    try:
        text = "They worked in order to finish the assignment."
        # Attempt to approve an unselected / fabricated edit on another span
        unselected_decision = [
            {
                "span": {"start": 0, "end": 4},
                "original": "They",
                "replacement": "We",
                "decision": "accepted",
            }
        ]
        conn.request(
            "POST",
            "/api/semantic/approve",
            body=json.dumps({
                "text": text,
                "decisions": unselected_decision,
                "arm": "b0",
                "encoding": "o200k_base",
            }),
            headers={"Content-Type": "application/json"},
        )
        res = conn.getresponse()
        assert res.status == 400
        err = json.loads(res.read())
        assert "All approval decisions must correspond to server-selected edits" in err["error"]

    finally:
        conn.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def test_validate_invalidates_if_live_configuration_no_longer_produces_substitution(tmp_path, monkeypatch):
    test_store = TemplateStore(tmp_path / "templates")
    monkeypatch.setattr(ui_server, "_TEMPLATE_STORE", test_store)

    server = ui_server.ThreadingHTTPServer(("127.0.0.1", 0), ui_server.WorkbenchHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    conn = http.client.HTTPConnection(*server.server_address, timeout=10)

    try:
        text = "They worked in order to finish the assignment."
        decisions = [
            {
                "span": {"start": 12, "end": 23},
                "original": "in order to",
                "replacement": "to",
                "decision": "accepted",
            }
        ]
        # Approve and save template under B0
        conn.request(
            "POST",
            "/api/semantic/approve",
            body=json.dumps({
                "text": text,
                "decisions": decisions,
                "arm": "b0",
                "encoding": "o200k_base",
                "template_id": "tpl-live-config-test",
            }),
            headers={"Content-Type": "application/json"},
        )
        res = conn.getresponse()
        assert res.status == 200
        template_id = json.loads(res.read())["template"]["template_id"]

        # Validate with protected spans covering the edit span [12:23]
        # so the live engine protects the span and no longer produces the edit
        conn.request(
            "POST",
            "/api/semantic/validate",
            body=json.dumps({
                "template_id": template_id,
                "text": text,
                "arm": "b0",
                "encoding": "o200k_base",
                "protected_spans": [[10, 25]],
            }),
            headers={"Content-Type": "application/json"},
        )
        res = conn.getresponse()
        assert res.status == 200
        val_data = json.loads(res.read())
        # Invariant: Invalidated, and text returned is the unchanged input text!
        assert val_data["is_valid"] is False
        assert val_data["status"] == "invalidated"
        assert val_data["text"] == text
        assert "no longer produces accepted substitution" in val_data["invalidation_reason"]

    finally:
        conn.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
