"""Local Streamlit workbench for experimental lexical substitution proposals.

Run from the repository root with ``python -m streamlit run semantic_ui/app.py``.
No model, tokenizer, corpus, or vector asset is downloaded by this application.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from promptcomp.lexical import PilotProvider, VectorIndex, suggest
from promptcomp.lexical.tokenization import local_counter


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "lexical-demo" / "prompts.jsonl"
SYNTHETIC_VECTORS = ROOT / "data" / "lexical-demo" / "synthetic-vectors.txt"
SYNTHETIC_LABEL = "Bundled synthetic vectors (software fixture, not trained embeddings)"


def load_samples() -> dict[str, str]:
    """Read authored samples without hiding a missing or malformed fixture."""
    samples: dict[str, str] = {}
    with SAMPLES.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                record = json.loads(line)
                samples[f"{record['id']} · {record['group']}"] = record["text"]
    return samples


@st.cache_resource(show_spinner="Loading local tokenizer…")
def get_counter(encoding: str):
    return local_counter(encoding)


@st.cache_resource(show_spinner="Reading local vectors…")
def get_vectors(path: str, modified_ns: int, size: int):
    # File metadata is part of the cache key so an edited local file is reloaded.
    return VectorIndex.from_text(path)


def guidance(exc: Exception) -> str:
    message = str(exc)
    if "en_core_web_sm" in message:
        return "Install the spaCy parser locally: `python -m spacy download en_core_web_sm`."
    if "Tokenizer asset" in message or "tiktoken" in message:
        return ("Provision the selected tokenizer explicitly while online, then rerun: "
                "`python -c \"import tiktoken; tiktoken.get_encoding('o200k_base')\"` "
                "(replace the encoding if you selected cl100k_base).")
    if "wordnet" in message.lower():
        return "Provision NLTK WordNet locally or turn off the WordNet option. The app never downloads it."
    return "Check the local asset path and installed optional dependencies, then rerun."


def decision_rows(decisions, *, selected: bool) -> list[dict]:
    rows = []
    for item in decisions:
        if (item.decision == "selected") == selected:
            rows.append({
                "span": f"{item.span.start}:{item.span.end}",
                "original": item.original,
                "replacement": item.replacement,
                "source": item.provenance,
                "cosine": item.similarity,
                "token gain": item.token_delta,
                "reason": item.reason,
                "candidate rank": item.candidate_rank,
            })
    return rows


def main() -> None:
    st.set_page_config(page_title="Semantic compression workbench", layout="wide")
    st.title("Semantic compression workbench")
    st.caption("Experimental lexical suggestions · local assets · human review required")
    st.info(
        "Token savings, cosine similarity, and grammar guards do not establish meaning "
        "preservation. Review the complete proposed output before using it."
    )

    try:
        samples = load_samples()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        samples = {}
        st.warning(f"Sample inputs unavailable at `{SAMPLES}`: {exc}. Paste your own text below.")

    with st.sidebar:
        st.header("Local assets and settings")
        arm = st.selectbox("Experiment arm", ("B0", "B1", "B2"),
                           help="B0: curated rules/optional WordNet. B1: same baseline pool ranked by vectors. B2: baseline plus vector neighbors.")
        use_wordnet = st.checkbox("Include locally installed WordNet", value=False,
                                  help="Off means B0 is curated rules only; this is not the complete lexical baseline.")
        if not use_wordnet:
            st.caption("WordNet off: baseline candidates are curated rules only.")
        encoding = st.selectbox("Tokenizer", ("o200k_base", "cl100k_base"))
        max_candidates = st.number_input("Candidates per source", 2, 32, 8, 1)
        embedding_slots = st.number_input("B2 vector slots", 1, int(max_candidates) - 1,
                                          min(4, int(max_candidates) - 1), 1,
                                          help="Remaining B2 slots are reserved for baseline candidates; unused slots are not borrowed.")
        vector_path = st.text_input("Local vector text path", str(SYNTHETIC_VECTORS),
                                    help="GloVe text or word2vec text with optional rows/dimensions header. No downloads.")
        resolved_vector_path = Path(vector_path).expanduser().resolve() if vector_path.strip() else None
        is_synthetic = resolved_vector_path == SYNTHETIC_VECTORS.resolve()
        if is_synthetic:
            st.warning(SYNTHETIC_LABEL + ". These values cannot support semantic-quality claims.")
        else:
            st.caption("User-supplied local vectors. Record their source before interpreting results.")
        if arm != "B0" and (resolved_vector_path is None or not resolved_vector_path.is_file()):
            st.error("B1/B2 need an existing local vector text file. Enter its path or use the bundled synthetic fixture.")
        threshold_on = st.checkbox("Require minimum embedding similarity", value=False,
                                   help="Applies to B2 embedding-generated candidates only; cosine is a proxy, not a semantic guarantee.")
        threshold = st.slider("Minimum cosine similarity", -1.0, 1.0, 0.75, 0.01,
                              disabled=not threshold_on)
        adaptive = st.checkbox("Adaptive similarity threshold", value=False,
                               disabled=not threshold_on,
                               help="Adds 0.05 for a one-token gain and 0.05 for noun/verb proposals, capped at 1.")
        price = st.number_input("Input price (USD per 1M tokens)", min_value=0.0,
                                value=0.0, step=0.01, format="%.4f",
                                help="Your hypothetical input-token price; excludes output tokens and other charges.")

    sample_names = ["Custom text", *samples]
    sample = st.selectbox("Sample input", sample_names, key="sample_choice")
    if "draft_text" not in st.session_state:
        st.session_state.draft_text = samples.get(sample, "")
    if st.session_state.get("last_sample") != sample:
        st.session_state.draft_text = samples.get(sample, "")
        st.session_state.last_sample = sample
    source_text = st.text_area("Text to inspect", key="draft_text", height=180,
                               placeholder="Paste text here or choose an authored sample above.")
    run = st.button("Generate proposal", type="primary", disabled=not source_text.strip())
    if not run:
        return

    if arm != "B0" and (resolved_vector_path is None or not resolved_vector_path.is_file()):
        st.error("No local vectors found. Supply a valid file path to run B1 or B2.")
        return

    try:
        counter = get_counter(encoding)
        vectors = None
        if arm != "B0":
            assert resolved_vector_path is not None
            stat = resolved_vector_path.stat()
            vectors = get_vectors(str(resolved_vector_path), stat.st_mtime_ns, stat.st_size)
        provider = PilotProvider(
            arm=arm.casefold(), vectors=vectors, use_wordnet=use_wordnet,
            max_candidates=int(max_candidates), embedding_slots=int(embedding_slots),
        )
        with st.spinner("Checking proposals with the local parser…"):
            result = suggest(
                source_text, provider, counter=counter,
                min_embedding_similarity=threshold if threshold_on else None,
                adaptive_similarity=bool(threshold_on and adaptive),
            )
    except (OSError, RuntimeError, ImportError, ValueError) as exc:
        st.error(f"Could not generate a proposal: {exc}")
        st.info(guidance(exc))
        return

    saved = result.tokens_before - result.tokens_after
    before_cost = result.tokens_before * price / 1_000_000
    after_cost = result.tokens_after * price / 1_000_000
    st.subheader("Tokenizer counts and price estimate")
    cols = st.columns(4)
    cols[0].metric("Original tokens", result.tokens_before)
    cols[1].metric("Proposed tokens", result.tokens_after)
    cols[2].metric("Tokens saved", saved)
    cols[3].metric("Estimated input cost saved", f"${before_cost - after_cost:.8f}")
    chart = go.Figure(go.Bar(
        x=["Original", "Proposed"],
        y=[result.tokens_before, result.tokens_after],
        text=[result.tokens_before, result.tokens_after],
        textposition="outside",
        marker_color=["#64748b", "#0d9488"],
    ))
    chart.update_layout(
        title="Tokens before and after", yaxis_title=f"Tokens · {result.token_counter}",
        yaxis_range=[0, max(result.tokens_before, result.tokens_after, 1) * 1.2],
        showlegend=False, height=270, margin=dict(l=30, r=20, t=45, b=25),
    )
    st.plotly_chart(chart, use_container_width=True)
    st.caption(f"Tokenizer: `{result.token_counter}` · Hypothetical input cost: "
               f"${before_cost:.8f} → ${after_cost:.8f} at ${price:.4f}/1M tokens. "
               "This is an estimate, not a measured bill.")

    st.subheader("Proposed output")
    st.code(result.suggested, language=None, wrap_lines=True)
    st.caption("Review status: pending. The proposal has not been approved by a person.")
    if result.suggested == source_text:
        st.info("No token-saving edit passed the selected guards for this input.")
    for warning in result.warnings:
        st.warning(warning)

    selected_tab, rejected_tab, details_tab = st.tabs(
        (f"Selected edits ({len(result.edits)})",
         f"Rejected candidates ({sum(d.decision != 'selected' for d in result.decisions)})",
         "Run details")
    )
    with selected_tab:
        rows = decision_rows(result.decisions, selected=True)
        if rows:
            st.dataframe(rows, hide_index=True, use_container_width=True)
            for index, edit in enumerate(result.edits, 1):
                with st.expander(f"Edit {index}: {edit.original} → {edit.replacement}"):
                    st.write("Before:", edit.sentence_before)
                    st.write("After:", edit.sentence_after)
        else:
            st.write("No selected edits.")
    with rejected_tab:
        rows = decision_rows(result.decisions, selected=False)
        if rows:
            st.dataframe(rows, hide_index=True, use_container_width=True)
        else:
            st.write("No rejected candidates.")
    with details_tab:
        st.write(f"Arm: {arm} · Candidate source: {provider.metadata['pool']}")
        st.write(f"Vectors: {SYNTHETIC_LABEL if is_synthetic and vectors is not None else str(resolved_vector_path) if vectors is not None else 'none'}")
        st.write(f"Minimum embedding similarity: {threshold if threshold_on else 'off'} · Adaptive: {bool(threshold_on and adaptive)}")
        st.write(f"Configuration hash: `{result.config_hash}`")
        st.json({"provider": provider.metadata, "timings_ms": result.timings_ms,
                 "edits": [asdict(edit) for edit in result.edits]})


if __name__ == "__main__":
    main()
