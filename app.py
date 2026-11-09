"""Abdullah Mughal's digital clone: hybrid RAG (BM25 + dense + RRF) in Streamlit.

Run locally:  streamlit run app.py
Deployed on Streamlit Community Cloud; the Gemini key comes from st.secrets
(the app's Secrets settings, or .streamlit/secrets.toml locally, which is gitignored).
"""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from clone import MODES, NAME, Gemini, respond
from rag import load_retriever

st.set_page_config(page_title=f"{NAME} · Digital Clone", page_icon="🤖", layout="wide")


@st.cache_resource(show_spinner="Loading my notes and the embedding model…")
def get_retriever():
    return load_retriever()  # loads the precomputed embeddings.npy


def secret(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
    except Exception:  # no secrets file at all
        value = None
    return value or os.environ.get(name) or None


@st.cache_resource
def get_llm():
    key = secret("GEMINI_API_KEY")
    return Gemini(key, secret("GEMINI_MODEL")) if key else None


retriever = get_retriever()
llm = get_llm()

EXAMPLES = [
    ("What is DataCrumbs?", "BM25 only"),
    ("Describe a nerve-wracking last-minute change of direction on a group task", "Dense only"),
    ("Tell me about Kollab", "Hybrid (RRF)"),
    ("What resolution do you play CS2 on?", "Hybrid (RRF)"),
    ("Where do you want to live and work after you finish your degree?", "Hybrid (RRF)"),
    ("What's your favourite programming language and why?", "Hybrid (RRF)"),
    ("Are you the real Abdullah?", "Hybrid (RRF)"),
]

# Example buttons write into the widgets' session state before they render.
if "question" not in st.session_state:
    st.session_state.question = ""
    st.session_state.mode = "Hybrid (RRF)"


def use_example(q: str, m: str):
    st.session_state.question = q
    st.session_state.mode = m
    st.session_state.run = True


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    mode_label = st.radio("Retrieval mode", list(MODES), key="mode")
    k = st.slider("Top-k chunks", 1, 6, 4)
    st.divider()
    st.subheader("Try an example")
    for q, m in EXAMPLES:
        st.button(q, key=f"ex-{q}", on_click=use_example, args=(q, m), width="stretch")
    st.divider()
    st.caption(
        "Hybrid score = Σ 1/(60 + rank) over the BM25 and dense rankings (Reciprocal Rank Fusion). "
        "Dense = all-MiniLM-L6-v2 cosine similarity. "
        + (f"LLM: Google Gemini ({llm.model})." if llm else "LLM: not configured (quoted answers).")
    )

# ------------------------------------------------------------------- main
st.title(f"🤖 {NAME} — Digital Clone")
st.warning(
    f"**This is an AI clone, not the real person.** It answers only from {NAME}'s own notes "
    "and will say so when it doesn't know.",
    icon="⚠️",
)

with st.form("ask"):
    st.text_area(
        "Your question", key="question", height=80,
        placeholder="e.g. What projects are you working on?",
    )
    submitted = st.form_submit_button("Ask", type="primary")

if submitted or st.session_state.pop("run", False):
    question = st.session_state.question.strip()
    if not question:
        st.info("Ask me something first 🙂")
        st.stop()

    with st.spinner("Searching my notes…"):
        reply, hits = respond(retriever, question, MODES[mode_label], k, llm)

    st.subheader("Answer")
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(reply)

    st.subheader(f"Retrieved chunks · {mode_label}")
    if hits:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Rank": h["rank"],
                        "Source": h["chunk"].source,
                        "Chunk ID": h["chunk"].chunk_id,
                        "Score (mode)": round(h["score"], 4),
                        "BM25 (rank)": f"{h['bm25']:.2f} (#{h['bm25_rank']})" if h["bm25"] > 0 else "0.00 (no match)",
                        "Dense cosine (rank)": f"{h['dense']:.3f} (#{h['dense_rank']})",
                    }
                    for h in hits
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        for h in hits:
            c = h["chunk"]
            with st.expander(f"#{h['rank']} · {c.chunk_id} · {c.source} — score {h['score']:.4f}"):
                st.caption(f"BM25 {h['bm25']:.2f} · cosine {h['dense']:.3f}")
                st.markdown(c.text)
    else:
        st.caption("No chunks matched in this mode.")
