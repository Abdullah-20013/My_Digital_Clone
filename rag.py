"""Hybrid RAG core: chunking, BM25, dense retrieval and Reciprocal Rank Fusion.

Shared by app.py (the Streamlit app), build_index.py (precomputes embeddings.npy)
and eval.py (the report's test questions and hit@1 / hit@3 / MRR).
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
EMB_PATH = ROOT / "embeddings.npy"
META_PATH = ROOT / "chunks.json"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TARGET_WORDS = 220   # a chunk closes once it reaches this many words ...
MAX_WORDS = 300      # ... and never grows past this
OVERLAP_WORDS = 50   # trailing words carried into the next chunk
MIN_WORDS = 200      # a shorter last chunk is right-aligned up to this size
RRF_K = 60

# ---------------------------------------------------------------- chunking


@dataclass
class Chunk:
    chunk_id: str   # e.g. "personal_faq#2"
    source: str     # file name, e.g. "personal_faq.txt"
    title: str      # the document's "# " heading
    text: str

    @property
    def n_words(self) -> int:
        return len(self.text.split())


def _paragraphs(raw: str) -> tuple[str, list[str]]:
    """Split a document into paragraphs; '## ' headings are glued to the
    paragraph that follows them so a heading never ends up alone."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", raw) if b.strip()]
    title = ""
    if blocks and blocks[0].startswith("# "):
        title = blocks.pop(0)[2:].strip()
    paras, pending_heading = [], ""
    for b in blocks:
        if b.startswith("#") and len(b.splitlines()) == 1:
            pending_heading = b.lstrip("#").strip()
            continue
        if pending_heading:
            b = f"{pending_heading}: {b}"
            pending_heading = ""
        paras.append(b)
    return title, paras


def _split_long(para: str) -> list[str]:
    """Break a paragraph longer than MAX_WORDS on sentence boundaries."""
    if len(para.split()) <= MAX_WORDS:
        return [para]
    sentences = re.split(r"(?<=[.!?])\s+", para)
    out, cur = [], []
    for s in sentences:
        if cur and len(" ".join(cur + [s]).split()) > MAX_WORDS:
            out.append(" ".join(cur))
            cur = []
        cur.append(s)
    if cur:
        out.append(" ".join(cur))
    return out


def chunk_document(path: Path) -> list[Chunk]:
    """Pack whole paragraphs into ~200-300 word chunks. Each new chunk starts
    with the last paragraph(s) of the previous one (up to OVERLAP_WORDS) so a
    fact sitting on a boundary is retrievable from both sides."""
    title, paras = _paragraphs(path.read_text(encoding="utf-8"))
    units = [u for p in paras for u in _split_long(p)]

    groups: list[list[str]] = []
    cur: list[str] = []
    fresh = 0  # words in `cur` that are not overlap from the previous chunk
    for u in units:
        if cur and fresh and len(" ".join(cur + [u]).split()) > MAX_WORDS:
            groups.append(cur)
            carry, n = [], 0
            for prev in reversed(cur):
                n += len(prev.split())
                if n > OVERLAP_WORDS:
                    break
                carry.insert(0, prev)
            cur, fresh = carry, 0
        cur.append(u)
        fresh += len(u.split())
        if len(" ".join(cur).split()) >= TARGET_WORDS:
            groups.append(cur)
            carry, n = [], 0
            for prev in reversed(cur):
                n += len(prev.split())
                if n > OVERLAP_WORDS:
                    break
                carry.insert(0, prev)
            cur, fresh = carry, 0
    if cur and fresh:
        groups.append(cur)

    # A short tail chunk is a weak retrieval target. Right-align it instead:
    # walk back from the document's end until it reaches MIN_WORDS
    # (a larger overlap with the previous chunk, never past MAX_WORDS).
    if len(groups) > 1 and len(" ".join(groups[-1]).split()) < MIN_WORDS:
        tail: list[str] = []
        for u in reversed(units):
            if tail and len(" ".join([u] + tail).split()) > MAX_WORDS:
                break
            tail.insert(0, u)
            if len(" ".join(tail).split()) >= MIN_WORDS:
                break
        groups[-1] = tail

    return [
        Chunk(f"{path.stem}#{i}", path.name, title, "\n\n".join(g))
        for i, g in enumerate(groups)
    ]


def load_chunks(data_dir: Path = DATA_DIR) -> list[Chunk]:
    files = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in {".txt", ".md"})
    return [c for f in files for c in chunk_document(f)]


def corpus_fingerprint(chunks: list[Chunk]) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c.chunk_id.encode() + b"\0" + c.text.encode() + b"\0")
    return h.hexdigest()[:16]


# -------------------------------------------------------------- tokenising

STOPWORDS = set(
    """a about above after again all am an and any are as at be because been
    before being below between both but by can could did do does doing down
    during each few for from further had has have having he her here hers him
    his how i if in into is it its itself just me more most my myself no nor
    not now of off on once only or other our ours out over own same she should
    so some such than that the their theirs them then there these they this
    those through to too under until up very was we were what when where which
    while who whom why will with would you your yours yourself tell know
    really thing things""".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


# --------------------------------------------------------------- retrieval


def _embed_text(c: Chunk) -> str:
    # The title gives the encoder document-level context for short chunks.
    return f"{c.title}\n{c.text}" if c.title else c.text


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, encoder=None):
        assert len(chunks) == len(embeddings), "embeddings.npy is out of date"
        self.chunks = chunks
        self.emb = embeddings.astype(np.float32)
        self.encoder = encoder
        self.bm25 = BM25Okapi([tokenize(_embed_text(c)) for c in chunks])

    # Each scorer returns a score for every chunk; ranking happens in search().
    def bm25_scores(self, query: str) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(tokenize(query)), dtype=np.float32)

    def dense_scores(self, query: str, q_emb: np.ndarray | None = None) -> np.ndarray:
        """Cosine similarity of every chunk to the query. Pass `q_emb` (an
        L2-normalised query vector) when the encoder runs outside Python."""
        if q_emb is None:
            q_emb = self.encoder.encode([query], normalize_embeddings=True)[0]
        return self.emb @ np.asarray(q_emb, dtype=np.float32)  # both sides L2-normalised

    @staticmethod
    def _ranks(scores: np.ndarray) -> np.ndarray:
        """1-based rank of every chunk (1 = best)."""
        order = np.argsort(-scores, kind="stable")
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(scores) + 1)
        return ranks

    def search(self, query: str, mode: str = "hybrid", k: int = 4, q_emb=None) -> list[dict]:
        bm = self.bm25_scores(query)
        de = self.dense_scores(query, q_emb)
        bm_rank, de_rank = self._ranks(bm), self._ranks(de)

        if mode == "bm25":
            final = bm
        elif mode == "dense":
            final = de
        elif mode == "hybrid":
            # Reciprocal Rank Fusion: score = sum over retrievers of 1 / (60 + rank).
            # Chunks with zero BM25 score share no query terms, so BM25 gives
            # them no vote instead of an arbitrary tie-broken rank.
            final = 1.0 / (RRF_K + de_rank)
            final = final + np.where(bm > 0, 1.0 / (RRF_K + bm_rank), 0.0)
        else:
            raise ValueError(f"unknown mode {mode!r}")

        order = np.argsort(-final, kind="stable")
        if mode == "bm25":
            # A zero BM25 score means "no query term in this chunk": not a match.
            order = order[final[order] > 0]
        order = order[:k]
        return [
            {
                "rank": r + 1,
                "chunk": self.chunks[i],
                "score": float(final[i]),
                "bm25": float(bm[i]),
                "bm25_rank": int(bm_rank[i]),
                "dense": float(de[i]),
                "dense_rank": int(de_rank[i]),
            }
            for r, i in enumerate(order)
        ]


# ------------------------------------------------------------- index files


def load_encoder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL, device="cpu")


def build_index(encoder=None) -> tuple[list[Chunk], np.ndarray]:
    chunks = load_chunks()
    encoder = encoder or load_encoder()
    emb = encoder.encode(
        [_embed_text(c) for c in chunks], normalize_embeddings=True, batch_size=32
    ).astype(np.float32)
    np.save(EMB_PATH, emb)
    META_PATH.write_text(
        json.dumps(
            {
                "model": EMBED_MODEL,
                "fingerprint": corpus_fingerprint(chunks),
                "chunks": [asdict(c) for c in chunks],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return chunks, emb


def load_static_retriever(root: Path = ROOT) -> HybridRetriever:
    """Browser path: no sentence-transformers, just the precomputed files."""
    meta = json.loads((root / META_PATH.name).read_text(encoding="utf-8"))
    chunks = [Chunk(**c) for c in meta["chunks"]]
    return HybridRetriever(chunks, np.load(root / EMB_PATH.name))


def load_retriever() -> HybridRetriever:
    """Load precomputed embeddings; rebuild only if data/ changed since."""
    encoder = load_encoder()
    chunks = load_chunks()
    emb = None
    if EMB_PATH.exists() and META_PATH.exists():
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        if meta.get("fingerprint") == corpus_fingerprint(chunks) and meta.get("model") == EMBED_MODEL:
            emb = np.load(EMB_PATH)
    if emb is None:
        print("embeddings.npy missing or stale - recomputing")
        chunks, emb = build_index(encoder)
    return HybridRetriever(chunks, emb, encoder)
