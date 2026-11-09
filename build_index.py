"""Run locally after editing anything in data/:  python build_index.py
Writes embeddings.npy and chunks.json so the Space never embeds at startup."""
from rag import build_index

chunks, emb = build_index()
print(f"{len(chunks)} chunks -> embeddings.npy {emb.shape}")
for c in chunks:
    print(f"  {c.chunk_id:32s} {c.n_words:4d} words")
