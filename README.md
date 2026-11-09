# Abdullah Mughal: Digital Clone

> ⚠️ **This is an AI clone, not the real person.** It answers only from documents I wrote about myself, and says "I don't have that in my notes." when the answer isn't there.

**Live app:** _paste your Streamlit Community Cloud link here_

A Streamlit chatbot that answers questions as me using a hybrid Retrieval-Augmented Generation pipeline. It is hosted on **Streamlit Community Cloud** (free). Hugging Face Spaces now require a paid plan for Gradio and Docker apps, and a free Static Space can't keep an API key secret.

1. **Chunk** (`rag.py`): each document is split on paragraph boundaries and packed into ~200–300 word chunks. Each new chunk repeats up to 50 words from the end of the previous one (overlap). A short last chunk is right-aligned so it reaches back to ~130+ words. Every chunk keeps its file name and an ID such as `personal_faq#1`.
2. **Index twice**: BM25 (`rank_bm25.BM25Okapi`) over lower-cased, stop-word-filtered tokens, and `all-MiniLM-L6-v2` sentence embeddings (sentence-transformers). Embeddings are precomputed once by `build_index.py` and saved in `embeddings.npy`. `chunks.json` stores a fingerprint of `data/`, so the app only re-embeds if the documents change.
3. **Retrieve**: BM25 scores, or cosine similarity (a numpy dot product of L2-normalised vectors; no vector database).
4. **Fuse** with **Reciprocal Rank Fusion**: `score(d) = Σ 1 / (60 + rank_r(d))` over the BM25 and dense rankings. Chunks with a BM25 score of 0 share no word with the question, so they get no BM25 vote.
5. **Generate** (`clone.py`): the top-k chunks go to **Google Gemini** (`google-genai`, a Flash model with low thinking) with a first-person persona prompt. The prompt says to answer only from the context and to reply "I don't have that in my notes." otherwise. Two cheap guards run before the LLM: "are you real?"-type questions get a fixed AI-clone answer, and questions where every chunk has cosine < 0.10 and BM25 < 1.0 are declined without calling the LLM. If Gemini fails or hits its free-tier limit, the app quotes the best-matching sentences from the retrieved chunks instead of inventing anything.
6. **Show the work**: the UI shows the answer, a sidebar mode selector (Hybrid / BM25 only / Dense only), a top-k slider, example questions and a table of the retrieved chunks. The table has each chunk's source, chunk ID, fused score, BM25 score with its BM25 rank, and cosine with its dense rank, so you can see the fusion re-order things. The full text of each chunk is in an expander below the table.

**API key:** the Gemini key is never in the code or the repo. It is read with `st.secrets["GEMINI_API_KEY"]` from the app's **Secrets** setting on Streamlit Community Cloud (locally from `.streamlit/secrets.toml`, which is gitignored). An optional `GEMINI_MODEL` secret pins one model. Otherwise the app tries current Flash models in order and keeps the first one the key can use.

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI |
| `clone.py` | Persona prompt, guards, Gemini call, templated fallback |
| `rag.py` | Chunking, BM25, dense retrieval, RRF, index loading |
| `build_index.py` | Precomputes `embeddings.npy` + `chunks.json` (run after editing `data/`) |
| `eval.py` | Report questions and hit@1 / hit@3 / MRR for each mode |
| `requirements.txt` | Pinned packages (CPU-only PyTorch to fit Streamlit Cloud) |
| `.streamlit/secrets.toml.example` | Template for the secrets file |
| `data/` | My knowledge base (6 documents) |

---

# Report

## 1. Knowledge base and why

| Document | Type | What it covers |
|---|---|---|
| `bio_and_cv.txt` | Factual (CV) | Degree at UMT Sialkot, Fiverr freelancing, projects (Nemesis, Zenith, ERPify, CAN), skills, certificates |
| `freelance_and_ai_projects.txt` | Factual / projects | AI automation work (LangChain, n8n, RAG, FastMCP), ERPify, Kollab, Zenith, POS, Maxing Enterprises |
| `academics_and_future_plans.txt` | Factual / plans | Database and forensics coursework, plans for a Master's in Europe and a career in Japan |
| `opinions_and_hobbies.txt` | Personality / opinions | Steam profile art, CS2 settings, anime, Punjabi music, football |
| `long_form_project.txt` | Long-form write-up | Retrospective on the FocusSpace last-minute pivot |
| `personal_faq.txt` | First-person FAQ | Setup, games, music, plans, food, payments, trips, hot takes, in my own voice |

I chose these because together they cover the questions people actually ask me: *what do you do* (CV, projects), *what are you like* (opinions, FAQ) and *tell me a story* (the FocusSpace write-up). They also give each retriever something to win on. There are many **rare exact terms** (DataCrumbs, Kollab, HospitalRBAC_DB, FTK Imager, Werkstudent) that only BM25 can match reliably, and plain **descriptive passages** (the pivot story, future plans) that dense search can match from a paraphrase. The documents give **11 chunks** of 165–286 words. No phone numbers, addresses, IDs or account details are included.

## 2. Five test questions across the three modes (top-3, k = 3)

`✓` = chunk that actually contains the answer. Scores: BM25 score / cosine / RRF score.

| # | Question | BM25 result | Dense result | Hybrid (RRF) result | Best mode and why |
|---|---|---|---|---|---|
| Q1 | **Rare keyword:** "What is DataCrumbs?" | ✓ `bio_and_cv#0` (1.67), nothing else matches | ✗ `long_form_project#1` (0.16), `bio_and_cv#1`, `long_form_project#0`. The answer is only **#9** | ✓ `bio_and_cv#0` first (0.031), then dense's picks | **BM25 / Hybrid.** "DataCrumbs" is a made-up brand name; MiniLM has no meaning for it, so its vector is near-random, while BM25's IDF weight on a word that appears once is huge. RRF keeps BM25's #1. |
| Q2 | **Paraphrase, zero shared words:** "Describe a nerve-wracking last-minute change of direction on a group task" | ✗ `personal_faq#2` (2.94, matched "group" from the Lahore Fort trip and "group of classmates"), `freelance_and_ai_projects#0` ("task" from "task execution") | ✓ `long_form_project#1` (0.244), ✓ `long_form_project#0` (0.238), ✓ `personal_faq#1` (FocusSpace FAQ) | `personal_faq#2`, `freelance#0`, ✓ `long_form_project#1` (#3) | **Dense.** The answer says "pivot", "11th hour", "one day before"; the question says "change of direction", "last-minute". No word overlaps, so BM25 latched onto incidental words. Dense got all three relevant chunks. Hybrid still found the answer, but BM25's noise pushed it to #3. |
| Q3 | **Synonym + spelling:** "Who is your favourite singer?" | ✗ **nothing**: my notes say "favorite *artist*", the question says "favourite *singer*" | ✓ `personal_faq#1` (0.113), `personal_faq#2`, ✓ `opinions_and_hobbies#1` | Same as dense (BM25 casts no votes) | **Dense / Hybrid.** British vs American spelling plus singer vs artist means zero token overlap; embeddings treat both pairs as near-synonyms. Because zero-score chunks get no BM25 vote, hybrid falls back to dense instead of failing. |
| Q4 | **Rare project name:** "Tell me about Kollab" | ✓ `freelance_and_ai_projects#0` (1.88) | `bio_and_cv#1` (0.167), `personal_faq#0` (0.146), ✓ `freelance#0` (0.144, **#3**) | ✓ `freelance_and_ai_projects#0` first (0.032) | **Hybrid.** Dense ranked three generic "about me" chunks almost equally (0.14–0.17). BM25's single confident match plus dense's #3 gives the right chunk the highest fused score. |
| Q5 | **Opinion:** "Is AI going to make developers unemployed?" | ✓ `personal_faq#2` (2.68), `freelance#0`, `bio_and_cv#0` | ✗ `freelance_and_ai_projects#0` (0.356), `long_form#1`, `bio_and_cv#1`. The answer is only **#7** (0.171) | ✗ `freelance#0` (0.0325), `long_form#1`, `bio_and_cv#1`. The answer is **#4** (0.0313) | **BM25.** "AI" and "jobs" sit in the hot-take answer, so BM25 ranks it #1. Dense prefers chunks that are *about* AI and work but hold no opinion. RRF rewards chunks that are middling in both lists over one that is #1 in one and #7 in the other, so hybrid loses it (see the failure case). |

**Over a hand-labelled set of 23 questions** (`python eval.py`; a chunk counts as relevant if it contains the answer phrase):

| Mode | hit@1 | hit@3 | MRR |
|---|---|---|---|
| BM25 only | **0.87** | **0.91** | **0.90** |
| Dense only | 0.57 | 0.78 | 0.69 |
| **Hybrid (RRF)** | 0.78 | **0.91** | 0.86 |

BM25 scores best on these numbers because my notes are full of exact names and the eval questions often reuse them. But when BM25 misses, it misses completely: for Q2 and Q3 it returns nothing useful, so the clone would have nothing to answer from. Hybrid ties BM25 on hit@3 (21 of 23) and its misses are different: the answer is still at #4–#5, not absent. Dense alone misses rare names (Q1). Real visitors paraphrase far more than my test set does, so I kept **Hybrid as the default mode**, and the next step (section 4) targets its two misses.

**Fusion choice.** I used RRF because it only needs ranks. BM25 scores (0–9 here) and cosines (−0.1–0.4) are on unrelated scales. The alternative is a weighted score `α·dense + (1−α)·bm25` after **min-max normalising** each to [0, 1]. That uses how confident each retriever is, not just the order, but it needs α to be tuned. Min-max is also fragile with only 11 chunks: one outlier stretches the whole range.

## 3. Failure case

**Question:** "What do you think will happen to programming jobs because of AI?" (one of the two hybrid misses in the eval set; Q5 is the other)

**What happened:** my hot take ("people creating professional AI agents that end up taking other people's jobs XD") is in `personal_faq#2`, BM25 ranks it **#1** (4.79), but dense ranks it **#9** (cosine 0.155) and hybrid only **#5**. With the app's default k = 4 the LLM never sees it. All it gets is my CV and AI-automation project descriptions, so the best it can do is a weak answer about what I build, or "I don't have that in my notes", even though the opinion *is* in my notes.

**Why:** two things combine.

1. **Chunk dilution.** `personal_faq#2` is a 241-word chunk that holds five unrelated Q&As: classmates, Steam badges, Payoneer, the Lahore Fort trip, and the AI hot take. Its embedding is an average of all five topics, so the AI opinion is only about a fifth of the vector. `bio_and_cv#1` and `freelance_and_ai_projects#0` are *entirely* about AI and software work, so they win on cosine even though they contain no opinion. (When I replaced my classmates' names with a privacy note, this chunk gained another off-topic sentence and its dense rank dropped from #5 to #9.)
2. **RRF flattens confidence.** RRF only looks at ranks. A BM25 #1 + dense #9 gives 1/61 + 1/69 = 0.0309, which is less than a chunk ranked #3 in both (1/63 + 1/63 = 0.0317). BM25's score of 4.79 here is very confident, but RRF can't see scores, so one confident retriever is outvoted by two lukewarm ones.

## 4. Improvement I would make next

**Chunk the FAQ one Q&A per chunk and add a cross-encoder reranker.** FAQ entries are self-contained, so a "one question = one chunk" rule (with the question text included in each chunk) would remove the dilution above. Retrieval would then match a user's question against *my* question wording. After that, re-scoring the hybrid top-10 with a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) would read the question and the chunk together, so it can tell "about AI" apart from "my opinion on AI and jobs". Both can be checked with the same `eval.py` hit@3 numbers. A weighted min-max fusion with α ≈ 0.4 is a cheaper alternative worth comparing, because it keeps BM25's confidence that RRF throws away.

---

### Run locally

```bash
pip install -r requirements.txt
python build_index.py     # only after editing data/
python eval.py            # report numbers
copy .streamlit\secrets.toml.example .streamlit\secrets.toml   # then put your Gemini key in it
streamlit run app.py
```

### Deploy on Streamlit Community Cloud

1. Push this folder to a **public GitHub repo**. `.gitignore` keeps `.streamlit/secrets.toml` out of it.
2. At share.streamlit.io choose **Create app** → *Deploy a public app from GitHub*. Pick the repo, the branch and `app.py` as the main file.
3. Open **Advanced settings**: choose **Python 3.12**, and in **Secrets** paste `GEMINI_API_KEY = "your-key"`.
4. Deploy. The first build installs PyTorch (CPU) and takes a few minutes.
