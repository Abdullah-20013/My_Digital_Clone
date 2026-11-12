# Abdullah Mughal: Digital Clone

> ⚠️ **This is an AI clone, not the real person.** It answers only from documents I wrote about myself, and says "I don't have that in my notes." when the answer isn't there.

**Live app:** _paste your Streamlit Community Cloud link here_

A Streamlit chatbot that answers questions as me using a hybrid Retrieval-Augmented Generation pipeline. It is hosted on **Streamlit Community Cloud** (free). Hugging Face Spaces now require a paid plan for Gradio and Docker apps, and a free Static Space can't keep an API key secret.

1. **Chunk** (`rag.py`): each document is split on paragraph boundaries and packed into ~200–300 word chunks. Each new chunk repeats up to 50 words from the end of the previous one (overlap). A short last chunk is right-aligned so it reaches back to at least 200 words. Every chunk keeps its file name and an ID such as `personal_faq#1`.
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
| `data/` | My knowledge base (11 documents) |

---

# Report

## 1. Knowledge base and why

| Document | Type | What it covers |
|---|---|---|
| `bio_and_cv.txt` | Factual (CV) | 7th semester BS CS at UMT Sialkot, Fiverr freelancing, projects, skills, certificates, 3rd place at the hackathon |
| `projects_deep_dive.txt` | Long-form / projects | How Nemesis, Zenith, ERPify, the store POS and the Cisco campus network work, their stacks, hard parts and status |
| `vibe_coding_hackathon.txt` | Long-form story | The UMT Vibe Coding Hackathon: theme, the flood-response app, 24 hours, rate-limit panic, 3rd place |
| `long_form_project.txt` | Long-form write-up | Retrospective on the FocusSpace last-minute pivot |
| `freelance_and_ai_projects.txt` | Factual / projects | AI automation work (LangChain, n8n, RAG, FastMCP), ERPify, Kollab, Zenith, POS, Maxing Enterprises |
| `freelancing_journey.txt` | Experience | Starting on Fiverr, the football-edit workflow, a RAG chatbot order, the worst client, pricing tips |
| `how_i_work.txt` | Habits / process | Daily routine, vibe-coding tools and limits, exam study, learning, procrastination, favourite subjects |
| `academics_and_future_plans.txt` | Factual / plans | Database and forensics coursework, plans for a Master's in Europe and a career in Japan |
| `advice_and_values.txt` | Opinions / values | Advice for juniors, skills that matter in Pakistan, values, hot takes |
| `opinions_and_hobbies.txt` | Personality / opinions | Steam profile art, CS2 settings, anime, Punjabi music, football |
| `personal_faq.txt` | First-person FAQ (30 Qs) | Setup, games, music, plans, semester, Japanese level, achievements, weakness, dream job, hot takes |

11 documents, about **5,000 words** (roughly 10–12 pages). I chose them so they cover the questions people actually ask me: *what do you do* (CV, project deep dive, freelancing), *how do you work and think* (habits, advice, opinions, FAQ) and *tell me a story* (the hackathon and FocusSpace write-ups). They also give each retriever something to win on. There are many **rare exact terms** (DataCrumbs, Kollab, HospitalRBAC_DB, FTK Imager, OSPF, Werkstudent, Demoman) that only BM25 can match reliably, and plain **descriptive passages** (the pivot story, procrastination habits, future plans) that dense search can match from a paraphrase. The documents give **26 chunks** of 200–292 words. The only exception is `academics_and_future_plans`, a single 181-word chunk because the whole document is that short. No phone numbers, addresses, IDs, account details or names of family, friends or classmates are included.

## 2. Five test questions across the three modes (top-3, k = 3)

`✓` = chunk that actually contains the answer. Scores: BM25 score / cosine / RRF score.

| # | Question | BM25 result | Dense result | Hybrid (RRF) result | Best mode and why |
|---|---|---|---|---|---|
| Q1 | **Rare keyword:** "What is DataCrumbs?" | ✓ `bio_and_cv#0` (1.97), ✓ `bio_and_cv#1` (1.89), nothing else matches | ✗ `projects_deep_dive#0` (0.207), `projects_deep_dive#2`, `projects_deep_dive#3`. The answer is only **#15** | ✓ `bio_and_cv#1` (0.029), ✓ `bio_and_cv#0`, then dense's #1 | **BM25 / Hybrid.** "DataCrumbs" is a made-up brand name; MiniLM has no meaning for it, so its vector is near-random, while BM25's IDF weight on a word found in one document is huge. RRF keeps BM25's picks on top. |
| Q2 | **Paraphrase, zero shared words:** "Describe a nerve-wracking last-minute change of direction on a group task" | ✗ `how_i_work#0` (4.76), `personal_faq#5`, `freelance_and_ai_projects#0`: incidental words like "group" and "task". The answer is **#4** | `vibe_coding_hackathon#1` (0.307), `vibe_coding_hackathon#0`, ✓ `long_form_project#0` (0.238, **#3**) | ✗ `how_i_work#0` (0.032), `vibe_coding_hackathon#1`, `vibe_coding_hackathon#0`. The answer is **#4** | **Dense.** The answer says "pivot", "11th hour", "one day before"; the question says "change of direction", "last-minute". No word overlaps, so BM25 latched onto incidental words. Dense found the FocusSpace story; its top 2 are the hackathon panic story, a near miss in meaning. BM25's noise pushed hybrid's answer to #4. |
| Q3 | **Paraphrase of a habit:** "How do you deal with putting things off?" | `how_i_work#0` (2.85) only; the answer chunk has **no BM25 match** (it says "procrastination") | ✓ `how_i_work#1` (0.209, **#1**), `personal_faq#2`, `how_i_work#0` | `how_i_work#0` (0.032), ✓ `how_i_work#1` (**#2**), `personal_faq#2` | **Dense.** "Putting things off" and "procrastination" share no token, but MiniLM puts them close together. Hybrid keeps the answer in the top 2 because chunks with a zero BM25 score get no BM25 vote, so dense's #1 is not buried by BM25's noise. |
| Q4 | **Rare project name:** "Tell me about Kollab" | ✓ `freelance_and_ai_projects#0` (2.76), nothing else matches | `bio_and_cv#1` (0.225), `projects_deep_dive#1`, `academics_and_future_plans#0`. The answer is **#6** | ✓ `freelance_and_ai_projects#0` first (0.032) | **BM25 / Hybrid.** Dense spreads "tell me about [a project]" over every project-like chunk. BM25's single confident match on the rare name wins the fusion. |
| Q5 | **Opinion:** "Is AI going to make developers unemployed?" | ✓ `advice_and_values#0` (5.40), `freelancing_journey#0`, ✓ `advice_and_values#1` | ✓ `personal_faq#3` (0.399), `freelance_and_ai_projects#0`, ✓ `advice_and_values#1` | ✓ `personal_faq#3` (0.032), ✓ `advice_and_values#1`, `freelancing_journey#0` | **All three, Hybrid best.** Before the KB expansion this was a hybrid miss: the hot take sat in one crowded FAQ chunk. Now it is also in the advice document under a "Hot take on AI and software jobs" heading, so both retrievers rank it high and RRF puts two relevant chunks in the top 2. |

**Over a hand-labelled set of 38 questions** (`python eval.py`; a chunk counts as relevant if it contains the answer phrase):

| Mode | hit@1 | hit@3 | MRR |
|---|---|---|---|
| BM25 only | 0.79 | 0.89 | 0.86 |
| Dense only | 0.58 | 0.76 | 0.70 |
| **Hybrid (RRF)** | **0.82** | **0.92** | **0.88** |

With the bigger KB, hybrid is now best on all three metrics (35 of 38 at hit@3). BM25 is still strong because my notes are full of exact names, but when it misses it misses completely: for Q3 and the failure case below, the answer chunk has no BM25 match at all. Dense alone misses rare names (Q1, Q4) but rescues paraphrases (Q2, Q3). Hybrid inherits most of both, so it is the **default mode** in the app. Its three misses are Q2, the failure case below, and "What does the AI in ERPify do?", where the answer paragraph landed in a different chunk from the ERPify heading.

**Fusion choice.** I used RRF because it only needs ranks. BM25 scores (0–11 here) and cosines (−0.1–0.55) are on unrelated scales. The alternative is a weighted score `α·dense + (1−α)·bm25` after **min-max normalising** each to [0, 1]. That uses how confident each retriever is, not just the order, but α has to be tuned, and min-max is fragile with only 26 chunks: one outlier stretches the whole range.

## 3. Failure case

**Question:** "Who is your favourite singer?" (all three modes miss it)

**What happened:** the answer ("Karan Aujla is definitely one of my favorite artists") is in `personal_faq#1` and `opinions_and_hobbies#1`. BM25 gives both a score of **0** and ranks `how_i_work#1` first (3.06). Dense also puts `how_i_work#1` first, with a cosine of only 0.120; the FAQ answer is **#4** (0.038). Hybrid ranks the answer **#6**, so with the app's default k = 4 the LLM never sees it and replies "I don't have that in my notes", even though the answer *is* in my notes.

**Why:** three things combine, and one of them was caused by growing the KB.

1. **Vocabulary mismatch.** My notes say "favorite *artists*" (American spelling) and "music"; the question says "favourite *singer*" (British spelling). After tokenising there is no shared word, so BM25 cannot reach the answer.
2. **A new lexical distractor.** Before the expansion, the British spelling "favourite" appeared nowhere in my notes, so BM25 cast no votes and hybrid fell back to dense, which ranked the answer #1. The new documents use "favourite" for university subjects, football memories and TF2 classes. BM25 now gives confident scores to those chunks, RRF counts those votes, and the wrong chunks win.
3. **Weak, diluted dense signal.** Every cosine is ≤ 0.12, close to noise. The music answer is one line inside chunks that also cover anime, Steam art, football and forensics, so its embedding is mostly about other things.

## 4. Improvement I would make next

**Chunk the FAQ one Q&A per chunk, normalise spelling, and add a cross-encoder reranker.** FAQ entries are self-contained, so a "one question = one chunk" rule (with the question text in each chunk) removes the dilution above and lets retrieval match a user's question against *my* question wording. A small normalisation step in the tokenizer (British → American spellings such as favourite → favorite, plus a few synonyms like singer → artist) would let BM25 match the failure case. After that, re-scoring the hybrid top-10 with a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) would read the question and each chunk together, so it can tell "favourite subject" apart from "favourite music". All three can be checked with the same `eval.py` numbers. A weighted min-max fusion with α ≈ 0.4 is a cheaper alternative worth comparing.

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
