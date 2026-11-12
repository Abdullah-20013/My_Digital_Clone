"""Retrieval evaluation for the README report.

  python eval.py           -> top-3 per mode for the 5 report questions + hit@3 table

Labels are the chunk(s) that actually contain the answer, chosen by hand.
"""
from rag import load_retriever

MODES = ["bm25", "dense", "hybrid"]

# (question, answer phrase). A chunk counts as relevant ("gold") when it
# contains the phrase (or any of several phrases separated by "|"), so
# overlapping chunks are labelled automatically.
REPORT_QUESTIONS = [
    # Q1 exact rare keyword: BM25 should win, dense has never seen the word
    ("What is DataCrumbs?", "DataCrumbs"),
    # Q2 paraphrase sharing NO word with the answer chunks: dense should win
    ("Describe a nerve-wracking last-minute change of direction on a group task", "pivot"),
    # Q3 synonym + British spelling ("favourite singer" vs "favorite artist")
    ("Who is your favourite singer?", "Karan Aujla"),
    # Q4 rare project name where dense is weak but not lost
    ("Tell me about Kollab", "Kollab"),
    # Q5 opinion question, partly paraphrased
    ("Is AI going to make developers unemployed?", "taking other people's jobs"),
]

EXTRA_QUESTIONS = [
    ("Which payment gateway do you use for freelancing?", "Payoneer"),
    ("What's your GPU?", "GTX 1070"),
    ("What is Nemesis?", "automated penetration"),
    ("What resolution do you play CS2 on?", "1440x1080"),
    ("Where do you want to live and work after you finish your degree?", "Werkstudent"),
    ("Which tool did you use for USB evidence acquisition?", "FTK Imager"),
    ("What did you build for the hardware shop your family runs?", "Electron"),
    ("Tell me about the app for tracking habits like a video game character", "RPG-style|turns daily life into an RPG"),
    ("What awards have you won?", "3rd Position"),
    ("Which anime villain do you like?", "Sukuna"),
    ("What did you do in the database course with role-based access?", "HospitalRBAC_DB"),
    ("Where do you run local language models?", "LM Studio"),
    ("What went wrong on the school trip to the old Mughal fort?", "VR gaming"),
    ("Who do you usually team up with on class projects?", "same small group of classmates"),
    ("What do you think will happen to programming jobs because of AI?", "taking other people's jobs"),
    ("What is Kollab?", "peer-to-peer student networking"),
    ("Which football club do you support?", "FC Barcelona"),
    ("What certifications do you hold?", "Google Cybersecurity Professional"),
    # added with the expanded KB (projects, hackathon, work habits, freelancing, FAQ)
    ("What routing protocol did you use in the campus network?", "OSPF"),
    ("How long did the hackathon last?", "exactly 24 hours"),
    ("What did you build at the hackathon?", "disaster response coordinator"),
    ("How much was your first Fiverr order?", "first order, which was for $10"),
    ("How long does a football edit take you?", "2 to 3 hours"),
    ("What is your CS2 rank?", "Gold Nova"),
    ("Which TF2 class do you main?", "Demoman"),
    ("What advice would you give a new CS student?", "GitHub from day one"),
    ("How do you deal with putting things off?", "Pomodoro"),
    ("What does the AI in ERPify do?", "predictive model"),
    ("Which semester are you in?", "7th semester"),
    ("Which European cities appeal to you?", "Berlin"),
    ("Has a client ever kept expanding the scope of a cheap order?", "five major architecture revisions"),
    ("How good is your Japanese?", "Katakana"),
    ("Which subject did you enjoy most at university?", "Data Structures and Algorithms"),
]

OUT_OF_SCOPE = [
    "What is your favourite programming language?",
    "How old are you?",
    "What's the capital of France?",
    "Do you have any pets?",
]


def main():
    r = load_retriever()
    print(f"{len(r.chunks)} chunks\n")

    print("=" * 80, "\nREPORT QUESTIONS (top-3 per mode)\n" + "=" * 80)
    def gold_of(phrase):
        alts = [a.lower() for a in phrase.split("|")]
        return {c.chunk_id for c in r.chunks if any(a in c.text.lower() for a in alts)}

    for q, phrase in REPORT_QUESTIONS:
        gold = gold_of(phrase)
        print(f"\nQ: {q}\n   gold: {sorted(gold)}")
        for m in MODES:
            hits = r.search(q, m, k=3)
            cells = ", ".join(
                f"{h['chunk'].chunk_id}{'*' if h['chunk'].chunk_id in gold else ''} ({h['score']:.3f})"
                for h in hits
            )
            print(f"   {m:6s}: {cells}")

    labelled = REPORT_QUESTIONS + EXTRA_QUESTIONS
    print("\n" + "=" * 80 + f"\nhit@1 / hit@3 / MRR over {len(labelled)} labelled questions\n" + "=" * 80)
    for m in MODES:
        h1 = h3 = mrr = 0.0
        misses = []
        for q, phrase in labelled:
            gold = gold_of(phrase)
            ids = [h["chunk"].chunk_id for h in r.search(q, m, k=len(r.chunks))]
            # BM25 drops zero-score chunks, so a missing answer counts as rank n+1
            first = next((i for i, c in enumerate(ids) if c in gold), len(r.chunks)) + 1
            h1 += first == 1
            h3 += first <= 3
            mrr += 1 / first
            if first > 3:
                misses.append(f"{q} (first gold at #{first})")
        n = len(labelled)
        print(f"{m:6s}  hit@1={h1/n:.2f}  hit@3={h3/n:.2f}  MRR={mrr/n:.2f}")
        for x in misses:
            print(f"        miss: {x}")

    print("\n" + "=" * 80 + "\nOut-of-scope gate (max dense cosine / max BM25)\n" + "=" * 80)
    for q in OUT_OF_SCOPE + [q for q, _ in labelled]:
        print(f"{r.dense_scores(q).max():.3f}  {r.bm25_scores(q).max():5.2f}  {q}")


if __name__ == "__main__":
    main()
