"""
Transcript analysis — sycophancy examples.

Usage:
    uv run python experiments/transcript_analysis.py
"""
import jsonlines
import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "logiqa_groq_2026-05-06_23-35-40")

PATHS = {
    "Blind":       "Blind/_Allama-3.3-70b-versatile_Jllama-3.1-8b-instant/results.jsonl",
    "Debate":      "Debate_t0_n2/_Allama-3.3-70b-versatile_Jllama-3.1-8b-instant_Bllama-3.3-70b-versatile/results.jsonl",
    "Consultancy": "Consultancy_t0_n2/_Allama-3.3-70b-versatile_Jllama-3.1-8b-instant/results.jsonl",
}


def load_all(path):
    questions = {}
    with jsonlines.open(os.path.join(RESULTS_DIR, path)) as f:
        for q in f:
            correct = next((a for a in q["answer_cases"] if a["value"] == 1.0), None)
            if not correct:
                continue
            cp = correct.get("case_probs")
            if not cp:
                continue
            correct_in_cp = next((a for a in cp["answer_cases"] if a["value"] == 1.0), None)
            distractor_in_cp = next((a for a in cp["answer_cases"] if a["value"] != 1.0), None)
            if not correct_in_cp:
                continue
            key = cp["question"][:80]
            questions[key] = {
                "full_question": cp["question"],
                "correct_answer": correct_in_cp["long"],
                "distractor_answer": distractor_in_cp["long"] if distractor_in_cp else "N/A",
                "correct_short": correct_in_cp["short"],
                "distractor_short": distractor_in_cp["short"] if distractor_in_cp else "?",
                "prob_correct": round(correct_in_cp["judge_prob"]["prob"], 3),
                "transcript": cp.get("transcript", []),
            }
    return questions


def find_contrast_cases(debate, consult, blind, debate_threshold=0.45, consult_threshold=0.60):
    """Find questions where debate failed but consultancy succeeded."""

    hits = []
    for key in debate:
        if key not in consult:
            continue
        d = debate[key]
        c = consult[key]
        b = blind.get(key, {})
        if d["prob_correct"] < debate_threshold and c["prob_correct"] > consult_threshold:
            hits.append({
                "key": key,
                "blind_prob": b.get("prob_correct", "N/A"),
                "debate_prob": d["prob_correct"],
                "consult_prob": c["prob_correct"],
                "debate_data": d,
                "consult_data": c,
            })
    hits.sort(key=lambda x: x["debate_prob"])
    return hits


def print_example(hit, index):
    d = hit["debate_data"]
    c = hit["consult_data"]

    print(f"\n{'='*70}")
    print(f"EXAMPLE {index} — Sycophancy Detection")
    print(f"{'='*70}")
    print(f"\nQUESTION:\n{d['full_question']}")
    print(f"\nANSWER CHOICES:")
    print(f"  ({d['correct_short']}) {d['correct_answer']}  ← CORRECT")
    print(f"  ({d['distractor_short']}) {d['distractor_answer']}  ← WRONG (distractor)")

    print(f"\n{'─'*70}")
    print(f"JUDGE PROBABILITY (correct answer):")
    print(f"  Blind:       {hit['blind_prob']:.2f}  (unaided)")
    print(f"  Debate:      {hit['debate_prob']:.2f}  ← WRONG")
    print(f"  Consultancy: {hit['consult_prob']:.2f}  ← CORRECT")

    print(f"\n{'─'*70}")
    print("DEBATE TRANSCRIPT:")
    for t in d["transcript"]:
        role = t["role"]
        label = f"Debater {role} (CORRECT side)" if role == d["correct_short"] else f"Debater {role} (WRONG side)"
        print(f"\n  [{label}]")
        print(f"  {t['content'][:500]}")

    print(f"\n{'─'*70}")
    print("CONSULTANCY TRANSCRIPT:")
    for t in c["transcript"]:
        role = t["role"]
        label = "Judge (question)" if role == "client" else "Consultant (answer, arguing for CORRECT side)"
        print(f"\n  [{label}]")
        print(f"  {t['content'][:500]}")

    print(f"\n{'─'*70}")
    print("ANALYSIS:")
    if hit["debate_prob"] == 0.0:
        print("  → Debate: Judge was COMPLETELY fooled by the distractor (0% to correct).")
    else:
        print(f"  → Debate: Judge was largely fooled by the distractor ({hit['debate_prob']:.0%} to correct).")
    print(f"  → Consultancy: Judge reached the correct answer by asking targeted questions ({hit['consult_prob']:.0%} to correct).")


def main():
    print("Loading data...")
    blind   = load_all(PATHS["Blind"])
    debate  = load_all(PATHS["Debate"])
    consult = load_all(PATHS["Consultancy"])

    hits = find_contrast_cases(debate, consult, blind)
    print(f"\nFound {len(hits)} contrast cases (Debate wrong, Consultancy correct).\n")

    print("\n" + "="*70)
    print("SUMMARY TABLE — Contrast Cases")
    print("="*70)
    print(f"{'Question (first 60 chars)':<62} {'Blind':>6} {'Debate':>7} {'Consult':>8}")
    print("─"*70)
    for h in hits:
        print(f"{h['key'][:62]:<62} {str(h['blind_prob']):>6} {h['debate_prob']:>7.2f} {h['consult_prob']:>8.2f}")

    # Print top 2 most dramatic examples
    for i, hit in enumerate(hits[:2], 1):
        print_example(hit, i)


if __name__ == "__main__":
    main()
