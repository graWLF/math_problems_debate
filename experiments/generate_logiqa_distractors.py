"""
Generates synthetic deceptive distractors for LogiQA questions using an LLM.
Output: solib/data/logiqa/logiqa_augmented.json

Usage:
    uv run python experiments/generate_logiqa_distractors.py [--limit N] [--model MODEL]
"""
import asyncio
import json
import re
import argparse
import os.path as osp
import os
from datasets import load_dataset

from solib.utils.globals import jinja_env
from solib.utils.llm_utils import acompletion_ratelimited

OUTPUT_DIR = osp.join(osp.dirname(__file__), "..", "solib", "data", "logiqa")
OUTPUT_PATH = osp.join(OUTPUT_DIR, "logiqa_augmented.json")

SYSTEM_TEMPLATE = jinja_env.get_template("data_generation/logiqa_distractor_system.jinja")
USER_TEMPLATE = jinja_env.get_template("data_generation/logiqa_distractor_user.jinja")


def parse_alternate_answer(response_text: str) -> str | None:
    match = re.search(r"<alternate_answer>(.*?)</alternate_answer>", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


async def generate_distractor(item: dict, model: str) -> str | None:
    context = item["context"]
    query = item["query"]
    options = item["options"]
    correct_idx = item["correct_option"]
    question = f"{context}\n\nQuestion: {query}"
    correct_answer = options[correct_idx]

    system_prompt = SYSTEM_TEMPLATE.render()
    user_prompt = USER_TEMPLATE.render(question=question, correct_answer=correct_answer)

    response = await acompletion_ratelimited(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    text = response.choices[0].message.content
    return parse_alternate_answer(text)


async def main(limit: int, model: str):
    dset = load_dataset("lucasmccabe/logiqa", split="train")
    if limit:
        dset = dset.select(range(limit))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []
    failed = 0

    raw_items = list(dset)
    print(f"Generating distractors for {len(raw_items)} questions using {model}...")

    # Process in batches to avoid TPM rate limits
    BATCH_SIZE = 5
    responses = []
    for i in range(0, len(raw_items), BATCH_SIZE):
        batch = raw_items[i:i + BATCH_SIZE]
        batch_tasks = [generate_distractor(item, model) for item in batch]
        batch_responses = await asyncio.gather(*batch_tasks)
        responses.extend(batch_responses)
        print(f"  {min(i + BATCH_SIZE, len(raw_items))}/{len(raw_items)} done")

    for item, distractor in zip(raw_items, responses):
        if distractor is None:
            failed += 1
            distractor = item["options"][
                next(i for i in range(len(item["options"])) if i != item["correct_option"])
            ]
        results.append({
            "context": item["context"],
            "query": item["query"],
            "options": item["options"],
            "correct_option": item["correct_option"],
            "synthetic_distractor": distractor,
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Done. {len(results)} items saved to {OUTPUT_PATH}")
    if failed:
        print(f"  Warning: {failed} items fell back to original distractor (parse failed).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100, help="Number of questions to process")
    parser.add_argument("--model", type=str, default="groq/llama-3.3-70b-versatile")
    args = parser.parse_args()
    asyncio.run(main(args.limit, args.model))
