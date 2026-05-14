"""Quick verification that the updated prompts produce non-refusal answers.

Uses the actual study config (post-update) against 2 models x 4 names.

Usage:
    python -m studies.god_by_name.test.test_verify
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))

from services.llm import OpenRouterProvider
from studies.god_by_name.config import (
    SYSTEM_MESSAGE,
    NAME_CONTEXT_TEMPLATE,
    CONVERSATION_SEED,
    QUESTION,
    EXTRACTION_SYSTEM_PROMPT,
)

MODELS = ["openai/gpt-5.4", "anthropic/claude-sonnet-4.6", "google/gemini-3.1-flash-lite-preview", "deepseek/deepseek-v3.2"]
EXTRACTOR_MODEL = "openai/gpt-5.4"
TEST_NAMES = ["Jake", "Shaniqua", "Priya", "Carlos"]


def build_messages(name: str) -> list[dict]:
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "system", "content": NAME_CONTEXT_TEMPLATE.format(name=name)},
    ]
    for role, content in CONVERSATION_SEED:
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": QUESTION})
    return messages


async def main():
    provider = OpenRouterProvider()
    semaphore = asyncio.Semaphore(30)

    print(f"\nQuestion: {QUESTION}")
    print(f"Names: {TEST_NAMES}")
    print(f"Models: {[m.split('/')[-1] for m in MODELS]}\n")

    # Collect responses
    async def query(model, name):
        messages = build_messages(name)
        async with semaphore:
            try:
                return await provider.complete_text(
                    messages=messages, model=model, temperature=1.0, max_tokens=300,
                )
            except Exception as e:
                return f"ERROR: {e}"

    tasks = {}
    for model in MODELS:
        for name in TEST_NAMES:
            tasks[(model, name)] = query(model, name)

    results_list = await asyncio.gather(*tasks.values())
    responses = dict(zip(tasks.keys(), results_list))

    # Extract per model
    for model in MODELS:
        short = model.split("/")[-1]
        batch = {}
        for idx, name in enumerate(TEST_NAMES):
            batch[str(idx)] = {
                "question": QUESTION,
                "response": responses[(model, name)],
            }

        try:
            async with semaphore:
                ext = await provider.complete_text(
                    messages=[
                        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps(batch, indent=2)},
                    ],
                    model=EXTRACTOR_MODEL, temperature=0.0, max_tokens=200,
                )
            text = ext.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
            extractions = json.loads(text)
        except Exception:
            extractions = {}

        print(f"--- {short} ---")
        for idx, name in enumerate(TEST_NAMES):
            answer = extractions.get(str(idx), "ERROR")
            preview = responses[(model, name)][:100].replace("\n", " ")
            marker = {"YES": "✅", "NO": "❌"}.get(answer, "⚪")
            print(f"  {marker} {answer:8} {name:12} {preview}...")

        yes = sum(1 for i, n in enumerate(TEST_NAMES) if extractions.get(str(i)) == "YES")
        no = sum(1 for i, n in enumerate(TEST_NAMES) if extractions.get(str(i)) == "NO")
        ref = len(TEST_NAMES) - yes - no
        print(f"  => Y={yes} N={no} R={ref} ({(yes+no)/len(TEST_NAMES)*100:.0f}% answered)\n")


if __name__ == "__main__":
    asyncio.run(main())
