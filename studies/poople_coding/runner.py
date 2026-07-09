"""Ask each model (one shot, reasoning ON) to write a Poople solver program."""

import asyncio
import re

from services.llm import OpenRouterProvider
from studies.poople_coding.config import MAX_TOKENS, MODELS, TEMPERATURE
from studies.poople_coding.cache import load_code, save_code
from studies.poople_coding.prompt import build_prompt


def extract_code(text: str) -> str:
    """Pull the program out of a model response.

    Prefers the longest fenced ``` block; otherwise returns the text as-is. Strips
    a leading language tag (```python) and surrounding whitespace.
    """
    if not text:
        return ""
    blocks = re.findall(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", text, re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip()
    return text.strip()


async def _generate_one(provider: OpenRouterProvider, model: str) -> dict:
    prompt = build_prompt()
    try:
        try:
            raw = await provider.complete_text(
                prompt=prompt, model=model, temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS, enable_reasoning=True,
            )
        except RuntimeError as e:
            if "reasoning" in str(e).lower():
                raw = await provider.complete_text(
                    prompt=prompt, model=model, temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS, omit_reasoning=True,
                )
            else:
                raise
        code = extract_code(raw)
        return save_code(model, code or None, raw, None if code else "empty code")
    except Exception as e:
        return save_code(model, None, None, str(e))


async def generate_all(models: list[str] = MODELS) -> dict[str, dict]:
    """Generate (or load cached) a program for each model. Returns {model: entry}."""
    out: dict[str, dict] = {}
    pending = []
    for model in models:
        cached = load_code(model)
        if cached is not None:
            out[model] = cached
            print(f"  [cached] {model.split('/')[-1]}")
        else:
            pending.append(model)

    if pending:
        provider = OpenRouterProvider()
        print(f"  Generating programs for {len(pending)} model(s) (reasoning ON)...")
        results = await asyncio.gather(*[_generate_one(provider, m) for m in pending])
        for model, entry in zip(pending, results):
            out[model] = entry
            status = "OK" if entry.get("code") else f"ERROR: {entry.get('error')}"
            print(f"  [done] {model.split('/')[-1]} -> {status}")
    return out
