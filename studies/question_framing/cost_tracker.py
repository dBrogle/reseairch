"""Cost tracking for the Question Framing study.

Fetches per-token pricing from OpenRouter's model catalogue once, then
accumulates cost as API calls are made. Provides running totals and
projections scoped to the current phase (runner vs grader).
"""

import asyncio
import httpx


class CostTracker:
    def __init__(self):
        self.total_cost = 0.0
        self.total_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._lock = asyncio.Lock()
        self._pricing: dict[str, tuple[float, float]] = {}
        self._phase_cost = 0.0

    async def fetch_pricing(self, api_key: str, models: list[str]):
        """Fetch per-token pricing from OpenRouter for the requested models."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    "https://openrouter.ai/api/v1/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                resp.raise_for_status()
                data = resp.json()

            model_set = set(models)
            for info in data.get("data", []):
                model_id = info.get("id")
                if model_id in model_set:
                    pricing = info.get("pricing", {})
                    self._pricing[model_id] = (
                        float(pricing.get("prompt", "0")),
                        float(pricing.get("completion", "0")),
                    )

            for m in models:
                if m in self._pricing:
                    p, c = self._pricing[m]
                    print(f"  Pricing {m}: ${p * 1_000_000:.2f}/1M prompt, ${c * 1_000_000:.2f}/1M completion")
                else:
                    print(f"  Pricing {m}: unknown (cost will not be tracked)")
        except Exception as e:
            print(f"  Warning: could not fetch pricing from OpenRouter: {e}")

    def start_phase(self):
        """Reset per-phase cost counter (call before runner / grader)."""
        self._phase_cost = 0.0

    async def record(self, model: str, usage: dict):
        """Record token usage from a single API call."""
        async with self._lock:
            pt = usage.get("prompt_tokens", 0)
            ct = usage.get("completion_tokens", 0)
            self.prompt_tokens += pt
            self.completion_tokens += ct
            self.total_calls += 1

            prompt_rate, completion_rate = self._pricing.get(model, (0, 0))
            call_cost = pt * prompt_rate + ct * completion_rate
            self.total_cost += call_cost
            self._phase_cost += call_cost

    def format_status(self, done: int, total: int) -> str:
        """Running total + projected phase cost."""
        if done > 0:
            phase_projected = self._phase_cost * total / done
            total_projected = (self.total_cost - self._phase_cost) + phase_projected
        else:
            total_projected = 0.0
        return f"${self.total_cost:.4f} spent, ~${total_projected:.4f} projected"

    def format_total(self) -> str:
        return (
            f"${self.total_cost:.4f} "
            f"({self.prompt_tokens:,} prompt + {self.completion_tokens:,} completion tokens, "
            f"{self.total_calls:,} calls)"
        )
