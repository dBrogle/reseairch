"""Cost tracking for the Emergent Values study.

Tracks per-model token usage and costs in a JSON file, updated
after each API call via an async lock to prevent write conflicts.
"""

import json
import asyncio
from pathlib import Path

from studies.emergent_values.config import OUTPUT_DIR

STUDY_DIR = Path(__file__).parent
COSTS_FILE = STUDY_DIR / OUTPUT_DIR / "costs.json"


class CostTracker:
    """Thread-safe cost tracker that persists to JSON after every update."""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._data = self._load()

    def _load(self) -> dict:
        if COSTS_FILE.exists():
            with open(COSTS_FILE, "r") as f:
                return json.load(f)
        return {}

    def _save(self):
        COSTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(COSTS_FILE, "w") as f:
            json.dump(self._data, f, indent=2)

    def _ensure_model(self, model: str):
        if model not in self._data:
            self._data[model] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "api_calls": 0,
                "errors": 0,
            }

    async def record(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float | None = None,
        is_error: bool = False,
    ):
        """Record usage from a single API call."""
        async with self._lock:
            self._ensure_model(model)
            entry = self._data[model]
            entry["prompt_tokens"] += prompt_tokens
            entry["completion_tokens"] += completion_tokens
            entry["total_tokens"] += prompt_tokens + completion_tokens
            if cost is not None:
                entry["total_cost_usd"] += cost
            entry["api_calls"] += 1
            if is_error:
                entry["errors"] += 1
            self._save()

    async def get_summary(self, model: str) -> dict | None:
        """Get current cost summary for a model."""
        async with self._lock:
            return self._data.get(model)

    def get_all(self) -> dict:
        """Get all cost data (sync, for printing after run)."""
        return self._data.copy()

    def print_summary(self, total_jobs: int | None = None, done_jobs: int | None = None):
        """Print a cost summary table."""
        data = self._data
        if not data:
            print("  No cost data recorded yet.")
            return

        print(f"\n  {'Model':<30} {'Calls':>7} {'Prompt':>10} {'Compl':>10} {'Total':>10} {'Cost':>10}")
        print(f"  {'-'*77}")

        for model, entry in data.items():
            short = model.split("/")[-1]
            cost_str = f"${entry['total_cost_usd']:.4f}" if entry['total_cost_usd'] > 0 else "N/A"
            print(
                f"  {short:<30} {entry['api_calls']:>7} "
                f"{entry['prompt_tokens']:>10,} {entry['completion_tokens']:>10,} "
                f"{entry['total_tokens']:>10,} {cost_str:>10}"
            )

        # Projection
        if total_jobs and done_jobs and done_jobs > 0:
            for model, entry in data.items():
                if entry["total_cost_usd"] > 0:
                    projected = entry["total_cost_usd"] * (total_jobs / done_jobs)
                    short = model.split("/")[-1]
                    print(f"\n  Projected total cost for {short}: ${projected:.4f} ({done_jobs}/{total_jobs} done)")
                elif entry["total_tokens"] > 0:
                    avg_tokens = entry["total_tokens"] / entry["api_calls"]
                    projected_tokens = avg_tokens * total_jobs
                    short = model.split("/")[-1]
                    print(f"\n  Projected total tokens for {short}: {int(projected_tokens):,} ({done_jobs}/{total_jobs} done)")
