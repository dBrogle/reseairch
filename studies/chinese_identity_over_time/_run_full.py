"""Non-interactive full run of the Chinese Identity Over Time study."""

import asyncio

from services.llm import OpenRouterProvider
from studies.chinese_identity_over_time.config import ALL_MODELS
from studies.chinese_identity_over_time.runner import run_all
from studies.chinese_identity_over_time.judge import run_judge
from studies.chinese_identity_over_time.graphs import (
    load_all_results, generate_graphs, export_frontend_json, print_summary,
)


async def main():
    p = OpenRouterProvider()
    await run_all(p, ALL_MODELS)
    await run_judge(p, ALL_MODELS)
    res = load_all_results()
    generate_graphs(res)
    export_frontend_json(res)
    print_summary(res)


if __name__ == "__main__":
    asyncio.run(main())
