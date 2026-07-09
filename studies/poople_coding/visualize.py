"""Charts for the Poople Coding benchmark (reuses the poople chart helpers)."""

from pathlib import Path

import numpy as np

from utils.graphing import heatmap
# brand-colored bars + donut-pie helpers
from studies.poople.visualize import _headline_barh, _outcome_pies


def generate_graphs(results: list[dict], save_dir: Path, subtitle: str = ""):
    save_dir.mkdir(parents=True, exist_ok=True)

    _headline_barh(
        results, lambda s: s["optimal_rate"],
        "Poople Coding — optimal-solve rate", subtitle,
        "% of test words solved optimally by the model's program",
        save_dir / "optimal_rate.png", value_fmt=".0f", suffix="%", xmax=100,
    )

    # Outcome donut per model (same 4-way scheme as the poople study). For a
    # correct solver every test word is "par", so these read as all-green.
    pie_stats = [
        {"model": r["model"], "overall": {"pie": r["pie"], "solve_rate": r["optimal_rate"]}}
        for r in results
    ]
    _outcome_pies(pie_stats, save_dir / "outcomes_pie.png", subtitle,
                  title="Poople Coding — outcome mix by model", solved_word="optimal")
    _headline_barh(
        results, lambda s: s["valid_rate"],
        "Poople Coding — valid-ladder rate", subtitle,
        "% of test words with a legal (if not optimal) ladder",
        save_dir / "valid_rate.png", value_fmt=".0f", suffix="%", xmax=100,
    )

    # Optimal rate by difficulty (model × distance), only over models that ran.
    ran = [r for r in results if r["ran"]]
    dists = sorted({int(d) for r in ran for d in r["per_distance"]}, )
    if ran and dists:
        labels = [r["short_name"] for r in ran]
        data, annot = [], []
        for r in ran:
            row, arow = [], []
            for d in dists:
                cell = r["per_distance"].get(str(d))
                if cell and cell["n"]:
                    row.append(cell["optimal_rate"]); arow.append(f"{cell['optimal_rate']:.0f}")
                else:
                    row.append(np.nan); arow.append("—")
            data.append(row); annot.append(arow)
        heatmap(
            data, labels, [f"par {d}" for d in dists],
            "Optimal-solve rate by difficulty (model's program)", "optimal distance", "",
            save_dir / "heatmap_optimal_by_distance.png",
            value_range=(0, 100), cmap="RdYlGn", annotations=annot,
        )

    print(f"  Graphs saved to {save_dir}/")
