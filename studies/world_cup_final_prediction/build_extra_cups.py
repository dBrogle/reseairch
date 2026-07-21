"""Pull the older World Cups (1998/2002/2006) that predate the refereeing-bias CSV.

We reuse that study's ESPN ingester (goals, stage, extra-time flag — everything our
models need) and cache the result to this study's own `data/extra_cups.csv`, so the
sibling study's 2010-2026 dataset is left untouched and this study stays offline on
re-runs. These cups give the models extra fully-out-of-sample finals to be tested on.

Run:  python -m studies.world_cup_final_prediction.build_extra_cups [--refetch]
"""

from __future__ import annotations

import sys
from datetime import date

import pandas as pd

from . import config
from ..world_cup_refereeing_bias import espn_ingest as E
from ..world_cup_refereeing_bias.common import COLUMNS


def build(refetch: bool = False) -> pd.DataFrame:
    if config.EXTRA_CSV.exists() and not refetch:
        return pd.read_csv(config.EXTRA_CSV)
    frames = []
    for year, (start, end) in config.EXTRA_CUPS.items():
        rows = E.ingest(year, date.fromisoformat(start), date.fromisoformat(end))
        frames.append(pd.DataFrame(rows))
    df = pd.concat(frames, ignore_index=True)[COLUMNS]
    config.EXTRA_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.EXTRA_CSV, index=False)
    return df


if __name__ == "__main__":
    df = build(refetch="--refetch" in sys.argv)
    print(f"\n[extra cups] {len(df)} rows -> {config.EXTRA_CSV}")
    print(df.drop_duplicates("match_id").groupby(["tournament", "stage"]).size())
