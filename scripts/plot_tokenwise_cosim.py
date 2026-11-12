#!/usr/bin/env python3
"""Plot tokenwise gradient cosine similarity for each HH–BB pair."""

import argparse
import json
import os
import os.path as osp

import pandas as pd
import seaborn as sns

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="artifacts/tamia/tokenwise-cosim/tokenwise-cosims.jsonl"
    )
    parser.add_argument("--outpath", default="artifacts/figures/tokenwise-cosims.pdf")
    args = parser.parse_args()

    rows = []
    with open(args.input) as handle:
        for line in handle:
            result = json.loads(line)
            for t, cosine in enumerate(result["cosims"]):
                rows.append(
                    {
                        "Token index": t,
                        "Cosine similarity": cosine,
                        "BB": str(result["bb_index"] + 1),
                        "HH": str(result["hh_index"] + 1),
                    }
                )

    df = pd.DataFrame(rows)
    sns.set_theme(context="paper", style="ticks", font_scale=1.2)
    grid = sns.relplot(
        data=df,
        x="Token index",
        y="Cosine similarity",
        hue="BB",
        style="HH",
        markers=True,
        dashes=False,
        kind="line",
        estimator=None,
        height=3,
        aspect=2,
    )
    grid.ax.axhline(0, color="gray", linewidth=0.8)
    os.makedirs(osp.dirname(args.outpath) or ".", exist_ok=True)
    grid.figure.savefig(args.outpath, bbox_inches="tight")
