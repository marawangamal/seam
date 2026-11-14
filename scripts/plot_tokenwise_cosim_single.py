#!/usr/bin/env python3
"""Plot tokenwise HH gradient alignment with a full-sequence Alpaca gradient."""

import argparse
import json
import os
import os.path as osp

import pandas as pd
import seaborn as sns

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="artifacts/tamia/tokenwise-cosim/tokenwise-cosims.json"
    )
    parser.add_argument("--outpath", default="artifacts/figures/tokenwise-cosim.pdf")
    args = parser.parse_args()

    with open(args.input) as handle:
        result = json.load(handle)
    rows = [
        {"Token": i, "Cosine similarity": cosine}
        for i, cosine in enumerate(result["cosims"])
    ]
    sns.set_theme(context="paper", style="ticks", font_scale=1.2)
    grid = sns.relplot(
        data=pd.DataFrame(rows),
        x="Token",
        y="Cosine similarity",
        kind="line",
        marker="o",
        height=3,
        aspect=max(2, len(rows) / 12),
    )
    grid.set(xticks=range(len(rows)))
    grid.set_xticklabels(
        [token.replace("\n", r"\n") for token in result["tokens"]], rotation=90
    )
    grid.ax.axhline(0, color="gray", linewidth=0.8)
    grid.set_axis_labels("HH completion token", "Cosine similarity")
    os.makedirs(osp.dirname(args.outpath) or ".", exist_ok=True)
    grid.figure.savefig(args.outpath, bbox_inches="tight")
