#!/usr/bin/env python3
"""Plot cosine distributions across HH–BB pairs at each completion token index."""

import argparse
import json
import os
import os.path as osp

import pandas as pd
import seaborn as sns

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="artifacts/tamia/tokenwise-cosim-7x7/tokenwise-cosims.jsonl"
    )
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument(
        "--outpath", default="artifacts/figures/tokenwise-cosims-7x7-violin.pdf"
    )
    args = parser.parse_args()

    rows = []
    with open(args.input) as handle:
        for line in handle:
            result = json.loads(line)
            for t, cosine in enumerate(result["cosims"][: args.max_tokens]):
                rows.append({"Token index": t, "Cosine similarity": cosine})

    df = pd.DataFrame(rows)
    sns.set_theme(context="paper", style="ticks", font_scale=1.2)
    grid = sns.catplot(
        data=df,
        x="Token index",
        y="Cosine similarity",
        kind="violin",
        cut=0,
        color="steelblue",
        height=3,
        aspect=2.5,
    )
    grid.ax.axhline(0, color="gray", linewidth=0.8)
    os.makedirs(osp.dirname(args.outpath) or ".", exist_ok=True)
    grid.figure.savefig(args.outpath, bbox_inches="tight")
