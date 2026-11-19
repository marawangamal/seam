#!/usr/bin/env python3
"""Compare SEAM attack gradient alignment and benign probe loss."""

import argparse
import json
import os
import os.path as osp

import pandas as pd
import seaborn as sns

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rootdir", default="artifacts/tamia/evals")
    parser.add_argument("--lr", default="5e-5")
    parser.add_argument(
        "--probe-mode",
        choices=["legacy", "batch", "fixed"],
        default="legacy",
        help="Fixed mode uses the 100+100 probes measured every five steps",
    )
    parser.add_argument("--outpath", required=True)
    args = parser.parse_args()

    base = "llama2-7b-beavertail--align-fft-seam--upstream-paper-config"
    suffix = {
        "legacy": "--attack-cosim",
        "batch": "-rerun--attack-batch-cosim",
        "fixed": "-rerun--attack-fixed-cosim-n100-i5",
    }[args.probe_mode]
    configs = [
        {
            "path": base + "--def-nl--coeff-sd0.01" + suffix + "-att-nl",
            "name": r"SEAM (def=\n, att=\n), $\beta=0.01$ [original]",
        },
        {
            "path": base + "--def-sp--coeff-sd1.0" + suffix + "-att-sp",
            "name": r"SEAM (def=sp, att=sp), $\beta=1.0$",
        },
    ]
    metrics = {
        "cosine_similarity": "Cosine similarity",
        (
            "benign_loss" if args.probe_mode == "batch" else "benign_probe_loss"
        ): "Benign loss",
    }

    rows = []
    for cfg in configs:
        model = cfg["path"] + f"--relearn-fft-lr{args.lr}-ms250"
        with open(osp.join(args.rootdir, model, "gradient_cosim.jsonl")) as handle:
            for line in handle:
                record = json.loads(line)
                for key, metric in metrics.items():
                    rows.append(
                        {
                            "Method": cfg["name"],
                            "Step": record["step"],
                            "Metric": metric,
                            "Value": record[key],
                        }
                    )

    df = pd.DataFrame(rows)
    sns.set_theme(context="paper", style="ticks", font_scale=1.5)
    grid = sns.relplot(
        data=df,
        x="Step",
        y="Value",
        hue="Method",
        col="Metric",
        kind="line",
        estimator=None,
        palette="colorblind",
        facet_kws={"sharey": False},
        height=3,
        aspect=1.4,
    )
    grid.set_axis_labels("Attack step", "")
    grid.set_titles("{col_name}")
    grid.axes_dict["Cosine similarity"].set(ylim=(-1, 1))
    if args.probe_mode == "legacy":
        for ax in grid.axes.flat:
            ax.axvline(12, color="gray", linestyle="--", linewidth=1)
    os.makedirs(osp.dirname(args.outpath) or ".", exist_ok=True)
    grid.figure.savefig(args.outpath, bbox_inches="tight")
