#!/usr/bin/env python3
"""Plot SEAM harmfulness and capability across attack learning rates."""

import argparse
import json
import os
import os.path as osp

import pandas as pd
import seaborn as sns

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rootdir", default="artifacts/tamia/evals")
    parser.add_argument("--sd-coefficients", action="store_true")
    parser.add_argument("--attack-cosim", action="store_true")
    parser.add_argument("--attack-sizes", action="store_true")
    parser.add_argument("--outpath")
    args = parser.parse_args()
    if args.attack_sizes and (args.sd_coefficients or args.attack_cosim):
        parser.error("Use --attack-sizes separately from the other comparison modes")

    if args.attack_cosim and not args.sd_coefficients:
        outpath = args.outpath or (
            "artifacts/figures/llama2-7b-beavertail--align-fft-seam"
            "--attack-fft-cosim--trajectory.pdf"
        )
        base = "llama2-7b-beavertail--align-fft-seam--upstream-paper-config"
        coefficients = ["0.1", "1.0", "5.0"]
        learning_rates = ["5e-5", "8e-5", "1e-4", "2e-4"]
        configs = [
            {
                "path": base
                + f"--def-sp--coeff-sd{coefficient}--attack-cosim"
                + f"-att-sp--relearn-fft-lr{lr}-ms250",
                "SD coefficient": coefficient,
                "LR": lr,
            }
            for coefficient in coefficients
            for lr in learning_rates
        ]
        rows = []
        for cfg in configs:
            path = osp.join(args.rootdir, cfg["path"], "gradient_cosim.jsonl")
            with open(path) as handle:
                for line in handle:
                    record = json.loads(line)
                    rows.append(
                        {
                            "SD coefficient": cfg["SD coefficient"],
                            "LR": cfg["LR"],
                            "Step": record["step"],
                            "Cosine similarity": record["cosine_similarity"],
                        }
                    )
        df = pd.DataFrame(rows)
        sns.set_theme(context="paper", style="ticks", font_scale=1.5)
        grid = sns.relplot(
            data=df,
            x="Step",
            y="Cosine similarity",
            hue="LR",
            col="SD coefficient",
            col_order=coefficients,
            hue_order=learning_rates,
            kind="line",
            estimator=None,
            palette="colorblind",
            height=3,
            aspect=1.2,
        )
        grid.set(ylim=(-1, 1))
        grid.set_axis_labels("Attack optimizer step", "Gradient cosine similarity")
        grid.set_titles(r"$\beta={col_name}$")
    else:
        comparison = "attack-fft"
        if args.attack_cosim:
            comparison += "-cosim"
        sweep = "-beta-sweep" if args.sd_coefficients else ""
        if args.attack_sizes:
            sweep = "-attack-size-sweep"
        outpath = args.outpath or (
            "artifacts/figures/llama2-7b-beavertail--align-fft-seam"
            f"--{comparison}--lr-trajectory{sweep}.pdf"
        )
        base = "llama2-7b-beavertail--align-fft-seam--upstream-paper-config"
        colors = sns.color_palette("colorblind", 4)
        configs = [
            {
                "path": base + "--relearn-fft-lr{lr}-ms250",
                "name": r"SEAM (def=\n, att=\n) [original]",
                "color": colors[0],
            },
            {
                "path": base + "--def-nl-att-sp--relearn-fft-lr{lr}-ms250",
                "name": r"SEAM (def=\n, att=sp)",
                "color": colors[1],
            },
            {
                "path": base + "--def-sp-att-sp--relearn-fft-lr{lr}-ms250",
                "name": r"SEAM (def=sp, att=sp)",
                "color": colors[2],
            },
        ]
        if args.sd_coefficients:
            configs = [
                {
                    "path": base + "--def-sp-att-sp--relearn-fft-lr{lr}-ms250",
                    "name": r"SEAM (def=sp, att=sp), $\beta=0.01$",
                    "color": colors[0],
                }
            ]
            if args.attack_cosim:
                configs = []
            suffix = "--attack-cosim" if args.attack_cosim else ""
            for coefficient, color in zip(["0.1", "1.0", "5.0"], colors[1:]):
                configs.append(
                    {
                        "path": base
                        + f"--def-sp--coeff-sd{coefficient}{suffix}"
                        + "-att-sp--relearn-fft-lr{lr}-ms250",
                        "name": rf"SEAM (def=sp, att=sp), $\beta={coefficient}$",
                        "color": color,
                    }
                )
        if args.attack_sizes:
            attack_sizes = [10, 100, 200, 500]
            configs = [
                {
                    "path": base
                    + f"--def-sp--coeff-sd5.0-att-sp--relearn-fft-as{size}-lr{{lr}}",
                    "name": r"SEAM (def=sp, att=sp), $\beta=5.0$",
                    "color": colors[3],
                    "attack_size": size,
                }
                for size in attack_sizes
            ]
        else:
            configs.append(
                {
                    "path": "llama2-7b-beavertail--data-overlapping--relearn-fft-author-codebase-lr{lr}-ms250",
                    "name": "Base (fine-tuned)",
                    "color": "black",
                }
            )
        learning_rates = ["5e-5", "8e-5", "1e-4", "2e-4"]
        task_metrics = {
            "mmlu": "acc,none",
            "truthfulqa_mc1": "acc,none",
            "arc_easy": "acc_norm,none",
            "hellaswag": "acc_norm,none",
        }

        rows = []
        for cfg in configs:
            for lr in learning_rates:
                model = cfg["path"].format(lr=lr)
                with open(osp.join(args.rootdir, model, "results.json")) as handle:
                    result = json.load(handle)
                capability = sum(
                    result["post_utility"][task][metric]
                    for task, metric in task_metrics.items()
                ) / len(task_metrics)
                for evaluation, score in [
                    ("Harmfulness ↓", result["harmfulness"]),
                    ("Capability ↑", capability),
                ]:
                    rows.append(
                        {
                            "Method": cfg["name"],
                            "Attack size": cfg.get("attack_size"),
                            "LR": lr,
                            "Eval": evaluation,
                            "Score (%)": 100 * score,
                        }
                    )

        df = pd.DataFrame(rows)
        sns.set_theme(context="paper", style="ticks", font_scale=1.5)
        grid = sns.catplot(
            data=df,
            x="Attack size" if args.attack_sizes else "LR",
            y="Score (%)",
            hue="Method",
            col="LR" if args.attack_sizes else "Eval",
            row="Eval" if args.attack_sizes else None,
            order=attack_sizes if args.attack_sizes else learning_rates,
            col_order=(
                learning_rates
                if args.attack_sizes
                else ["Harmfulness ↓", "Capability ↑"]
            ),
            row_order=["Harmfulness ↓", "Capability ↑"] if args.attack_sizes else None,
            margin_titles=args.attack_sizes,
            kind="point",
            errorbar=None,
            palette={cfg["name"]: cfg["color"] for cfg in configs},
            height=3,
            aspect=1.4,
        )
        grid.set(ylim=(0, 100))
        grid.set_axis_labels(
            "Attack samples" if args.attack_sizes else "Attack learning rate",
            "Score (%)",
        )
        if args.attack_sizes:
            grid.set_titles(col_template="LR = {col_name}", row_template="{row_name}")
        else:
            grid.set_titles("{col_name}")
    os.makedirs(osp.dirname(outpath) or ".", exist_ok=True)
    grid.figure.savefig(outpath, bbox_inches="tight")
