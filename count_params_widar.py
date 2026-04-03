"""
Count total trainable parameters for every model on the Widar dataset.

Loads each model architecture via load_data_n_model (same factory used during
training) and counts parameters.  No GPU and no data loading required.

Output: results/params_widar.csv
"""

import argparse
import csv
import os

import torch

from util import load_data_n_model

MODELS = [
    "MLP", "LeNet", "ResNet18", "ResNet50", "ResNet101",
    "RNN", "GRU", "LSTM", "BiLSTM", "CNN+GRU", "ViT",
]

DATASET = "Widar"


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def human_readable(n):
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def main():
    parser = argparse.ArgumentParser("Count parameters for Widar models")
    parser.add_argument("--root", default="./Data/", help="Data root directory")
    parser.add_argument("--results-dir", default="results", help="Results root directory")
    args = parser.parse_args()

    output_path = os.path.join(args.results_dir, "params_widar.csv")
    fieldnames = ["model", "dataset", "total_params", "params_human", "note"]

    rows = []

    for model_name in MODELS:
        checkpoint_path = os.path.join(
            args.results_dir, model_name, DATASET, "model_checkpoint.pt"
        )
        print(f"\n[{model_name} / {DATASET}]", flush=True)

        if not os.path.exists(checkpoint_path):
            print(f"  SKIP — checkpoint not found: {checkpoint_path}", flush=True)
            rows.append({
                "model": model_name,
                "dataset": DATASET,
                "total_params": "",
                "params_human": "",
                "note": "checkpoint missing",
            })
            continue

        try:
            _, _, model, _ = load_data_n_model(DATASET, model_name, args.root)
            n = count_params(model)
            hr = human_readable(n)
            print(f"  params = {n:,}  ({hr})", flush=True)
            rows.append({
                "model": model_name,
                "dataset": DATASET,
                "total_params": n,
                "params_human": hr,
                "note": "",
            })

        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            rows.append({
                "model": model_name,
                "dataset": DATASET,
                "total_params": "",
                "params_human": "",
                "note": f"error: {e}",
            })

    os.makedirs(args.results_dir, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Results written to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
