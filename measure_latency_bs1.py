"""
Measure batch-size-1 inference latency for every model/dataset checkpoint.

Methodology:
  - Load the saved checkpoint for each (model, dataset) pair.
  - Feed a single sample (batch size 1) through the model on GPU.
  - Warm up with 50 forward passes, then time 200 passes using CUDA events.
  - Report the median latency in milliseconds.

Output: results/latency_bs1.csv
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

from util import load_data_n_model


MODELS = [
    "MLP", "LeNet", "ResNet18", "ResNet50", "ResNet101",
    "RNN", "GRU", "LSTM", "BiLSTM", "CNN+GRU", "ViT",
]

DATASETS = ["UT_HAR_data", "NTU-Fi_HAR", "NTU-Fi-HumanID", "Widar"]

WARMUP_RUNS = 50
MEASURE_RUNS = 200


def measure_latency_bs1(model, sample, device):
    """Return median latency in ms over MEASURE_RUNS forward passes."""
    model.eval()
    sample = sample.to(device)

    # Warm up
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            _ = model(sample)
    torch.cuda.synchronize()

    # Timed runs using CUDA events for accurate GPU timing
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    times = []

    with torch.no_grad():
        for _ in range(MEASURE_RUNS):
            starter.record()
            _ = model(sample)
            ender.record()
            torch.cuda.synchronize()
            times.append(starter.elapsed_time(ender))  # milliseconds

    return float(np.median(times))


def main():
    parser = argparse.ArgumentParser("Batch-size-1 latency measurement")
    parser.add_argument("--root", default="./Data/", help="Data root directory")
    parser.add_argument("--results-dir", default="results", help="Results root directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, latency will reflect CPU timing.", flush=True)

    output_path = os.path.join(args.results_dir, "latency_bs1.csv")
    fieldnames = ["model", "dataset", "latency_bs1_ms", "warmup_runs", "measure_runs", "note"]

    rows = []

    for dataset in DATASETS:
        for model_name in MODELS:
            checkpoint_path = os.path.join(
                args.results_dir, model_name, dataset, "model_checkpoint.pt"
            )
            print(f"\n[{model_name} / {dataset}]", flush=True)

            if not os.path.exists(checkpoint_path):
                print(f"  SKIP — checkpoint not found: {checkpoint_path}", flush=True)
                rows.append({
                    "model": model_name,
                    "dataset": dataset,
                    "latency_bs1_ms": "",
                    "warmup_runs": WARMUP_RUNS,
                    "measure_runs": MEASURE_RUNS,
                    "note": "checkpoint missing",
                })
                continue

            try:
                # Build model architecture and get a test loader for one sample
                _, test_loader, model, _ = load_data_n_model(dataset, model_name, args.root)

                # Load trained weights
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.to(device)
                model.eval()

                # Grab one sample from the test set (batch_size may be >1, so slice to 1)
                sample_batch, _ = next(iter(test_loader))
                sample = sample_batch[:1]  # batch size 1

                latency_ms = measure_latency_bs1(model, sample, device)
                print(f"  latency (bs=1) = {latency_ms:.4f} ms", flush=True)

                rows.append({
                    "model": model_name,
                    "dataset": dataset,
                    "latency_bs1_ms": round(latency_ms, 4),
                    "warmup_runs": WARMUP_RUNS,
                    "measure_runs": MEASURE_RUNS,
                    "note": "",
                })

            except Exception as e:
                print(f"  ERROR: {e}", flush=True)
                rows.append({
                    "model": model_name,
                    "dataset": dataset,
                    "latency_bs1_ms": "",
                    "warmup_runs": WARMUP_RUNS,
                    "measure_runs": MEASURE_RUNS,
                    "note": f"error: {e}",
                })

    # Write output CSV
    os.makedirs(args.results_dir, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Results written to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
