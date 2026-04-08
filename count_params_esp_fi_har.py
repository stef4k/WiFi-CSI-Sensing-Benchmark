"""
Count total parameters for every model on the ESP-Fi-HAR dataset.

Models are instantiated directly from esp_fi_har_model — no data loading
or GPU required.

Output: results/params_esp_fi_har.csv
"""

import argparse
import csv
import os

from esp_fi_har_model import (
    ESP_Fi_HAR_MLP,
    ESP_Fi_HAR_LeNet,
    ESP_Fi_HAR_ResNet18,
    ESP_Fi_HAR_ResNet50,
    ESP_Fi_HAR_ResNet101,
    ESP_Fi_HAR_RNN,
    ESP_Fi_HAR_GRU,
    ESP_Fi_HAR_LSTM,
    ESP_Fi_HAR_BiLSTM,
    ESP_Fi_HAR_CNN_GRU,
    ESP_Fi_HAR_ViT,
)

DATASET = "esp-fi-har"

MODELS = {
    "MLP":      ESP_Fi_HAR_MLP,
    "LeNet":    ESP_Fi_HAR_LeNet,
    "RNN":      ESP_Fi_HAR_RNN,
    "GRU":      ESP_Fi_HAR_GRU,
    "LSTM":     ESP_Fi_HAR_LSTM,
    "BiLSTM":   ESP_Fi_HAR_BiLSTM,
    "CNN+GRU":  ESP_Fi_HAR_CNN_GRU,
    "ResNet18":  ESP_Fi_HAR_ResNet18,
    "ResNet50":  ESP_Fi_HAR_ResNet50,
    "ResNet101": ESP_Fi_HAR_ResNet101,
    "ViT":      ESP_Fi_HAR_ViT,
}


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def human_readable(n):
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def main():
    parser = argparse.ArgumentParser("Count parameters for ESP-Fi-HAR models")
    parser.add_argument("--results-dir", default="results", help="Results root directory")
    args = parser.parse_args()

    output_path = os.path.join(args.results_dir, "params_esp_fi_har.csv")
    fieldnames = ["model", "dataset", "total_params", "params_human", "note"]

    rows = []

    for model_name, model_cls in MODELS.items():
        print(f"\n[{model_name} / {DATASET}]", flush=True)
        try:
            model = model_cls()
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
