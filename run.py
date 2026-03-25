import argparse
import csv
import os
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn

from util import load_data_n_model


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(model, tensor_loader, num_epochs, learning_rate, criterion, device):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []

    train_start = time.perf_counter()
    for epoch in range(num_epochs):
        epoch_start = time.perf_counter()
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for data in tensor_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.long().to(device)

            optimizer.zero_grad()
            outputs = model(inputs).float()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)
            predict_y = torch.argmax(outputs, dim=1)
            epoch_correct += (predict_y == labels).sum().item()
            epoch_total += labels.size(0)

        epoch_loss = epoch_loss / len(tensor_loader.dataset)
        epoch_accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0.0
        epoch_time_s = time.perf_counter() - epoch_start
        history.append(
            {
                "epoch": epoch + 1,
                "train_accuracy": float(epoch_accuracy),
                "train_loss": float(epoch_loss),
                "epoch_time_s": float(epoch_time_s),
            }
        )
        print(
            "Epoch:{}, Accuracy:{:.4f},Loss:{:.9f}".format(
                epoch + 1, float(epoch_accuracy), float(epoch_loss)
            )
        )

    training_time_s = time.perf_counter() - train_start
    return history, training_time_s


def compute_classification_metrics(y_true, y_pred, num_classes):
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (y_true, y_pred), 1)

    tp = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)

    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )

    total_samples = confusion.sum()
    accuracy = float(tp.sum() / total_samples) if total_samples > 0 else 0.0

    present_mask = support > 0
    if np.any(present_mask):
        macro_precision = float(np.mean(precision[present_mask]))
        macro_recall = float(np.mean(recall[present_mask]))
        macro_f1 = float(np.mean(f1[present_mask]))
        balanced_accuracy = float(np.mean(recall[present_mask]))
    else:
        macro_precision = 0.0
        macro_recall = 0.0
        macro_f1 = 0.0
        balanced_accuracy = 0.0

    support_sum = float(np.sum(support))
    if support_sum > 0:
        weighted_precision = float(np.sum(precision * support) / support_sum)
        weighted_recall = float(np.sum(recall * support) / support_sum)
        weighted_f1 = float(np.sum(f1 * support) / support_sum)
    else:
        weighted_precision = 0.0
        weighted_recall = 0.0
        weighted_f1 = 0.0

    micro_precision = accuracy
    micro_recall = accuracy
    micro_f1 = accuracy

    overall_metrics = {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "balanced_accuracy": balanced_accuracy,
    }

    per_class_metrics = []
    for class_id in range(num_classes):
        per_class_metrics.append(
            {
                "class_id": class_id,
                "support": int(support[class_id]),
                "precision": float(precision[class_id]),
                "recall": float(recall[class_id]),
                "f1": float(f1[class_id]),
            }
        )

    return overall_metrics, per_class_metrics, confusion


def test(model, tensor_loader, criterion, device):
    model.eval()
    all_preds = []
    all_labels = []
    test_loss = 0.0
    num_classes = None

    with torch.no_grad():
        for data in tensor_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.long().to(device)

            outputs = model(inputs).float()
            if num_classes is None:
                num_classes = int(outputs.shape[1])

            loss = criterion(outputs, labels)
            predict_y = torch.argmax(outputs, dim=1)

            all_preds.append(predict_y.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
            test_loss += loss.item() * inputs.size(0)

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    test_loss = test_loss / len(tensor_loader.dataset)

    overall_metrics, per_class_metrics, confusion = compute_classification_metrics(
        y_true=y_true, y_pred=y_pred, num_classes=num_classes
    )
    print(
        "validation accuracy:{:.4f}, loss:{:.5f}".format(
            float(overall_metrics["accuracy"]), float(test_loss)
        )
    )
    return overall_metrics, per_class_metrics, confusion, float(test_loss)


def append_row_to_csv(file_path, row):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file_exists = os.path.exists(file_path)
    with open(file_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def write_rows_to_csv(file_path, rows):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", newline="") as f:
        if len(rows) == 0:
            return
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_matrix(file_path, confusion):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    num_classes = confusion.shape[0]
    header = ["true\\pred"] + [f"class_{i}" for i in range(num_classes)]
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(num_classes):
            writer.writerow([f"class_{i}"] + confusion[i].tolist())


def main():
    root = "./Data/"
    parser = argparse.ArgumentParser("WiFi Imaging Benchmark")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["UT_HAR_data", "NTU-Fi-HumanID", "NTU-Fi_HAR", "Widar"],
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "MLP",
            "LeNet",
            "ResNet18",
            "ResNet50",
            "ResNet101",
            "RNN",
            "GRU",
            "LSTM",
            "BiLSTM",
            "CNN+GRU",
            "ViT",
        ],
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Root directory for saved artifacts.",
    )
    parser.add_argument(
        "--checkpoint-name",
        default="model_checkpoint.pt",
        help="Checkpoint filename saved under results/<model>/<dataset>/.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    run_start = time.perf_counter()
    run_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    train_loader, test_loader, model, train_epoch = load_data_n_model(
        args.dataset, args.model, root
    )
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    total_params, trainable_params = count_parameters(model)
    non_trainable_params = total_params - trainable_params

    history, training_time_s = train(
        model=model,
        tensor_loader=train_loader,
        num_epochs=train_epoch,
        learning_rate=1e-3,
        criterion=criterion,
        device=device,
    )

    eval_start = time.perf_counter()
    overall_metrics, per_class_metrics, confusion, test_loss = test(
        model=model,
        tensor_loader=test_loader,
        criterion=criterion,
        device=device,
    )
    eval_time_s = time.perf_counter() - eval_start
    total_runtime_s = time.perf_counter() - run_start

    output_dir = os.path.join(args.results_dir, args.model, args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_path = os.path.join(output_dir, args.checkpoint_name)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": args.model,
            "dataset_name": args.dataset,
            "seed": args.seed,
            "train_epochs": train_epoch,
            "timestamp_utc": run_timestamp,
            "metrics": overall_metrics,
            "validation_loss": test_loss,
        },
        checkpoint_path,
    )

    final_train_accuracy = history[-1]["train_accuracy"] if len(history) > 0 else 0.0
    final_train_loss = history[-1]["train_loss"] if len(history) > 0 else 0.0

    summary_row = {
        "timestamp_utc": run_timestamp,
        "dataset": args.dataset,
        "model": args.model,
        "seed": args.seed,
        "device": str(device),
        "epochs": train_epoch,
        "final_train_accuracy": final_train_accuracy,
        "final_train_loss": final_train_loss,
        "validation_loss": test_loss,
        "validation_accuracy": overall_metrics["accuracy"],
        "macro_precision": overall_metrics["macro_precision"],
        "macro_recall": overall_metrics["macro_recall"],
        "macro_f1": overall_metrics["macro_f1"],
        "weighted_precision": overall_metrics["weighted_precision"],
        "weighted_recall": overall_metrics["weighted_recall"],
        "weighted_f1": overall_metrics["weighted_f1"],
        "micro_precision": overall_metrics["micro_precision"],
        "micro_recall": overall_metrics["micro_recall"],
        "micro_f1": overall_metrics["micro_f1"],
        "balanced_accuracy": overall_metrics["balanced_accuracy"],
        "training_time_s": training_time_s,
        "evaluation_time_s": eval_time_s,
        "total_runtime_s": total_runtime_s,
        "checkpoint_path": checkpoint_path,
        "output_dir": output_dir,
    }
    append_row_to_csv(os.path.join(output_dir, "results.csv"), summary_row)
    append_row_to_csv(os.path.join(args.results_dir, "benchmark_summary.csv"), summary_row)

    write_rows_to_csv(os.path.join(output_dir, "per_class_metrics.csv"), per_class_metrics)
    write_rows_to_csv(os.path.join(output_dir, "train_history.csv"), history)
    save_confusion_matrix(os.path.join(output_dir, "confusion_matrix.csv"), confusion)

    print("saved artifacts in: {}".format(output_dir))
    print("checkpoint: {}".format(checkpoint_path))
    return


if __name__ == "__main__":
    main()
