"""
Training script for the ESP-Fi-HAR dataset.

Two evaluation strategies:
  loso          – Leave One Subject Out (8 subjects → 8 folds)
  leave-one-env – Leave One Environment Out (4 environments → 4 folds)

For each fold the model is trained on the complement split and evaluated on
the held-out split.  Results are saved following the same conventions as run.py:

  results/<model>/esp-fi-har-loso/subject_<n>/results.csv
  results/<model>/esp-fi-har-leave-env/env_<n>/results.csv

Usage examples
--------------
  python run_esp_fi_har.py --model GRU --strategy loso --held-out 1
  python run_esp_fi_har.py --model ResNet18 --strategy leave-one-env --held-out 2
"""

import argparse
import csv
import os
import random
import time
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn

from dataset import (
    get_esp_fi_har_files,
    ESP_Fi_HAR_Dataset,
    NORM_TYPES,
    compute_norm_stats,
    _parse_esp_fi_har_file,
)
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

# Reuse helpers from run.py
from run import (
    set_seed,
    count_parameters,
    test,
    append_row_to_csv,
    write_rows_to_csv,
    save_confusion_matrix,
)

NUM_CLASSES = 7


def train(model, tensor_loader, num_epochs, learning_rate, criterion, device,
          weight_decay=1e-4):
    """Training loop with weight decay and cosine LR annealing."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=learning_rate * 0.01)
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

        scheduler.step()
        epoch_loss /= len(tensor_loader.dataset)
        epoch_accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0.0
        epoch_time_s = time.perf_counter() - epoch_start
        history.append({
            'epoch': epoch + 1,
            'train_accuracy': float(epoch_accuracy),
            'train_loss': float(epoch_loss),
            'epoch_time_s': float(epoch_time_s),
        })
        print(f'Epoch [{epoch+1}/{num_epochs}]  '
              f'loss={epoch_loss:.4f}  acc={epoch_accuracy:.4f}  '
              f'lr={scheduler.get_last_lr()[0]:.2e}  '
              f'time={epoch_time_s:.1f}s', flush=True)

    return history, time.perf_counter() - train_start


def _stratified_split(all_files, test_size=0.2, seed=42):
    """Split files with proportional class representation in the test set."""
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for f in all_files:
        by_class[f['action_id']].append(f)
    train_files, test_files = [], []
    for cls_files in by_class.values():
        shuffled = cls_files[:]
        rng.shuffle(shuffled)
        n_test = max(1, round(len(shuffled) * test_size))
        test_files.extend(shuffled[:n_test])
        train_files.extend(shuffled[n_test:])
    return train_files, test_files

MODEL_REGISTRY = {
    'MLP':      lambda: ESP_Fi_HAR_MLP(NUM_CLASSES),
    'LeNet':    lambda: ESP_Fi_HAR_LeNet(NUM_CLASSES),
    'ResNet18': lambda: ESP_Fi_HAR_ResNet18(NUM_CLASSES),
    'ResNet50': lambda: ESP_Fi_HAR_ResNet50(NUM_CLASSES),
    'ResNet101':lambda: ESP_Fi_HAR_ResNet101(NUM_CLASSES),
    'RNN':      lambda: ESP_Fi_HAR_RNN(NUM_CLASSES),
    'GRU':      lambda: ESP_Fi_HAR_GRU(NUM_CLASSES),
    'LSTM':     lambda: ESP_Fi_HAR_LSTM(NUM_CLASSES),
    'BiLSTM':   lambda: ESP_Fi_HAR_BiLSTM(NUM_CLASSES),
    'CNN+GRU':  lambda: ESP_Fi_HAR_CNN_GRU(NUM_CLASSES),
    'ViT':      lambda: ESP_Fi_HAR_ViT(NUM_CLASSES),
}

# Number of training epochs per model.
# CNN/ResNet models already overfit at 50 — we keep epochs similar but rely on
# weight decay + cosine LR to regularize.  Recurrent models were badly
# underfitting so they get significantly more epochs to converge.
EPOCHS = {
    'MLP':      100,
    'LeNet':    100,
    'ResNet18': 100,
    'ResNet50': 100,
    'ResNet101':100,
    'RNN':      200,
    'GRU':      200,
    'LSTM':     200,
    'BiLSTM':   200,
    'CNN+GRU':  200,
    'ViT':      100,
}


def build_split(all_files, strategy, held_out):
    """Return (train_file_list, test_file_list) for the given strategy and fold.

    For 'random-split', held_out is used as the random seed so that multiple
    independent runs can be averaged.  The split is stratified by action_id so
    that every class is proportionally represented in both sets (test ≈ 20 %).
    """
    if strategy == 'loso':
        train_files = [f for f in all_files if f['person_id'] != held_out]
        test_files  = [f for f in all_files if f['person_id'] == held_out]
    elif strategy == 'leave-one-env':
        train_files = [f for f in all_files if f['env_id'] != held_out]
        test_files  = [f for f in all_files if f['env_id'] == held_out]
    elif strategy == 'random-split':
        train_files, test_files = _stratified_split(
            all_files, test_size=0.2, seed=held_out * 42)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return train_files, test_files


def fold_name(strategy, held_out):
    if strategy == 'loso':
        return f'subject_{held_out}'
    if strategy == 'random-split':
        return f'seed_{held_out}'
    return f'env_{held_out}'


def dataset_tag(strategy):
    if strategy == 'loso':
        return 'esp-fi-har-loso'
    if strategy == 'random-split':
        return 'esp-fi-har-random'
    return 'esp-fi-har-leave-env'


def main():
    data_root = os.path.join('.', 'Data', 'esp-fi-har')

    parser = argparse.ArgumentParser('ESP-Fi-HAR training')
    parser.add_argument('--model', required=True, choices=list(MODEL_REGISTRY))
    parser.add_argument('--strategy', required=True,
                        choices=['loso', 'leave-one-env', 'random-split'])
    parser.add_argument('--held-out', required=True, type=int,
                        help='Subject ID (1-8) for loso; env ID (1-4) for leave-one-env; '
                             'random seed index (e.g. 1-5) for random-split')
    parser.add_argument('--data-root', default=data_root)
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--checkpoint-name', default='model_checkpoint.pt')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--norm', default='per-sample', choices=NORM_TYPES,
                        help='Normalization strategy. Statistics are always computed '
                             'from the training split only to prevent data leakage.')
    args = parser.parse_args()

    set_seed(args.seed)

    run_start = time.perf_counter()
    run_timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    print(f'Loading ESP-Fi-HAR files from {args.data_root} …', flush=True)
    all_files = get_esp_fi_har_files(args.data_root)
    if not all_files:
        raise FileNotFoundError(f'No CSV files found under {args.data_root}')

    train_files, test_files = build_split(all_files, args.strategy, args.held_out)
    print(f'Strategy={args.strategy}  held-out={args.held_out}  '
          f'train={len(train_files)}  test={len(test_files)}', flush=True)

    print(f'Building datasets (norm={args.norm}) …', flush=True)
    # Load raw training data first so we can compute stats from training split only
    train_raw = np.array(
        [_parse_esp_fi_har_file(f['path'], normalize=False) for f in train_files],
        dtype=np.float32,
    )
    norm_stats = compute_norm_stats(train_raw, args.norm, train_files)
    # Both datasets receive the same training-derived stats → no data leakage
    train_set = ESP_Fi_HAR_Dataset(train_files, norm_type=args.norm, norm_stats=norm_stats)
    test_set  = ESP_Fi_HAR_Dataset(test_files,  norm_type=args.norm, norm_stats=norm_stats)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=64, shuffle=True, drop_last=True)
    test_loader  = torch.utils.data.DataLoader(
        test_set,  batch_size=64, shuffle=False)

    model = MODEL_REGISTRY[args.model]()
    train_epoch = EPOCHS[args.model]

    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    total_params, trainable_params = count_parameters(model)

    print(f'Model={args.model}  device={device}  '
          f'params={total_params:,}  epochs={train_epoch}', flush=True)

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

    ds_tag = dataset_tag(args.strategy)
    fold   = fold_name(args.strategy, args.held_out)
    # Include norm type in directory so different normalizations don't overwrite each other
    norm_tag = args.norm.replace('-', '_')
    output_dir = os.path.join(args.results_dir, args.model, ds_tag, norm_tag, fold)
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_path = os.path.join(output_dir, args.checkpoint_name)
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'model_name': args.model,
            'dataset_name': 'esp-fi-har',
            'strategy': args.strategy,
            'held_out': args.held_out,
            'norm': args.norm,
            'seed': args.seed,
            'train_epochs': train_epoch,
            'timestamp_utc': run_timestamp,
            'metrics': overall_metrics,
            'validation_loss': test_loss,
        },
        checkpoint_path,
    )

    final_train_accuracy = history[-1]['train_accuracy'] if history else 0.0
    final_train_loss     = history[-1]['train_loss']     if history else 0.0

    summary_row = {
        'timestamp_utc': run_timestamp,
        'dataset': 'esp-fi-har',
        'strategy': args.strategy,
        'held_out': args.held_out,
        'model': args.model,
        'norm': args.norm,
        'seed': args.seed,
        'device': str(device),
        'epochs': train_epoch,
        'train_samples': len(train_set),
        'test_samples': len(test_set),
        'final_train_accuracy': final_train_accuracy,
        'final_train_loss': final_train_loss,
        'validation_loss': test_loss,
        'validation_accuracy': overall_metrics['accuracy'],
        'macro_precision': overall_metrics['macro_precision'],
        'macro_recall': overall_metrics['macro_recall'],
        'macro_f1': overall_metrics['macro_f1'],
        'weighted_precision': overall_metrics['weighted_precision'],
        'weighted_recall': overall_metrics['weighted_recall'],
        'weighted_f1': overall_metrics['weighted_f1'],
        'micro_precision': overall_metrics['micro_precision'],
        'micro_recall': overall_metrics['micro_recall'],
        'micro_f1': overall_metrics['micro_f1'],
        'balanced_accuracy': overall_metrics['balanced_accuracy'],
        'training_time_s': training_time_s,
        'evaluation_time_s': eval_time_s,
        'total_runtime_s': total_runtime_s,
        'checkpoint_path': checkpoint_path,
        'output_dir': output_dir,
    }

    append_row_to_csv(os.path.join(output_dir, 'results.csv'), summary_row)
    # Master comparison file — every norm × model × strategy fold in one place
    append_row_to_csv(
        os.path.join(args.results_dir, 'esp_fi_har_norm_comparison.csv'), summary_row)
    # Strategy-specific file for convenience
    strategy_csv = f'esp_fi_har_{args.strategy.replace("-", "_")}_norm_comparison.csv'
    append_row_to_csv(
        os.path.join(args.results_dir, strategy_csv), summary_row)

    write_rows_to_csv(
        os.path.join(output_dir, 'per_class_metrics.csv'), per_class_metrics)
    write_rows_to_csv(
        os.path.join(output_dir, 'train_history.csv'), history)
    save_confusion_matrix(
        os.path.join(output_dir, 'confusion_matrix.csv'), confusion)

    print(f'Saved artifacts in: {output_dir}', flush=True)
    print(f'Checkpoint: {checkpoint_path}', flush=True)


if __name__ == '__main__':
    main()
