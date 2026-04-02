import os
import re
from collections import defaultdict
import numpy as np
import glob
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader


def UT_HAR_dataset(root_dir):
    data_list = glob.glob(root_dir+'/UT_HAR/data/*.csv')
    label_list = glob.glob(root_dir+'/UT_HAR/label/*.csv')
    WiFi_data = {}
    for data_dir in data_list:
        data_name = data_dir.split('/')[-1].split('.')[0]
        with open(data_dir, 'rb') as f:
            data = np.load(f)
            data = data.reshape(len(data),1,250,90)
            data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        WiFi_data[data_name] = torch.Tensor(data_norm)
    for label_dir in label_list:
        label_name = label_dir.split('/')[-1].split('.')[0]
        with open(label_dir, 'rb') as f:
            label = np.load(f)
        WiFi_data[label_name] = torch.Tensor(label)
    return WiFi_data


# dataset: /class_name/xx.mat
class CSI_Dataset(Dataset):
    """CSI dataset."""

    def __init__(self, root_dir, modal='CSIamp', transform=None, few_shot=False, k=5, single_trace=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            modal (CSIamp/CSIphase): CSI data modal
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.modal=modal
        self.transform = transform
        self.data_list = glob.glob(root_dir+'/*/*.mat')
        self.folder = glob.glob(root_dir+'/*/')
        self.category = {self.folder[i].split('/')[-2]:i for i in range(len(self.folder))}

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample_dir = self.data_list[idx]
        y = self.category[sample_dir.split('/')[-2]]
        x = sio.loadmat(sample_dir)[self.modal]
        
        # normalize
        x = (x - 42.3199)/4.9802
        
        # sampling: 2000 -> 500
        x = x[:,::4]
        x = x.reshape(3, 114, 500)
        
        if self.transform:
            x = self.transform(x)
        
        x = torch.FloatTensor(x)

        return x,y


class Widar_Dataset(Dataset):
    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.data_list = glob.glob(root_dir+'/*/*.csv')
        self.folder = glob.glob(root_dir+'/*/')
        self.category = {self.folder[i].split('/')[-2]:i for i in range(len(self.folder))}

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_dir = self.data_list[idx]
        y = self.category[sample_dir.split('/')[-2]]
        x = np.genfromtxt(sample_dir, delimiter=',')

        # normalize
        x = (x - 0.0025)/0.0119

        # reshape: 22,400 -> 22,20,20
        x = x.reshape(22,20,20)
        # interpolate from 20x20 to 32x32
        # x = self.reshape(x)
        x = torch.FloatTensor(x)

        return x,y


def _parse_esp_fi_har_file(file_path, n_timesteps=500, normalize=True):
    """Load a single ESP-Fi-HAR CSV file and return amplitude array (1, 52, n_timesteps).

    If normalize=False the raw amplitude values are returned without any scaling.
    """
    iq_rows = []
    with open(file_path, 'r') as f:
        next(f)  # skip header
        for line in f:
            m = re.search(r'\[([^\]]+)\]', line)
            if m:
                nums = np.array(m.group(1).split(', '), dtype=np.float32)
                if len(nums) == 104:
                    iq_rows.append(nums)

    if not iq_rows:
        return np.zeros((1, 52, n_timesteps), dtype=np.float32)

    iq = np.stack(iq_rows)          # (N, 104)
    I = iq[:, 0::2]                  # (N, 52)
    Q = iq[:, 1::2]                  # (N, 52)
    amplitude = np.sqrt(I ** 2 + Q ** 2)  # (N, 52)

    N = len(amplitude)
    if N >= n_timesteps:
        indices = np.linspace(0, N - 1, n_timesteps, dtype=int)
        amplitude = amplitude[indices]       # (n_timesteps, 52)
    else:
        pad = np.zeros((n_timesteps - N, 52), dtype=np.float32)
        amplitude = np.concatenate([amplitude, pad], axis=0)

    amplitude = amplitude.T          # (52, n_timesteps)

    if normalize:
        amp_min, amp_max = amplitude.min(), amplitude.max()
        if amp_max > amp_min:
            amplitude = (amplitude - amp_min) / (amp_max - amp_min)

    return amplitude[np.newaxis, :]  # (1, 52, n_timesteps)


# ---------------------------------------------------------------------------
# Normalization utilities — all statistics computed from training data only
# ---------------------------------------------------------------------------

NORM_TYPES = [
    'per-sample',            # current default: normalize each recording independently
    'global',                # global min-max over training set
    'global-zscore',         # global z-score over training set
    'per-subcarrier',        # per-subcarrier min-max over training set
    'per-subcarrier-zscore', # per-subcarrier z-score over training set
    'per-subject',           # per-subject min-max (fallback to global for unseen subjects)
    'per-environment',       # per-environment min-max (fallback to global for unseen envs)
    'robust',                # 1st–99th percentile min-max over training set
]

_EPS = 1e-8


def compute_norm_stats(raw_data, norm_type, file_metadata=None):
    """Compute normalization statistics from training data only.

    Parameters
    ----------
    raw_data      : (N, 1, 52, T) float32 numpy array — raw amplitude, no normalization
    norm_type     : one of NORM_TYPES
    file_metadata : list of dicts with 'person_id' and 'env_id' keys
                    (required for 'per-subject' and 'per-environment')

    Returns
    -------
    dict of statistics to be passed to apply_normalization()
    """
    data = raw_data[:, 0, :, :]  # (N, 52, T)

    if norm_type == 'per-sample':
        return {}

    if norm_type == 'global':
        return {'min': float(data.min()), 'max': float(data.max())}

    if norm_type == 'global-zscore':
        return {'mean': float(data.mean()), 'std': float(data.std())}

    if norm_type == 'per-subcarrier':
        # aggregate over samples (axis 0) and time (axis 2) → shape (52,)
        return {
            'min': data.min(axis=(0, 2)).astype(np.float32),
            'max': data.max(axis=(0, 2)).astype(np.float32),
        }

    if norm_type == 'per-subcarrier-zscore':
        return {
            'mean': data.mean(axis=(0, 2)).astype(np.float32),
            'std':  data.std(axis=(0, 2)).astype(np.float32),
        }

    if norm_type in ('per-subject', 'per-environment'):
        assert file_metadata is not None, f"file_metadata required for {norm_type}"
        key = 'person_id' if norm_type == 'per-subject' else 'env_id'
        groups = defaultdict(list)
        for i, meta in enumerate(file_metadata):
            groups[meta[key]].append(i)
        stats = {}
        for group_id, indices in groups.items():
            g = data[indices]  # (n, 52, T)
            stats[group_id] = {'min': float(g.min()), 'max': float(g.max())}
        # Global fallback for held-out groups not seen during training
        stats['_global'] = {'min': float(data.min()), 'max': float(data.max())}
        return stats

    if norm_type == 'robust':
        return {
            'p1':  float(np.percentile(data,  1)),
            'p99': float(np.percentile(data, 99)),
        }

    raise ValueError(f"Unknown norm_type: {norm_type!r}. Choose from {NORM_TYPES}")


def apply_normalization(raw_data, norm_type, stats, file_metadata=None):
    """Apply pre-computed normalization statistics to a data array.

    Parameters
    ----------
    raw_data      : (N, 1, 52, T) float32 numpy array
    norm_type     : one of NORM_TYPES
    stats         : dict returned by compute_norm_stats()
    file_metadata : list of dicts (required for 'per-subject' / 'per-environment')

    Returns
    -------
    Normalized copy, same shape as raw_data.
    """
    data = raw_data.copy()

    if norm_type == 'per-sample':
        for i in range(len(data)):
            s = data[i, 0]  # (52, T)
            s_min, s_max = s.min(), s.max()
            if s_max - s_min > _EPS:
                data[i, 0] = (s - s_min) / (s_max - s_min)
        return data

    if norm_type == 'global':
        g_min, g_max = stats['min'], stats['max']
        denom = g_max - g_min
        if denom > _EPS:
            data = (data - g_min) / denom
        return np.clip(data, 0.0, 1.0)

    if norm_type == 'global-zscore':
        return (data - stats['mean']) / (stats['std'] + _EPS)

    if norm_type == 'per-subcarrier':
        # stats['min'/'max'] shape: (52,) — broadcast over batch and time
        s_min = stats['min'][np.newaxis, np.newaxis, :, np.newaxis]  # (1,1,52,1)
        s_max = stats['max'][np.newaxis, np.newaxis, :, np.newaxis]
        denom = np.where(s_max - s_min > _EPS, s_max - s_min, 1.0)
        return np.clip((data - s_min) / denom, 0.0, 1.0)

    if norm_type == 'per-subcarrier-zscore':
        s_mean = stats['mean'][np.newaxis, np.newaxis, :, np.newaxis]
        s_std  = stats['std'][np.newaxis, np.newaxis, :, np.newaxis]
        return (data - s_mean) / (s_std + _EPS)

    if norm_type in ('per-subject', 'per-environment'):
        assert file_metadata is not None, f"file_metadata required for {norm_type}"
        key = 'person_id' if norm_type == 'per-subject' else 'env_id'
        for i, meta in enumerate(file_metadata):
            group_id = meta[key]
            s = stats.get(group_id, stats['_global'])
            denom = s['max'] - s['min']
            if denom > _EPS:
                data[i, 0] = np.clip(
                    (data[i, 0] - s['min']) / denom, 0.0, 1.0)
        return data

    if norm_type == 'robust':
        p1, p99 = stats['p1'], stats['p99']
        denom = p99 - p1
        if denom > _EPS:
            data = (data - p1) / denom
        return np.clip(data, 0.0, 1.0)

    raise ValueError(f"Unknown norm_type: {norm_type!r}")


def get_esp_fi_har_files(data_root):
    """Return list of dicts with path, env_id, person_id, action_id for all ESP-Fi-HAR CSVs."""
    files = []
    for env_dir in sorted(glob.glob(os.path.join(data_root, 'EnvironmentNo.*'))):
        csv_dir = os.path.join(env_dir, 'csv')
        for f in sorted(glob.glob(os.path.join(csv_dir, '*.csv'))):
            fname = os.path.splitext(os.path.basename(f))[0]
            parts = fname.split('-')
            files.append({
                'path': f,
                'env_id': int(parts[0]),
                'person_id': int(parts[1]),
                'action_id': int(parts[2]) - 1,  # 0-indexed (0-6)
            })
    return files


class ESP_Fi_HAR_Dataset(Dataset):
    """
    ESP-Fi-HAR dataset. Accepts a list of file metadata dicts (from get_esp_fi_har_files).
    Each sample is loaded from a CSV file and returned as (1, 52, 500) amplitude tensor.

    Parameters
    ----------
    file_list   : list of dicts from get_esp_fi_har_files()
    n_timesteps : number of time steps to resample each recording to
    norm_type   : normalization strategy (one of NORM_TYPES, default 'per-sample')
    norm_stats  : pre-computed stats dict from compute_norm_stats() applied to the
                  *training* set.  Must be provided for all norm_types except
                  'per-sample'.  Pass the same stats object to both the train and
                  test dataset to avoid data leakage.
    """

    def __init__(self, file_list, n_timesteps=500,
                 norm_type='per-sample', norm_stats=None):
        raw = []
        self.labels = []
        for item in file_list:
            # Always load raw amplitude — normalization applied below
            raw.append(_parse_esp_fi_har_file(item['path'], n_timesteps, normalize=False))
            self.labels.append(item['action_id'])
        raw = np.array(raw, dtype=np.float32)  # (N, 1, 52, n_timesteps)

        if norm_stats is None:
            if norm_type != 'per-sample':
                raise ValueError(
                    f"norm_stats must be provided for norm_type={norm_type!r}. "
                    "Compute it from the training set with compute_norm_stats() "
                    "and pass the same object to both train and test datasets."
                )
            norm_stats = {}

        self.data = apply_normalization(raw, norm_type, norm_stats, file_list)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), self.labels[idx]

