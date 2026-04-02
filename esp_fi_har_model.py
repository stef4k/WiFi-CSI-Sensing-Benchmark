"""
Deep learning models for the ESP-Fi-HAR dataset.

Input shape : (batch, 1, 52, 500)
  - 1 channel (amplitude)
  - 52 subcarriers
  - 500 time steps (uniformly downsampled from raw recordings)

num_classes : 7  (actions 1-7, stored as 0-indexed labels)

Transformer building blocks (MultiHeadAttention, TransformerEncoder, etc.)
are imported from NTU_Fi_model to avoid duplication.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from einops.layers.torch import Rearrange, Reduce

from NTU_Fi_model import (
    Block,
    Bottleneck,
    MultiHeadAttention,
    ResidualAdd,
    FeedForwardBlock,
    TransformerEncoderBlock,
    TransformerEncoder,
    ClassificationHead,
)

NUM_CLASSES = 7
_N_SUB = 52    # subcarriers
_N_TIME = 500  # time steps after downsampling


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class ESP_Fi_HAR_MLP(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(_N_SUB * _N_TIME, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.view(-1, _N_SUB * _N_TIME)
        return self.classifier(self.fc(x))


# ---------------------------------------------------------------------------
# LeNet-style CNN
# ---------------------------------------------------------------------------

class ESP_Fi_HAR_LeNet(nn.Module):
    """
    Input (1, 52, 500):
      Conv(1,32,(5,11),(2,4))  -> (32, 24, 123)
      Conv(32,64,3,2)          -> (64, 11, 61)
      Conv(64,96,3,2)          -> (96, 5, 30)
      AdaptiveAvgPool(4,4)     -> (96, 4, 4)
      Linear(1536,128) -> Linear(128, num_classes)
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, (5, 11), stride=(2, 4)),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 96, 3, stride=2),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Sequential(
            nn.Linear(96 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 96 * 4 * 4)
        return self.fc(x)


# ---------------------------------------------------------------------------
# ResNet family
# ---------------------------------------------------------------------------

class ESP_Fi_HAR_ResNet(nn.Module):
    """
    Reshape input (1, 52, 500) -> (3, 64, 64) via 1x1 conv + AdaptiveAvgPool,
    then standard ResNet backbone.
    """

    def __init__(self, ResBlock, layer_list, num_classes=NUM_CLASSES):
        super().__init__()
        self.reshape = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.AdaptiveAvgPool2d((64, 64)),
        )
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(512 * ResBlock.expansion, num_classes)

    def forward(self, x):
        x = self.reshape(x)
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        return self.fc(x)

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * ResBlock.expansion,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * ResBlock.expansion),
            )
        layers.append(ResBlock(self.in_channels, planes,
                               i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes * ResBlock.expansion
        for _ in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes))
        return nn.Sequential(*layers)


def ESP_Fi_HAR_ResNet18(num_classes=NUM_CLASSES):
    return ESP_Fi_HAR_ResNet(Block, [2, 2, 2, 2], num_classes)

def ESP_Fi_HAR_ResNet50(num_classes=NUM_CLASSES):
    return ESP_Fi_HAR_ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

def ESP_Fi_HAR_ResNet101(num_classes=NUM_CLASSES):
    return ESP_Fi_HAR_ResNet(Bottleneck, [3, 4, 23, 3], num_classes)


# ---------------------------------------------------------------------------
# Recurrent models: input (batch, 1, 52, 500) -> view (batch, 52, 500)
#                   -> permute (500, batch, 52)  [seq, batch, features]
# ---------------------------------------------------------------------------

class ESP_Fi_HAR_RNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.rnn = nn.RNN(_N_SUB, 128, num_layers=2, dropout=0.3)
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, num_classes))

    def forward(self, x):
        x = x.view(-1, _N_SUB, _N_TIME).permute(2, 0, 1)  # (500, batch, 52)
        _, ht = self.rnn(x)
        return self.fc(ht[-1])


class ESP_Fi_HAR_GRU(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.gru = nn.GRU(_N_SUB, 128, num_layers=2, dropout=0.3)
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, num_classes))

    def forward(self, x):
        x = x.view(-1, _N_SUB, _N_TIME).permute(2, 0, 1)
        _, ht = self.gru(x)
        return self.fc(ht[-1])


class ESP_Fi_HAR_LSTM(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.lstm = nn.LSTM(_N_SUB, 128, num_layers=2, dropout=0.3)
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, num_classes))

    def forward(self, x):
        x = x.view(-1, _N_SUB, _N_TIME).permute(2, 0, 1)
        _, (ht, _) = self.lstm(x)
        return self.fc(ht[-1])


class ESP_Fi_HAR_BiLSTM(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.lstm = nn.LSTM(_N_SUB, 128, num_layers=2, dropout=0.3, bidirectional=True)
        # ht shape: (num_layers*2, batch, hidden) — concatenate last forward + backward
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(128 * 2, num_classes))

    def forward(self, x):
        x = x.view(-1, _N_SUB, _N_TIME).permute(2, 0, 1)
        _, (ht, _) = self.lstm(x)
        # ht[-2]: last layer forward; ht[-1]: last layer backward
        x = torch.cat([ht[-2], ht[-1]], dim=1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# CNN + GRU hybrid
# ---------------------------------------------------------------------------

class ESP_Fi_HAR_CNN_GRU(nn.Module):
    """
    Per-timestep 1-D CNN encoder followed by a GRU.

    (batch, 1, 52, 500)
      -> view  (batch, 52, 500)
      -> permute (batch, 500, 52)
      -> reshape (batch*500, 1, 52)
      -> Conv1d encoder  -> (batch*500, 32, 10)
      -> permute (batch*500, 10, 32) -> AvgPool1d(32) -> (batch*500, 10, 1)
      -> reshape (batch, 500, 10) -> permute (500, batch, 10)
      -> GRU(10, 128) -> classifier(128, num_classes)
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        # Input per step: (batch*500, 1, 52)
        # Conv1d(1,16,5,2): (52-5)//2+1 = 24
        # MaxPool1d(2):     12
        # Conv1d(16,32,3,1): (12-3)//1+1 = 10
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 5, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 3, stride=1),
            nn.ReLU(),
        )
        self.mean = nn.AvgPool1d(32)   # (batch*500, 10, 32) -> (batch*500, 10, 1)
        self.gru = nn.GRU(10, 128, num_layers=2, dropout=0.3)
        # Softmax removed: CrossEntropyLoss applies log-softmax internally
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, _N_SUB, _N_TIME)          # (batch, 52, 500)
        x = x.permute(0, 2, 1)                            # (batch, 500, 52)
        x = x.reshape(batch_size * _N_TIME, 1, _N_SUB)   # (batch*500, 1, 52)
        x = self.encoder(x)                               # (batch*500, 32, 10)
        x = x.permute(0, 2, 1)                            # (batch*500, 10, 32)
        x = self.mean(x)                                  # (batch*500, 10, 1)
        x = x.reshape(batch_size, _N_TIME, 10)            # (batch, 500, 10)
        x = x.permute(1, 0, 2)                            # (500, batch, 10)
        _, ht = self.gru(x)
        return self.classifier(ht[-1])


# ---------------------------------------------------------------------------
# Vision Transformer (ViT)
# ---------------------------------------------------------------------------

class ESP_Fi_HAR_PatchEmbedding(nn.Module):
    """
    Divide (1, 52, 500) into non-overlapping patches of size (4, 25).
    Number of patches: (52//4) * (500//25) = 13 * 20 = 260.
    Embedding size: 4 * 25 = 100.
    """

    def __init__(self, in_channels=1, patch_h=4, patch_w=25, emb_size=100):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size,
                      kernel_size=(patch_h, patch_w),
                      stride=(patch_h, patch_w)),
            Rearrange('b e h w -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        num_patches = (_N_SUB // patch_h) * (_N_TIME // patch_w)  # 260
        self.position = nn.Parameter(torch.randn(num_patches + 1, emb_size))

    def forward(self, x):
        # x: (batch, 1, 52, 500)
        b = x.size(0)
        x = self.projection(x)                              # (batch, 260, 100)
        cls = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls, x], dim=1)                     # (batch, 261, 100)
        x = x + self.position
        return x


class ESP_Fi_HAR_ViT(nn.Sequential):
    def __init__(self, num_classes=NUM_CLASSES, depth=1, num_heads=5):
        emb_size = 100
        super().__init__(
            ESP_Fi_HAR_PatchEmbedding(in_channels=1, patch_h=4, patch_w=25,
                                      emb_size=emb_size),
            TransformerEncoder(depth, emb_size=emb_size, num_heads=num_heads),
            ClassificationHead(emb_size, num_classes),
        )
