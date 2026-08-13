"""
Our topic-segmentation model: a BiLSTM boundary classifier we train ourselves.

It takes a sequence of frozen sentence embeddings (one per turn) and outputs a
boundary logit for each turn. The pretrained encoder only gives us the input
features; the classifier's weights are trained from scratch on QMSum, so this
is genuinely our own model.
"""
import torch.nn as nn


class BiLSTMSegmenter(nn.Module):
    def __init__(self, emb_dim=384, hidden=128, num_layers=1, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            emb_dim, hidden, num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden, 1)

    def forward(self, x):
        """x: FloatTensor[B, T, emb_dim] -> boundary logits FloatTensor[B, T]."""
        h, _ = self.lstm(x)
        return self.head(self.drop(h)).squeeze(-1)
