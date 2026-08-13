"""
Our topic-segmentation model: a BiLSTM boundary classifier over frozen sentence
embeddings, trained from scratch on QMSum.
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
