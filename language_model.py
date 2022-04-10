import numpy as np
from segtok import tokenizer
import torch as th
from torch import nn

# Using a basic RNN/LSTM for Language modeling
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, rnn_size, num_layers=1, dropout=0):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, rnn_size)
        self.lstm = nn.LSTM(
            rnn_size, 
            rnn_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(rnn_size, vocab_size)
        self.embedding.weight = self.output.weight

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.output(x)
        return x
