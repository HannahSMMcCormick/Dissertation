import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Decoder

PAD_ID = 0


class CNNLSTMEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        num_lstm_layers=2,
        dropout=0.3
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(256, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
        )

        self.output_dim = hidden_dim * 2

    def forward(self, x, pad_mask=None):
        # x: [B, T, D]
        x = x.transpose(1, 2)      # [B, D, T]
        x = self.cnn(x)
        x = x.transpose(1, 2)      # [B, T, H]

        x, _ = self.lstm(x)

        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        return x


class CNNLSTMSeq2Seq(nn.Module):
    def __init__(
        self,
        input_dim,
        vocab_size,
        hidden_dim=256,
        num_lstm_layers=2
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        self.encoder = CNNLSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_lstm_layers=num_lstm_layers,
        )

        self.decoder = Decoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim * 2,
        )

    def generate_square_subsequent_mask(self, sz, device=None):
        if device is None:
            device = next(self.parameters()).device

        return torch.triu(
            torch.ones(sz, sz, device=device),
            diagonal=1,
        ).bool()

    def forward(self, src, src_mask, tgt):
        # src_mask from data_pipeline:
        # True = valid frame
        # False = padding
        src_pad_mask = src_mask == 0

        tgt_pad_mask = tgt == PAD_ID

        memory = self.encoder(src, src_pad_mask)

        tgt_mask = self.generate_square_subsequent_mask(
            tgt.size(1),
            tgt.device,
        )

        logits, attn = self.decoder(
            tgt,
            memory,
            tgt_mask,
            tgt_pad_mask,
            src_pad_mask,
        )

        return logits, None, attn