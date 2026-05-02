"""
CNN-LSTM Model for Sign Language Recognition
Replaces Transformer with CNN for local features + LSTM for sequence.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


PAD_ID = 0


class CNNLSTMEncoder(nn.Module):
    """
    CNN-LSTM encoder:
    - CNN1D: Extract local temporal features
    - LSTM: Capture long-range sequential dependencies
    """
    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        num_lstm_layers=2,
        dropout=0.3
    ):
        super().__init__()

        # CNN layers for temporal feature extraction
        self.cnn = nn.Sequential(
            # Conv1: (T, 516) → (T, 128)
            nn.Conv1d(input_dim, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Conv2: (T, 128) → (T, 256)
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Conv3: (T, 256) → (T, hidden_dim)
            nn.Conv1d(256, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Bidirectional LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )

        self.output_dim = hidden_dim * 2  # Bidirectional

    def forward(self, x, pad_mask=None):
        """
        x: (B, T, input_dim)
        Returns: (B, T, hidden_dim * 2)
        """
        B, T, D = x.shape

        # CNN expects (B, D, T)
        x = x.transpose(1, 2)

        # CNN forward
        x = self.cnn(x)

        # Back to (B, T, D)
        x = x.transpose(1, 2)

        # LSTM forward - skip packing to avoid zero-length issues
        lstm_out, (hidden, cell) = self.lstm(x)

        return lstm_out


class Seq2SeqCNN(nn.Module):
    """
    Seq2Seq model with CNN-LSTM encoder.
    """
    def __init__(
        self,
        input_dim,
        vocab_size,
        hidden_dim=256,
        use_ctc=False,
        num_lstm_layers=2
    ):
        super().__init__()

        self.use_ctc = use_ctc
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # CNN-LSTM Encoder
        self.encoder = CNNLSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_lstm_layers=num_lstm_layers
        )

        # Decoder (same as before)
        from model import Decoder
        self.decoder = Decoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim * 2  # Bidirectional
        )

        # CTC head
        if use_ctc:
            self.ctc_head = nn.Linear(hidden_dim * 2, vocab_size + 1)

    def generate_square_subsequent_mask(self, sz, device=None):
        if device is None:
            device = next(self.parameters()).device
        return torch.triu(
            torch.ones(sz, sz, device=device),
            diagonal=1
        ).bool()

    def forward(self, src, src_mask, tgt, use_ctc=None):

        src_pad_mask = (src_mask == 0)
        tgt_pad_mask = (tgt == PAD_ID)

        memory = self.encoder(src, src_pad_mask)

        tgt_mask = self.generate_square_subsequent_mask(
            tgt.size(1), tgt.device
        )

        logits, attn = self.decoder(
            tgt,
            memory,
            tgt_mask,
            tgt_pad_mask,
            src_pad_mask
        )

        ctc_logits = None
        should_use_ctc = self.use_ctc if use_ctc is None else use_ctc

        if should_use_ctc:
            ctc_logits = self.ctc_head(memory)
            ctc_logits = F.log_softmax(ctc_logits, dim=-1)

        return logits, ctc_logits, attn


if __name__ == "__main__":
    # Test
    model = Seq2SeqCNN(
        input_dim=516,
        vocab_size=184,
        hidden_dim=256,
        use_ctc=True
    )

    src = torch.randn(2, 100, 516)  # (B, T, features)
    src_mask = torch.ones(2, 100).bool()
    tgt = torch.randint(0, 184, (2, 50))

    logits, ctc_logits, attn = model(src, src_mask, tgt)

    print(f"Logits: {logits.shape}")       # (B, T, V)
    print(f"CTC: {ctc_logits.shape}")      # (B, T, V+1)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")