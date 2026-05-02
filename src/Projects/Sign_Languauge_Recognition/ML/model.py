import torch
import torch.nn as nn
import torch.nn.functional as F


PAD_ID = 0


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pe, std=0.02)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.pos = LearnablePositionalEncoding(hidden_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(layer, num_layers=6)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, pad_mask):
        x = self.proj(x)
        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        return self.norm(x)


class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.2):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True
        )

        self.cross_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True
        )

        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask,
        tgt_key_padding_mask,
        memory_key_padding_mask
    ):

        # Self-attention
        _tgt, _ = self.self_attn(
            tgt,
            tgt,
            tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )

        tgt = self.norm1(tgt + self.dropout1(_tgt))

        # Cross-attention
        _tgt, cross_attn_weights = self.cross_attn(
            tgt,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )

        tgt = self.norm2(tgt + self.dropout2(_tgt))

        # Feed-forward
        ff = self.linear2(
            self.dropout(
                F.gelu(
                    self.linear1(tgt)
                )
            )
        )

        tgt = self.norm3(tgt + self.dropout3(ff))

        return tgt, cross_attn_weights


class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, num_layers=3):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos = LearnablePositionalEncoding(hidden_dim)

        self.layers = nn.ModuleList([
            CustomTransformerDecoderLayer(hidden_dim, 8)
            for _ in range(num_layers)
        ])

        # IMPORTANT:
        # Decoder predicts only real vocabulary tokens.
        # CTC blank is handled separately by ctc_head.
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask,
        tgt_pad_mask,
        src_pad_mask
    ):

        x = self.embed(tgt)
        x = self.pos(x)

        attn = None

        for layer in self.layers:
            x, attn = layer(
                x,
                memory,
                tgt_mask,
                tgt_pad_mask,
                src_pad_mask
            )

        logits = self.fc(x)

        return logits, attn


class Seq2Seq(nn.Module):
    def __init__(
        self,
        input_dim,
        vocab_size,
        hidden_dim=256,
        use_ctc=False
    ):
        super().__init__()

        self.use_ctc = use_ctc
        self.vocab_size = vocab_size

        self.encoder = Encoder(input_dim, hidden_dim)

        self.decoder = Decoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim
        )

        if use_ctc:
            self.ctc_head = nn.Linear(hidden_dim, vocab_size + 1)

    def generate_square_subsequent_mask(self, sz, device=None):
        if device is None:
            device = next(self.parameters()).device

        return torch.triu(
            torch.ones(sz, sz, device=device),
            diagonal=1
        ).bool()

    def forward(self, src, src_mask, tgt, use_ctc=None):

        # src_mask from data_pipeline:
        # True = valid frame
        # False = padding
        #
        # PyTorch expects:
        # True = padding
        src_pad_mask = (src_mask == 0)

        tgt_pad_mask = (tgt == PAD_ID)

        memory = self.encoder(src, src_pad_mask)

        tgt_mask = self.generate_square_subsequent_mask(
            tgt.size(1),
            tgt.device
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