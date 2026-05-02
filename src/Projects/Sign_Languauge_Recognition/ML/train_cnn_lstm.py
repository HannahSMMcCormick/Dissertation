"""
CNN-LSTM Training Script
"""
import os
import torch
import torch.nn.functional as F

from model_cnn_lstm import Seq2SeqCNN
from config import Config
from data_pipeline import train_loader, val_loader, vocab_size

cfg = Config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_DIM = 516
cfg.vocab_size = vocab_size

# CNN-LSTM model
model = Seq2SeqCNN(
    input_dim=INPUT_DIM,
    vocab_size=cfg.vocab_size,
    hidden_dim=256,
    use_ctc=True,
    num_lstm_layers=2
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

os.makedirs("checkpoints", exist_ok=True)

best_val_loss = float("inf")


def prepare_ctc_targets(tgt, cfg):
    targets = []
    lengths = []

    for seq in tgt:
        clean = [
            int(t.item()) for t in seq
            if int(t.item()) not in (cfg.sos_id, cfg.eos_id, cfg.pad_id)
        ]
        if len(clean) > 0:
            targets.extend(clean)
            lengths.append(len(clean))
        else:
            lengths.append(0)

    return (
        torch.tensor(targets, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long)
    )


for epoch in range(cfg.max_epochs):
    model.train()
    train_loss = 0.0

    for src, src_mask, tgt in train_loader:
        src = src.to(device)
        src_mask = src_mask.to(device)
        tgt = tgt.to(device)

        optimizer.zero_grad()

        # Forward pass
        memory = model.encoder(src, src_mask == 0)
        ctc_logits = model.ctc_head(memory)
        ctc_logits = F.log_softmax(ctc_logits, dim=-1)

        # CTC loss
        targets, target_lengths = prepare_ctc_targets(tgt, cfg)
        input_lengths = src_mask.sum(dim=1).long()

        if targets.numel() == 0:
            continue

        loss = F.ctc_loss(
            ctc_logits.transpose(0, 1),
            targets,
            input_lengths,
            target_lengths,
            blank=cfg.vocab_size,
            zero_infinity=True
        )

        if loss.isfinite():
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for src, src_mask, tgt in val_loader:
            src = src.to(device)
            src_mask = src_mask.to(device)
            tgt = tgt.to(device)

            memory = model.encoder(src, src_mask == 0)
            ctc_logits = model.ctc_head(memory)
            ctc_logits = F.log_softmax(ctc_logits, dim=-1)

            targets, target_lengths = prepare_ctc_targets(tgt, cfg)
            input_lengths = src_mask.sum(dim=1).long()

            if targets.numel() == 0:
                continue

            loss = F.ctc_loss(
                ctc_logits.transpose(0, 1),
                targets,
                input_lengths,
                target_lengths,
                blank=cfg.vocab_size,
                zero_infinity=True
            )

            if loss.isfinite():
                val_loss += loss.item()

    val_loss /= max(1, len(val_loader))

    print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss and val_loss > 0:
        best_val_loss = val_loss
        torch.save({
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state": model.state_dict(),
            "config": {
                "input_dim": INPUT_DIM,
                "vocab_size": cfg.vocab_size,
                "hidden_dim": 256,
            }
        }, "checkpoints/cnn_lstm_best.pt")
        print("Saved checkpoints/cnn_lstm_best.pt")