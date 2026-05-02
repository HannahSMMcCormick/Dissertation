import os
import torch

from model_cnn_seq2seq import CNNLSTMSeq2Seq
from losses import total_loss
from config import Config
from data_pipeline import train_loader, val_loader, vocab_size, input_dim

cfg = Config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg.vocab_size = vocab_size
cfg.use_ctc = False
cfg.use_coverage = False

model = CNNLSTMSeq2Seq(
    input_dim=input_dim,
    vocab_size=cfg.vocab_size,
    hidden_dim=cfg.hidden_dim,
    num_lstm_layers=2,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

os.makedirs("checkpoints", exist_ok=True)

best_val_loss = float("inf")
patience = 10
bad_epochs = 0

for epoch in range(cfg.max_epochs):
    model.train()
    train_loss = 0.0

    for src, src_mask, tgt in train_loader:
        src = src.to(device)
        src_mask = src_mask.to(device)
        tgt = tgt.to(device)

        optimizer.zero_grad()

        inp = tgt[:, :-1]
        gold = tgt[:, 1:]

        outputs = model(src, src_mask, inp)
        loss = total_loss(outputs, (src, src_mask, gold), cfg)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss += loss.item()

    train_loss /= max(1, len(train_loader))

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for src, src_mask, tgt in val_loader:
            src = src.to(device)
            src_mask = src_mask.to(device)
            tgt = tgt.to(device)

            inp = tgt[:, :-1]
            gold = tgt[:, 1:]

            outputs = model(src, src_mask, inp)
            loss = total_loss(outputs, (src, src_mask, gold), cfg)

            val_loss += loss.item()

    val_loss /= max(1, len(val_loader))

    print(
        f"Epoch {epoch} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        bad_epochs = 0

        torch.save(
            {
                "epoch": epoch,
                "val_loss": val_loss,
                "model_state": model.state_dict(),
                "config": {
                    "input_dim": input_dim,
                    "vocab_size": cfg.vocab_size,
                    "hidden_dim": cfg.hidden_dim,
                    "model_type": "cnn_lstm_seq2seq",
                    "max_target_tokens": 12,
                    "features": "position+velocity+acceleration",
                },
            },
            "checkpoints/cnn_seq2seq_best.pt",
        )

        print("Saved checkpoints/cnn_seq2seq_best.pt")
    else:
        bad_epochs += 1

    if bad_epochs >= patience:
        print("Early stopping")
        break