import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data_pipeline import y_encoded, vocab_size
from config import Config

cfg = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------
# DATASET
# ----------------------
class LMDataset(Dataset):
    def __init__(self, sequences):
        self.data = sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return torch.tensor(seq[:-1]), torch.tensor(seq[1:])


def collate(batch):
    xs, ys = zip(*batch)

    max_len = max(len(x) for x in xs)

    X = torch.full((len(xs), max_len), cfg.pad_id)
    Y = torch.full((len(xs), max_len), cfg.pad_id)

    for i in range(len(xs)):
        X[i, :len(xs[i])] = xs[i]
        Y[i, :len(ys[i])] = ys[i]

    return X, Y


loader = DataLoader(
    LMDataset(y_encoded),
    batch_size=32,
    shuffle=True,
    collate_fn=collate
)


# ----------------------
# MODEL
# ----------------------
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, hidden=512, num_layers=2):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, hidden)

        self.lstm = nn.LSTM(
            hidden,
            hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.lstm(x)
        return self.fc(out)
    
model = LanguageModel(vocab_size, hidden=512, num_layers=2).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)


# ----------------------
# TRAIN
# ----------------------
for epoch in range(10):
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        opt.zero_grad()

        logits = model(x)

        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=cfg.pad_id
        )

        loss.backward()
        opt.step()

        total += loss.item()

    print(f"Epoch {epoch} | LM Loss: {total / len(loader):.4f}")

torch.save(model.state_dict(), "checkpoints/lm.pt")
print("Saved LM → checkpoints/lm.pt")