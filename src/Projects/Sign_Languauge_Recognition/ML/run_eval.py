import torch

from evaluate import evaluate
from config import Config
from model import Seq2Seq
from data_pipeline import test_loader
import torch.nn as nn


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
    
    
cfg = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state = torch.load("checkpoints/best.pt", map_location=device)

cfg.vocab_size = state["config"]["vocab_size"]
use_ctc = state["config"].get("use_ctc", False)

model = Seq2Seq(
    input_dim=state["config"]["input_dim"],
    vocab_size=cfg.vocab_size,
    hidden_dim=state["config"].get("hidden_dim", 256),
    use_ctc=use_ctc,
).to(device)

model.load_state_dict(state["model_state"])
model.eval()

lm = None

try:
    lm = LanguageModel(cfg.vocab_size).to(device)
    lm.load_state_dict(torch.load("checkpoints/lm.pt", map_location=device))
    lm.eval()
    print("Loaded language model: checkpoints/lm.pt")
except FileNotFoundError:
    print("No LM found at checkpoints/lm.pt — running normal beam only.")

print("Running Transformer Seq2Seq evaluation...")
print(f"Checkpoint epoch: {state.get('epoch', 'N/A')}")
print(f"Checkpoint val loss: {state.get('val_loss', 'N/A')}")
print(f"Config: {state.get('config', {})}")

print("\n=== GREEDY ===")
print(evaluate(model, test_loader, cfg, mode="greedy"))

print("\n=== BEAM ===")
print(evaluate(model, test_loader, cfg, mode="beam"))

if lm is not None:
    for w in [0.1, 0.3, 0.6, 1.0, 1.5]:
        print(f"\n=== BEAM + LM weight {w} ===")
        print(evaluate(model, test_loader, cfg, mode="beam", lm=lm, lm_weight=w))