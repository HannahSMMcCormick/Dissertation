import torch

from evaluate import evaluate
from config import Config
from model_cnn_seq2seq import CNNLSTMSeq2Seq
from data_pipeline import test_loader

cfg = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state = torch.load("checkpoints/cnn_seq2seq_best.pt", map_location=device)

cfg.vocab_size = state["config"]["vocab_size"]
cfg.use_ctc = False

model = CNNLSTMSeq2Seq(
    input_dim=state["config"]["input_dim"],
    vocab_size=cfg.vocab_size,
    hidden_dim=state["config"].get("hidden_dim", 256),
).to(device)

model.load_state_dict(state["model_state"])
model.eval()

print("Running CNN-LSTM Seq2Seq evaluation...")
print(f"Checkpoint epoch: {state.get('epoch', 'N/A')}")
print(f"Checkpoint val loss: {state.get('val_loss', 'N/A')}")
print(f"Config: {state.get('config', {})}")

print("\n=== GREEDY ===")
print(evaluate(model, test_loader, cfg, mode="greedy"))

print("\n=== BEAM ===")
print(evaluate(model, test_loader, cfg, mode="beam"))