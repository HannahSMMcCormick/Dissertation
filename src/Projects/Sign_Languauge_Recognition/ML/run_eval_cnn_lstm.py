import torch
from evaluate import evaluate
from config import Config
from model_cnn_lstm import Seq2SeqCNN
from data_pipeline import test_loader

cfg = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained CNN-LSTM checkpoint
state = torch.load("checkpoints/cnn_lstm_best.pt", map_location=device)

# IMPORTANT: sync vocab size
cfg.vocab_size = state["config"]["vocab_size"]

model = Seq2SeqCNN(
    input_dim=state["config"]["input_dim"],
    vocab_size=cfg.vocab_size,
    hidden_dim=state["config"]["hidden_dim"],
    use_ctc=True,
).to(device)

model.load_state_dict(state["model_state"])
model.eval()

print("Running CNN-LSTM evaluation...")

print("\n=== CTC GREEDY ===")
print(evaluate(model, test_loader, cfg, mode="ctc"))

print("\n=== CTC BEAM ===")
print(evaluate(model, test_loader, cfg, mode="ctc_beam"))
