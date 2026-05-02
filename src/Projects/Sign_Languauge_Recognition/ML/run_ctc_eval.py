"""
Pure CTC Evaluation (no seq2seq decoder)
"""
import torch
from evaluate import evaluate
from decoding import decode
from config import Config
from model import Seq2Seq
from data_pipeline import test_loader

cfg = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pure CTC model
state = torch.load("checkpoints/ctc_best.pt", map_location=device)

model = Seq2Seq(
    input_dim=state["config"]["input_dim"],
    vocab_size=state["config"]["vocab_size"],
    hidden_dim=state["config"].get("hidden_dim", 256),
    use_ctc=True,
).to(device)

model.load_state_dict(state["model_state"])
model.eval()

print("Running CTC evaluation...")
print(f"Epoch: {state.get('epoch', 'N/A')}")
print(f"Val Loss: {state.get('val_loss', 'N/A'):.4f}")

# CTC greedy evaluation
print("\n=== CTC (Greedy) ===")
ctc_results = evaluate(model, test_loader, cfg, mode="ctc")
print(ctc_results)

# CTC beam search evaluation
print("\n=== CTC (Beam) ===")
ctc_beam_results = evaluate(model, test_loader, cfg, mode="ctc_beam")
print(ctc_beam_results)

# CTC A* search evaluation
print("\n=== CTC (A*) ===")
ctc_astar_results = evaluate(model, test_loader, cfg, mode="ctc_astar")
print(ctc_astar_results)