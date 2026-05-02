import torch
from evaluate import evaluate
from decoding import decode
from config import Config
from model import Seq2Seq
from data_pipeline import test_loader

cfg = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CTC model
state = torch.load("checkpoints/ctc_best.pt", map_location=device)

cfg.vocab_size = state["config"]["vocab_size"]


model = Seq2Seq(
    input_dim=state["config"]["input_dim"],
    vocab_size=state["config"]["vocab_size"],
    hidden_dim=state["config"].get("hidden_dim", 256),
    use_ctc=True,
).to(device)

model.load_state_dict(state["model_state"])
model.eval()

print("Running CTC evaluation...")

print("\n=== CTC GREEDY ===")
print(evaluate(model, test_loader, cfg, mode="ctc"))

print("\n=== CTC BEAM ===")
print(evaluate(model, test_loader, cfg, mode="ctc_beam"))


