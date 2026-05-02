import torch

from decoding import decode
from metrics import compute_metrics
from utils import clean


def evaluate(
    model,
    loader,
    cfg,
    mode="beam",
    lm=None,
    lm_weight=0.3,
    show_examples=True,
    max_examples=4,
):
    model.eval()

    if lm is not None:
        lm.eval()

    preds = []
    trues = []

    device = next(model.parameters()).device

    with torch.no_grad():
        for src, src_mask, tgt in loader:
            src = src.to(device)
            src_mask = src_mask.to(device)
            tgt = tgt.to(device)

            for i in range(src.size(0)):
                pred_ids = decode(
                    model,
                    src[i : i + 1],
                    src_mask[i : i + 1],
                    cfg,
                    mode=mode,
                    lm=lm,
                    lm_weight=lm_weight,
                )

                true_ids = tgt[i].tolist()

                pred_clean = clean(pred_ids, cfg)
                true_clean = clean(true_ids, cfg)

                preds.append(pred_clean)
                trues.append(true_clean)

                if show_examples and len(preds) <= max_examples:
                    print("PRED IDS:", pred_clean)
                    print("TRUE IDS:", true_clean)

    return compute_metrics(preds, trues)