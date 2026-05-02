import torch
import torch.nn.functional as F


def ce_loss(logits, tgt, pad_id):
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        tgt.reshape(-1),
        ignore_index=pad_id,
        label_smoothing=0.1,
    )


def prepare_ctc_targets(tgt, cfg):
    targets = []
    lengths = []

    for seq in tgt:
        clean = [
            int(t.item())
            for t in seq
            if int(t.item()) not in (cfg.sos_id, cfg.eos_id, cfg.pad_id)
        ]

        targets.extend(clean)
        lengths.append(len(clean))

    return (
        torch.tensor(targets, dtype=torch.long, device=tgt.device),
        torch.tensor(lengths, dtype=torch.long, device=tgt.device),
    )


def ctc_loss(ctc_logits, tgt, src_mask, cfg):
    targets, target_lengths = prepare_ctc_targets(tgt, cfg)

    if targets.numel() == 0:
        return torch.tensor(0.0, device=ctc_logits.device)

    input_lengths = src_mask.sum(dim=1).long()

    return F.ctc_loss(
        ctc_logits.transpose(0, 1),
        targets,
        input_lengths,
        target_lengths,
        blank=cfg.vocab_size,
        zero_infinity=True,
    )


def coverage_loss(attn):
    if attn is None:
        return torch.tensor(0.0)

    attn = attn.mean(dim=1)

    coverage = torch.zeros_like(attn[:, 0])
    loss = 0.0

    for t in range(attn.size(1)):
        step = attn[:, t]
        loss = loss + torch.min(step, coverage).sum(dim=-1).mean()
        coverage = torch.clamp(coverage + step, max=1.0)

    return loss / max(1, attn.size(1))


def total_loss(outputs, batch, cfg):
    logits, ctc_logits, attn = outputs
    src, src_mask, gold = batch

    loss = ce_loss(logits, gold, cfg.pad_id)

    if cfg.use_ctc and ctc_logits is not None:
        loss = loss + cfg.ctc_weight * ctc_loss(
            ctc_logits,
            gold,
            src_mask,
            cfg,
        )

    if cfg.use_coverage and attn is not None:
        loss = loss + cfg.coverage_weight * coverage_loss(attn).to(logits.device)

    return loss