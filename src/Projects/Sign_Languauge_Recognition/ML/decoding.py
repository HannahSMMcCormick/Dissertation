import torch
import torch.nn.functional as F
import heapq


def decode(model, src, mask, cfg, mode="beam", lm=None, lm_weight=0.3):
    if mode == "greedy":
        return greedy_decode(model, src, mask, cfg)
    elif mode == "beam":
        return beam_decode(model, src, mask, cfg, lm=lm, lm_weight=lm_weight)
    elif mode == "best_first":
        return best_first_decode(model, src, mask, cfg)
    elif mode == "ctc":
        return ctc_decode(model, src, mask, cfg)
    elif mode == "ctc_beam":
        return ctc_beam_decode(model, src, mask, cfg)
    else:
        raise ValueError(f"Unknown decode mode: {mode}")
    
    
def greedy_decode(model, src, mask, cfg):
    model.eval()

    with torch.no_grad():
        src_pad_mask = mask == 0
        memory = model.encoder(src, src_pad_mask)

        seq = [cfg.sos_id]

        for _ in range(cfg.max_decode_len):
            tgt = torch.tensor(seq, device=src.device).unsqueeze(0)

            tgt_pad_mask = tgt == cfg.pad_id
            tgt_mask = model.generate_square_subsequent_mask(
                tgt.size(1),
                tgt.device,
            )

            logits, _ = model.decoder(
                tgt,
                memory,
                tgt_mask,
                tgt_pad_mask,
                src_pad_mask,
            )

            next_token = logits[:, -1].argmax(-1).item()
            seq.append(next_token)

            if next_token == cfg.eos_id:
                break

        return seq

def beam_decode(model, src, mask, cfg, lm=None, lm_weight=0.3):
    model.eval()
    if lm is not None:
        lm.eval()

    with torch.no_grad():

        src_pad_mask = (mask == 0)
        memory = model.encoder(src, src_pad_mask)

        beams = [([cfg.sos_id], 0.0)]

        for _ in range(cfg.max_decode_len):

            new_beams = []

            for seq, score in beams:

                if seq[-1] == cfg.eos_id:
                    new_beams.append((seq, score))
                    continue

                tgt = torch.tensor(seq).unsqueeze(0).to(src.device)

                tgt_pad_mask = (tgt == cfg.pad_id)
                tgt_mask = model.generate_square_subsequent_mask(
                    tgt.size(1), tgt.device
                )

                logits, _ = model.decoder(
                    tgt, memory, tgt_mask, tgt_pad_mask, src_pad_mask
                )

                log_probs = F.log_softmax(logits[:, -1], dim=-1)
                topk = torch.topk(log_probs, cfg.beam_size)

                for i in range(cfg.beam_size):
                    tok = topk.indices[0][i].item()
                    logp = topk.values[0][i].item()

                    if tok in seq:
                        logp -= 0.3

                    new_seq = seq + [tok]

                    # ----------------------
                    # LANGUAGE MODEL SCORE
                    # ----------------------
                    lm_score = 0.0
                    if lm is not None and len(new_seq) > 1:
                        lm_input = torch.tensor(new_seq[:-1]).unsqueeze(0).to(src.device)
                        lm_logits = lm(lm_input)
                        lm_log_probs = F.log_softmax(lm_logits[:, -1], dim=-1)
                        lm_score = lm_log_probs[0, tok].item()

                    total_score = score + logp + lm_weight * lm_score

                    new_beams.append((new_seq, total_score))

            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:cfg.beam_size]

        return beams[0][0]

def best_first_decode(model, src, mask, cfg):
    model.eval()

    with torch.no_grad():
        src_pad_mask = mask == 0
        memory = model.encoder(src, src_pad_mask)

        heap = [(0.0, [cfg.sos_id])]
        completed = []

        while heap and len(completed) < cfg.beam_size:
            neg_score, seq = heapq.heappop(heap)
            score = -neg_score

            if seq[-1] == cfg.eos_id:
                completed.append((seq, score))
                continue

            if len(seq) >= cfg.max_decode_len:
                completed.append((seq, score))
                continue

            tgt = torch.tensor(seq, device=src.device).unsqueeze(0)

            tgt_pad_mask = tgt == cfg.pad_id
            tgt_mask = model.generate_square_subsequent_mask(
                tgt.size(1),
                tgt.device,
            )

            logits, _ = model.decoder(
                tgt,
                memory,
                tgt_mask,
                tgt_pad_mask,
                src_pad_mask,
            )

            log_probs = F.log_softmax(logits[:, -1], dim=-1)
            topk = torch.topk(log_probs, cfg.beam_size)

            for i in range(cfg.beam_size):
                token = topk.indices[0][i].item()
                logp = topk.values[0][i].item()

                new_seq = seq + [token]
                new_score = score + logp

                heapq.heappush(heap, (-new_score, new_seq))

        if completed:
            return max(
                completed,
                key=lambda item: item[1] / (len(item[0]) ** cfg.alpha),
            )[0]

        return [cfg.sos_id]


def ctc_decode(model, src, mask, cfg):
    model.eval()

    with torch.no_grad():
        src_pad_mask = mask == 0
        memory = model.encoder(src, src_pad_mask)

        ctc_logits = model.ctc_head(memory)
        log_probs = F.log_softmax(ctc_logits, dim=-1)

        pred = log_probs.argmax(dim=-1)[0]
        valid_len = int(mask.sum(dim=1).item())
        pred = pred[:valid_len]

        collapsed = []
        prev = None

        for token in pred.tolist():
            if token != prev and token != cfg.vocab_size:
                collapsed.append(token)
            prev = token

        return collapsed if collapsed else [cfg.sos_id]


def ctc_beam_decode(model, src, mask, cfg):
    model.eval()

    with torch.no_grad():
        src_pad_mask = mask == 0
        memory = model.encoder(src, src_pad_mask)

        ctc_logits = model.ctc_head(memory)
        log_probs = F.log_softmax(ctc_logits, dim=-1)[0]

        valid_len = int(mask.sum(dim=1).item())
        log_probs = log_probs[:valid_len]

        beams = [([], 0.0)]

        for t in range(log_probs.size(0)):
            new_beams = []

            topk = torch.topk(log_probs[t], cfg.beam_size)

            for seq, score in beams:
                for i in range(cfg.beam_size):
                    token = topk.indices[i].item()
                    logp = topk.values[i].item()

                    if token == cfg.vocab_size:
                        new_beams.append((seq.copy(), score + logp))
                    elif seq and seq[-1] == token:
                        new_beams.append((seq.copy(), score + logp))
                    else:
                        new_beams.append((seq + [token], score + logp))

            beams = sorted(
                new_beams,
                key=lambda item: item[1],
                reverse=True,
            )[: cfg.beam_size]

        best_seq, _ = beams[0] if beams else ([], 0.0)
        return best_seq if best_seq else [cfg.sos_id]