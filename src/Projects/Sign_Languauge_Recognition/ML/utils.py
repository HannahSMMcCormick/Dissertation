def clean(seq, cfg):
    out = []
    for t in seq:
        if t in (cfg.sos_id, cfg.pad_id):
            continue
        if t == cfg.eos_id:
            break
        out.append(t)
    return out