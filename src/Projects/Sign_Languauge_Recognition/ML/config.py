from dataclasses import dataclass


@dataclass
class Config:
    hidden_dim: int = 256
    vocab_size: int = 1000

    use_ctc: bool = False
    use_coverage: bool = False

    ctc_weight: float = 0.05
    coverage_weight: float = 0.1

    lr: float = 3e-4
    batch_size: int = 16
    max_epochs: int = 100

    beam_size: int = 3
    max_decode_len: int = 40
    alpha: float = 0.7

    blank_penalty: float = 1.5

    pad_id: int = 0
    sos_id: int = 1
    eos_id: int = 2