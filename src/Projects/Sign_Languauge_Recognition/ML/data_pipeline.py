import os
import json
import numpy as np
import pandas as pd
import torch
import unicodedata

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2

USE_VELOCITY = False
USE_ACCELERATION = False

# Set to None for full sequence.
# Try 12 first. Later try 16, 24, then None.
MAX_TARGET_TOKENS = 12

videos = os.path.join("..", "videos_all_languages.json")

with open(videos, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.json_normalize(data["videos"])

Hand_Landmarks_paths = df["HandLandmark"]
PROJECT_ROOT = os.path.abspath(os.path.join("..", "..", "..", ".."))


def resolve_path(p):
    if p.startswith("/ephemeral/Project/Data/"):
        rel = p.replace("/ephemeral/Project/Data/", "")
        new_path = os.path.join(PROJECT_ROOT, "Data", rel)
        if os.path.exists(new_path):
            return new_path

    candidates = [
        p,
        os.path.join(PROJECT_ROOT, p.lstrip("/")),
        os.path.abspath(p),
    ]

    for c in candidates:
        if os.path.exists(c):
            return c

    return None


print("PROJECT_ROOT:", PROJECT_ROOT)
print("Example raw path:", Hand_Landmarks_paths.iloc[0])
print("Resolved path:", resolve_path(Hand_Landmarks_paths.iloc[0]))

Hand_Landmarks = []

for p in Hand_Landmarks_paths:
    resolved = resolve_path(p)
    if resolved is not None:
        Hand_Landmarks.append(resolved)

print("Valid landmark files:", len(Hand_Landmarks))


def fix_mojibake(name):
    try:
        name = name.encode("latin1").decode("utf8")
    except Exception:
        pass
    return unicodedata.normalize("NFC", name)


def normalize(seq):
    valid = np.any(seq != 0, axis=1)

    if valid.sum() < 2:
        return np.zeros_like(seq)

    center = seq[valid].mean(axis=0)
    seq[valid] -= center

    scale = np.std(seq[valid])
    if scale < 1e-6:
        return np.zeros_like(seq)

    seq[valid] /= scale
    return seq


HAND = list(range(21))
POSE = list(range(33))

FACE = (
    list(range(70, 103)) +
    list(range(0, 17)) +
    list(range(61, 88)) +
    list(range(152, 172))
)


def convert_frames_to_sequence(frames):
    rows = []

    for frame in frames:
        feat = []

        left = {
            lm["id"]: lm
            for h in frame["hands"] if h["handedness"] == "Left"
            for lm in h["landmarks"]
        }

        right = {
            lm["id"]: lm
            for h in frame["hands"] if h["handedness"] == "Right"
            for lm in h["landmarks"]
        }

        for i in HAND:
            lm = left.get(i)
            feat.extend([0, 0, 0] if lm is None else [lm["x"], lm["y"], lm["z"]])

        for i in HAND:
            lm = right.get(i)
            feat.extend([0, 0, 0] if lm is None else [lm["x"], lm["y"], lm["z"]])

        pose = {lm["id"]: lm for p in frame["pose"] for lm in p["landmarks"]}
        for i in POSE:
            lm = pose.get(i)
            feat.extend([0, 0, 0] if lm is None else [lm["x"], lm["y"], lm["z"]])

        face = {lm["id"]: lm for f in frame["face"] for lm in f["landmarks"]}
        for i in FACE:
            lm = face.get(i)
            feat.extend([0, 0, 0] if lm is None else [lm["x"], lm["y"], lm["z"]])

        rows.append(np.array(feat, dtype=np.float32))

    return np.stack(rows)


def add_motion_features(seq):
    features = [seq]

    if USE_VELOCITY:
        if len(seq) > 1:
            vel = np.diff(seq, axis=0)
            vel = np.vstack([np.zeros_like(seq[0]), vel])
        else:
            vel = np.zeros_like(seq)
        features.append(vel)

    if USE_ACCELERATION:
        if len(seq) > 2:
            acc = np.diff(features[-1], axis=0)
            acc = np.vstack([np.zeros_like(seq[0]), acc])
        else:
            acc = np.zeros_like(seq)
        features.append(acc)

    return np.concatenate(features, axis=1).astype(np.float32)


df["filepath"] = df["filepath"].apply(
    lambda p: fix_mojibake(
        os.path.basename(p).replace(".mp4", "").replace(".MOV", "")
    )
)

label_dict = dict(zip(df["filepath"], df["HamNoSys"]))

X, y = [], []

for path in Hand_Landmarks:
    with open(path, "r") as f:
        data = json.load(f)

    vid = fix_mojibake(os.path.basename(path).replace(".json", ""))

    if vid in label_dict:
        seq = convert_frames_to_sequence(data["frames"])
        seq = add_motion_features(seq)
        seq = normalize(seq)

        X.append(seq)
        y.append(label_dict[vid])

print("Samples:", len(X))


def tokenize(text):
    raw_tokens = list(text.strip())

    grouped = []

    for t in raw_tokens:
        code = ord(t)

        # Keep special separator-like HamNoSys symbols because they structure the sequence
        if t in ["", "", "", ""]:
            grouped.append(t)

        # Latin letters
        elif t.isalpha():
            grouped.append("LETTER")

        # Numbers
        elif t.isdigit():
            grouped.append("DIGIT")

        # Private-use HamNoSys symbols
        elif 0xE000 <= code <= 0xF8FF:
            grouped.append(f"PUA_{code // 16}")

        # Everything else
        else:
            grouped.append("SYM")

    return grouped

all_tokens = sorted({tok for seq in y for tok in tokenize(seq)})

token_to_id = {t: i + 3 for i, t in enumerate(all_tokens)}
token_to_id["<PAD>"] = PAD_ID
token_to_id["<SOS>"] = SOS_ID
token_to_id["<EOS>"] = EOS_ID

id_to_token = {v: k for k, v in token_to_id.items()}
vocab_size = len(token_to_id)

y_encoded = []

for seq in y:
    tokens = tokenize(seq)

    if MAX_TARGET_TOKENS is not None:
        tokens = tokens[:MAX_TARGET_TOKENS]

    encoded = [SOS_ID] + [token_to_id[t] for t in tokens] + [EOS_ID]
    y_encoded.append(encoded)


def pad_X(seqs, max_len):
    feat = seqs[0].shape[1]
    X_tensor = torch.zeros(len(seqs), max_len, feat)
    mask = torch.zeros(len(seqs), max_len, dtype=torch.bool)

    for i, s in enumerate(seqs):
        l = min(len(s), max_len)
        X_tensor[i, :l] = torch.tensor(s[:l])
        mask[i, :l] = 1

    return X_tensor, mask


def pad_y(seqs, max_len):
    Y = torch.full((len(seqs), max_len), PAD_ID, dtype=torch.long)

    for i, s in enumerate(seqs):
        l = min(len(s), max_len)
        Y[i, :l] = torch.tensor(s[:l], dtype=torch.long)

    return Y


X_train, X_tmp, y_train, y_tmp = train_test_split(
    X,
    y_encoded,
    test_size=0.3,
    random_state=42,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_tmp,
    y_tmp,
    test_size=0.5,
    random_state=42,
)

max_len_X = max(len(s) for s in X)
max_len_y = max(len(s) for s in y_encoded)

train_X, train_mask = pad_X(X_train, max_len_X)
val_X, val_mask = pad_X(X_val, max_len_X)
test_X, test_mask = pad_X(X_test, max_len_X)

train_y = pad_y(y_train, max_len_y)
val_y = pad_y(y_val, max_len_y)
test_y = pad_y(y_test, max_len_y)

input_dim = train_X.shape[-1]

print("Input dim:", input_dim)
print("Max target length:", max_len_y)
print("Vocab size:", vocab_size)

batch_size = 16

train_loader = DataLoader(
    TensorDataset(train_X, train_mask, train_y),
    batch_size=batch_size,
    shuffle=True,
)

val_loader = DataLoader(
    TensorDataset(val_X, val_mask, val_y),
    batch_size=batch_size,
)

test_loader = DataLoader(
    TensorDataset(test_X, test_mask, test_y),
    batch_size=batch_size,
)