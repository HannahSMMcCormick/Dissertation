import numpy as np


def edit_distance(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]

    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    return dp[-1][-1]


def compute_metrics(preds, trues):
    cer_scores = []
    exact = 0

    for pred, true in zip(preds, trues):
        cer_scores.append(edit_distance(pred, true) / max(1, len(true)))

        if pred == true:
            exact += 1

    return {
        "CER": float(np.mean(cer_scores)),
        "EXACT": float(exact / len(preds)) if preds else 0.0,
    }