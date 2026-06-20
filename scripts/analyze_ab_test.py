"""
Analyze any A/B JSONL log: compare one metric between variants with a t-test.
"""

import json
import sys
from collections import defaultdict

from scipy import stats


def load(path, metric):
    groups = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            val = row.get(metric)
            if val is not None:
                groups[row["variant"]].append(float(val))
    return groups


if __name__ == "__main__":
    path = sys.argv[1]
    metric = sys.argv[2] if len(sys.argv) > 2 else "faithfulness"

    groups = load(path, metric)
    a, b = groups.get("A", []), groups.get("B", [])

    mean_a, mean_b = sum(a) / len(a), sum(b) / len(b)
    print(f"Metric: {metric}")
    print(f"Variant A: n={len(a)}, mean={mean_a:.4f}")
    print(f"Variant B: n={len(b)}, mean={mean_b:.4f}")
    print(f"Difference (B - A): {mean_b - mean_a:+.4f}")

    # Independent two-sample t-test. Welch's (equal_var=False) doesn't assume the
    # two groups have the same variance - the safer default.
    t_stat, p_value = stats.ttest_ind(b, a, equal_var=False)
    print(f"\nt = {t_stat:.3f}, p = {p_value:.4f}")

    if p_value < 0.05:
        winner = "B" if mean_b > mean_a else "A"
        print(f"=> Significant (p < 0.05). Variant {winner} wins on {metric}.")
    else:
        print(
            "=> Not significant (p >= 0.05). Can't conclude B differs from A. "
            "Either collect more data or the effect is just small."
        )
