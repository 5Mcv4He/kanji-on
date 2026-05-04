"""Phase 2: Statistical baseline model for kanji on'yomi prediction.
P(on'yomi | component) with smoothing + phonetic component identification.
"""
import pandas as pd
import numpy as np
import json
from collections import defaultdict
from sklearn.model_selection import train_test_split

# ── Load dataset ────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv("dataset/kanji_dataset.csv")
with open("dataset/component_vocab.json") as f:
    component_to_id = json.load(f)
with open("dataset/onyomi_vocab.json") as f:
    onyomi_to_id = json.load(f)

id_to_onyomi = {v: k for k, v in onyomi_to_id.items()}
id_to_component = {v: k for k, v in component_to_id.items()}

# Parse component IDs
df["comp_id_list"] = df["component_ids"].apply(lambda x: json.loads(x) if isinstance(x, str) else [])

# ── 1. Phonetic component identification ────────────────────────────
# For each component, measure how predictively it maps to a single on'yomi
# The "best phonetic component" for a kanji is the one that most reliably
# predicts the kanji's on'yomi across the entire dataset.

print("Identifying phonetic components...")
comp_onyomi_counts = defaultdict(lambda: defaultdict(int))  # comp -> onyomi -> count
comp_total = defaultdict(int)

for _, row in df.iterrows():
    kanji = row["kanji"]
    onyomi = row["onyomi"]
    if not onyomi:
        continue
    comps = row["comp_id_list"]
    for cid in comps:
        comp_onyomi_counts[cid][onyomi] += 1
        comp_total[cid] += 1

# For each component, find its dominant on'yomi and predictive power
comp_predictive_power = {}  # component_id -> (dominant_on, consistency, total_kanji)
for cid, on_counts in comp_onyomi_counts.items():
    if comp_total[cid] < 2:
        continue
    dominant = max(on_counts, key=on_counts.get)
    consistency = on_counts[dominant] / comp_total[cid]
    comp_predictive_power[cid] = {
        "dominant": dominant,
        "consistency": consistency,
        "total": comp_total[cid],
        "counts": dict(on_counts)
    }

print(f"  Components with predictive power (appear in 2+ kanji): {len(comp_predictive_power)}")

# ── 2. Build conditional probability tables ─────────────────────────
# P(on'yomi | component) with Laplace smoothing (alpha=1)
# We'll use additive smoothing
def smoothed_prob(on_val, comp_id, alpha=0.5):
    """P(on'yomi | component) with Laplace smoothing."""
    if comp_id not in comp_onyomi_counts:
        return 1.0 / len(onyomi_to_id)  # uniform prior
    counts = comp_onyomi_counts[comp_id]
    total = comp_total[comp_id]
    num_classes = len(onyomi_to_id)
    return (counts.get(on_val, 0) + alpha) / (total + alpha * num_classes)

# ── 3. Prediction functions ─────────────────────────────────────────
def predict_onyomi_stats(kanji_component_ids, top_k=3):
    """Predict on'yomi using statistical model.
    Combines evidence from all components, weighted by predictive power.
    Returns top-k predictions with scores.
    """
    scores = defaultdict(float)
    total_weight = 0.0

    for cid in kanji_component_ids:
        if cid in comp_predictive_power:
            info = comp_predictive_power[cid]
            weight = info["consistency"] * np.log(info["total"] + 1)  # weight by reliability and support
            for on_val, count in info["counts"].items():
                scores[on_val] += weight * (count / info["total"])
            total_weight += weight

    if total_weight == 0:
        return []  # no components found

    # Normalize
    for k in scores:
        scores[k] /= total_weight

    # Sort and return top-k
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return ranked[:top_k]

def predict_onyomi_combined(kanji_component_ids, pinyin_final):
    """Predict on'yomi using components + pinyin final as additional evidence."""
    scores = defaultdict(float)
    total_weight = 0.0

    # Component evidence
    for cid in kanji_component_ids:
        if cid in comp_predictive_power:
            info = comp_predictive_power[cid]
            weight = info["consistency"] * np.log(info["total"] + 1)
            for on_val, count in info["counts"].items():
                scores[on_val] += weight * (count / info["total"])
            total_weight += weight

    if total_weight == 0:
        return []

    for k in scores:
        scores[k] /= total_weight

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return ranked

# ── 4. Evaluation per JLPT level ────────────────────────────────────
print("\n=== Baseline Accuracy by JLPT Level ===")
print(f"{'Level':<6} {'Kanji':<8} {'Top-1':<8} {'Top-3':<8} {'Top-5':<8} {'MeanRank':<10}")

level_order = ["N5", "N4", "N3", "N2", "N1", "N1+"]
all_results = []

for level in level_order:
    level_df = df[df["jlpt_level"] == level]
    if len(level_df) == 0:
        continue

    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0
    total_rank = 0
    total = 0

    for _, row in level_df.iterrows():
        onyomi = row["onyomi"]
        if not onyomi:
            continue

        comps = row["comp_id_list"]
        predictions = predict_onyomi_stats(comps, top_k=20)

        if not predictions:
            continue

        total += 1
        pred_ons = [p[0] for p in predictions]

        if onyomi == pred_ons[0]:
            correct_top1 += 1
        if onyomi in pred_ons[:3]:
            correct_top3 += 1
        if onyomi in pred_ons[:5]:
            correct_top5 += 1

        # Mean reciprocal rank
        try:
            rank = pred_ons.index(onyomi) + 1
        except ValueError:
            rank = len(onyomi_to_id)
        total_rank += rank

    if total > 0:
        acc1 = correct_top1 / total
        acc3 = correct_top3 / total
        acc5 = correct_top5 / total
        mean_rank = total_rank / total
        print(f"{level:<6} {total:<8} {acc1:<8.1%} {acc3:<8.1%} {acc5:<8.1%} {mean_rank:<10.1f}")
        all_results.append({
            "level": level,
            "total": total,
            "top1_acc": acc1,
            "top3_acc": acc3,
            "top5_acc": acc5,
            "mean_rank": mean_rank
        })

# ── 5. Top phonetic components by coverage ──────────────────────────
print("\n=== Top 20 Phonetic Components (by kanji coverage) ===")
sorted_comps = sorted(comp_predictive_power.items(),
                      key=lambda x: -x[1]["total"])
print(f"{'Comp':<6} {'Dominant':<8} {'Consistency':<14} {'Coverage':<10} {'Examples'}")
for cid, info in sorted_comps[:20]:
    comp_char = id_to_component[cid]
    examples = [k for k in sorted(info["counts"].items(), key=lambda x: -x[1])[:3]]
    example_str = " ".join(f"{on}({c})" for on, c in examples)
    print(f"{comp_char:<6} {info['dominant']:<8} {info['consistency']:<14.1%} {info['total']:<10} {example_str}")

# ── 6. Save results ─────────────────────────────────────────────────
results_df = pd.DataFrame(all_results)
results_df.to_csv("output/baseline_accuracy.csv", index=False)

# Save component predictive power for later use
comp_power_json = {}
for cid, info in comp_predictive_power.items():
    comp_char = id_to_component[cid]
    comp_power_json[comp_char] = {
        "dominant": info["dominant"],
        "consistency": round(info["consistency"], 4),
        "total": info["total"],
    }
with open("dataset/comp_predictive_power.json", "w", encoding="utf-8") as f:
    json.dump(comp_power_json, f, ensure_ascii=False, indent=2)

print(f"\nResults saved to output/baseline_accuracy.csv")
print(f"Component power data saved to dataset/comp_predictive_power.json")
