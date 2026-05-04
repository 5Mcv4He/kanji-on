"""Extract human-readable rules via binary one-vs-rest decision trees.

Strategy: For each common on'yomi reading, train a binary decision tree
(XGBoost distilled) that predicts "is this kanji read as X?". Extract
the root-to-leaf paths as if-then rules.

This avoids the 169-class sparsity problem by treating each on'yomi independently.
"""
import pandas as pd
import numpy as np
import sys
import pickle
import json
import os
from collections import defaultdict, Counter
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

LEVEL = sys.argv[1] if len(sys.argv) > 1 else "N5"
FBASE = LEVEL.lower().replace("+", "p")

print(f"{'='*60}")
print(f"XGBoost Rule Distillation — {LEVEL} (Binary OvR Trees)")
print(f"{'='*60}")

# ── Load ──
df = pd.read_csv(f"dataset/{FBASE}_dataset.csv")
df = df[df["onyomi"].notna() & (df["onyomi"] != "")].copy()
df = df[df["observed_readings"].notna() & (df["observed_readings"] != "")].copy()
df = df.reset_index(drop=True)
print(f"Dataset: {len(df)} kanji")

# ── Load XGBoost (for distillation target) ──
with open(f"models/saved/{FBASE}_label_encoders.pkl", "rb") as f:
    saved = pickle.load(f)
le_on = saved["le_on"]
le_train = saved["le_train"]
on_to_train = saved["on_to_train"]
train_to_on = {v: k for k, v in on_to_train.items()}

import xgboost as xgb
model = xgb.XGBClassifier()
model.load_model(f"models/saved/{FBASE}_xgboost.json")

# Get XGBoost predictions
encs = {}
for col, enc_name in [("radical", "radical_enc"), ("non_radical_part", "nonrad_enc"),
                       ("pinyin_initial", "init_enc"), ("pinyin_final", "final_enc"),
                       ("nasal_coda", "nasal_enc")]:
    le = LabelEncoder()
    le.fit(df[col].fillna(""))
    encs[enc_name] = le

df_enc = df.copy()
for enc_name, le in encs.items():
    col = {"radical_enc": "radical", "nonrad_enc": "non_radical_part",
           "init_enc": "pinyin_initial", "final_enc": "pinyin_final",
           "nasal_enc": "nasal_coda"}[enc_name]
    df_enc[enc_name] = le.transform(df[col].fillna(""))

feature_cols_xgb = [
    "radical_enc", "nonrad_enc", "init_enc", "final_enc",
    "pinyin_tone_num", "nasal_enc",
    "radical_strokes", "non_radical_strokes", "total_strokes",
    "is_entering_tone",
]
X_xgb = df_enc[feature_cols_xgb].fillna(0).astype(float)
y_xgb_proba = model.predict_proba(X_xgb)
y_xgb = np.argmax(y_xgb_proba, axis=1)

# Map XGBoost preds → on'yomi strings
xgb_on_map = {}
for pred_id, on_str in zip(y_xgb, df["onyomi"]):
    if pred_id in train_to_on:
        on_id = train_to_on[pred_id]
        xgb_on_map[on_str] = le_on.classes_[on_id]
# Actually, use XGBoost prediction directly:
xgb_on_labels = []
for pred_id in y_xgb:
    if pred_id in train_to_on:
        on_id = train_to_on[pred_id]
        xgb_on_labels.append(le_on.classes_[on_id])
    else:
        xgb_on_labels.append(str(pred_id))

# ── Build one-hot feature matrix ──
TOP_NONRAD = 100
TOP_RADICAL = 60
TOP_FINAL = 60

nonrad_counts = df["non_radical_part"].value_counts()
radical_counts = df["radical"].value_counts()
final_counts = df["pinyin_final"].value_counts()
init_counts = df["pinyin_initial"].value_counts()

top_nonrads = set(nonrad_counts.head(TOP_NONRAD).index)
top_radicals = set(radical_counts.head(TOP_RADICAL).index)
top_finals = set(final_counts.head(TOP_FINAL).index)
top_initials = set(init_counts.head(21).index)

feature_names = []
X_list = []

# Numerical
for col, name in [("pinyin_tone_num", "声调"), ("radical_strokes", "部首画数"),
                   ("non_radical_strokes", "声符画数"), ("total_strokes", "总画数"),
                   ("is_entering_tone", "入声")]:
    X_list.append(df[col].fillna(0).astype(float).values)
    feature_names.append(name)

# One-hot categorical
for col, top_set, prefix in [
    ("non_radical_part", top_nonrads, "声符"),
    ("radical", top_radicals, "部首"),
    ("pinyin_initial", top_initials, "声母"),
    ("pinyin_final", top_finals, "韵母"),
]:
    col_vals = df[col].fillna("")
    for val in sorted(top_set):
        if val and str(val) != "nan":
            X_list.append((col_vals == val).astype(float).values)
            feature_names.append(f"{prefix}={val}")

X = np.column_stack(X_list)
print(f"Feature matrix: {X.shape}")

# ── Select on'yomi classes with enough samples ──
onyomi_counts = Counter(df["onyomi"])
common_onyomi = [(on, cnt) for on, cnt in onyomi_counts.most_common(40) if cnt >= 5]
print(f"Training binary trees for {len(common_onyomi)} on'yomi classes (≥5 samples)")

# ── For each common on'yomi, train a binary DT and extract rules ──
all_rules = []
MAX_DEPTH = 5
MIN_LEAF = 4

for on_reading, total_count in common_onyomi:
    # Binary target: 1 = this on'yomi, 0 = other
    y_true_bin = (df["onyomi"] == on_reading).astype(int).values

    # XGBoost binary: did XGBoost predict this on'yomi?
    y_xgb_bin = np.array([1 if lbl == on_reading else 0 for lbl in xgb_on_labels])

    # Train tree to match XGBoost (distillation)
    # Only if XGBoost knows this class
    xgb_pos = y_xgb_bin.sum()
    if xgb_pos < 3:
        continue  # XGBoost can't identify this class

    dtree = DecisionTreeClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_LEAF,
                                    random_state=42)
    dtree.fit(X, y_xgb_bin)

    # Extract rules
    tree_ = dtree.tree_
    paths = []

    def recurse(node_id, conditions):
        if tree_.feature[node_id] != _tree.TREE_UNDEFINED:
            feat = feature_names[tree_.feature[node_id]]
            thresh = tree_.threshold[node_id]
            recurse(tree_.children_left[node_id],
                    conditions + [(feat, thresh, "≤")])
            recurse(tree_.children_right[node_id],
                    conditions + [(feat, thresh, ">")])
        else:
            value = tree_.value[node_id][0]
            prob = value[1] / value.sum()  # probability of positive class
            n = tree_.n_node_samples[node_id]
            if prob >= 0.4 and len(conditions) >= 1:
                paths.append({"conditions": list(conditions), "prob": float(prob),
                              "n_samples": int(n)})

    recurse(0, [])

    # Evaluate each path on TRUE labels
    for p in paths:
        # Build mask
        mask = np.ones(len(X), dtype=bool)
        for feat_name, thresh, op in p["conditions"]:
            if feat_name in feature_names:
                idx = feature_names.index(feat_name)
                if op == "≤":
                    mask &= (X[:, idx] <= thresh)
                else:
                    mask &= (X[:, idx] > thresh)

        coverage = mask.sum()
        if coverage < 3:
            continue

        true_covered = y_true_bin[mask]
        correct = true_covered.sum()
        accuracy = correct / coverage
        recall = correct / max(total_count, 1)

        if accuracy < 0.55:
            continue

        # Simplify rule text
        present = []
        numerical = []

        def is_binary_feat(fname):
            return any(fname.startswith(p) for p in ["声符=", "部首=", "声母=", "韵母=", "鼻音="])

        num_bounds = {}  # feat_name → (min_lower, max_upper) for same-feature dedup
        for feat_name, thresh, op in p["conditions"]:
            if is_binary_feat(feat_name):
                if op == ">":
                    present.append(feat_name)
            else:
                val = int(thresh) if thresh == int(thresh) else round(thresh, 1)
                if feat_name not in num_bounds:
                    num_bounds[feat_name] = [float("-inf"), float("inf")]
                if op == "≤":
                    num_bounds[feat_name][1] = min(num_bounds[feat_name][1], val)
                else:  # >
                    num_bounds[feat_name][0] = max(num_bounds[feat_name][0], val)

        for feat_name, (lo, hi) in num_bounds.items():
            if lo != float("-inf") and hi != float("inf"):
                # Both bounds present: lo < feat ≤ hi
                numerical.append(f"{lo}<{feat_name}≤{hi}")
            elif hi != float("inf"):
                numerical.append(f"{feat_name}≤{hi}")
            elif lo != float("-inf"):
                numerical.append(f"{feat_name}>{lo}")

        parts = present + numerical
        if not parts:
            continue
        rule_text = " AND ".join(parts)

        # Examples
        matched_idx = np.where(mask)[0]
        correct_idx = matched_idx[true_covered > 0]
        wrong_idx = matched_idx[true_covered == 0]

        examples = [f"{df.iloc[i]['kanji']}" for i in correct_idx[:6]]
        counter_ex = []
        for i in wrong_idx[:4]:
            counter_ex.append(f"{df.iloc[i]['kanji']}({df.iloc[i]['onyomi']})")

        all_rules.append({
            "rule": rule_text,
            "target_on": on_reading,
            "coverage": int(coverage),
            "correct": int(correct),
            "accuracy": accuracy,
            "recall": recall,
            "examples": " ".join(examples),
            "counter": " ".join(counter_ex) if counter_ex else "无",
            "tree_prob": p["prob"],
            "n_conditions": len(present) + len(numerical),
        })

print(f"Extracted {len(all_rules)} binary rules before dedup")

# ── Sort by accuracy * sqrt(coverage) ──
all_rules.sort(key=lambda x: -x["accuracy"] * np.sqrt(x["coverage"]))

# ── Deduplicate ──
def similar(r1, r2):
    if r1["target_on"] != r2["target_on"]:
        return False
    # Share most examples
    ex1 = set(r1["examples"].split())
    ex2 = set(r2["examples"].split())
    common = len(ex1 & ex2)
    if min(len(ex1), len(ex2)) == 0:
        return False
    return common / min(len(ex1), len(ex2)) > 0.5

deduped = []
for r in all_rules:
    if not any(similar(r, d) for d in deduped):
        deduped.append(r)

print(f"After dedup: {len(deduped)} rules")

# ── Also extract direct DT rules (trained on ground truth, not XGBoost) ──
print("\nExtracting direct DT rules...")
direct_rules = []
for on_reading, total_count in common_onyomi:
    y_true_bin = (df["onyomi"] == on_reading).astype(int).values
    if y_true_bin.sum() < 5:
        continue

    dtree = DecisionTreeClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_LEAF,
                                    random_state=42)
    dtree.fit(X, y_true_bin)

    tree_ = dtree.tree_
    paths = []
    def recurse(node_id, conditions):
        if tree_.feature[node_id] != _tree.TREE_UNDEFINED:
            feat = feature_names[tree_.feature[node_id]]
            thresh = tree_.threshold[node_id]
            recurse(tree_.children_left[node_id],
                    conditions + [(feat, thresh, "≤")])
            recurse(tree_.children_right[node_id],
                    conditions + [(feat, thresh, ">")])
        else:
            value = tree_.value[node_id][0]
            prob = value[1] / value.sum()
            n = tree_.n_node_samples[node_id]
            if prob >= 0.4 and len(conditions) >= 1:
                paths.append({"conditions": list(conditions), "prob": float(prob),
                              "n_samples": int(n)})
    recurse(0, [])

    for p in paths:
        mask = np.ones(len(X), dtype=bool)
        for feat_name, thresh, op in p["conditions"]:
            if feat_name in feature_names:
                idx = feature_names.index(feat_name)
                if op == "≤":
                    mask &= (X[:, idx] <= thresh)
                else:
                    mask &= (X[:, idx] > thresh)
        coverage = mask.sum()
        if coverage < 3:
            continue
        true_covered = y_true_bin[mask]
        correct = true_covered.sum()
        accuracy = correct / coverage
        if accuracy < 0.55:
            continue
        present = []
        def is_bin_feat(fname):
            return any(fname.startswith(p) for p in ["声符=", "部首=", "声母=", "韵母=", "鼻音="])
        num_bounds = {}
        for feat_name, thresh, op in p["conditions"]:
            if is_bin_feat(feat_name):
                if op == ">":
                    present.append(feat_name)
            else:
                val = int(thresh) if thresh == int(thresh) else round(thresh, 1)
                if feat_name not in num_bounds:
                    num_bounds[feat_name] = [float("-inf"), float("inf")]
                if op == "≤":
                    num_bounds[feat_name][1] = min(num_bounds[feat_name][1], val)
                else:
                    num_bounds[feat_name][0] = max(num_bounds[feat_name][0], val)
        numerical = []
        for feat_name, (lo, hi) in num_bounds.items():
            if lo != float("-inf") and hi != float("inf"):
                numerical.append(f"{lo}<{feat_name}≤{hi}")
            elif hi != float("inf"):
                numerical.append(f"{feat_name}≤{hi}")
            elif lo != float("-inf"):
                numerical.append(f"{feat_name}>{lo}")
        parts = present + numerical
        if not parts:
            continue
        rule_text = " AND ".join(parts)
        matched_idx = np.where(mask)[0]
        correct_idx = matched_idx[true_covered > 0]
        wrong_idx = matched_idx[true_covered == 0]
        examples = [f"{df.iloc[i]['kanji']}" for i in correct_idx[:6]]
        counter_ex = [f"{df.iloc[i]['kanji']}({df.iloc[i]['onyomi']})"
                      for i in wrong_idx[:4]]
        direct_rules.append({
            "rule": rule_text, "target_on": on_reading,
            "coverage": int(coverage), "correct": int(correct),
            "accuracy": accuracy,
            "examples": " ".join(examples),
            "counter": " ".join(counter_ex) if counter_ex else "无",
        })

direct_rules.sort(key=lambda x: -x["accuracy"] * np.sqrt(x["coverage"]))

# Dedup direct
direct_dedup = []
for r in direct_rules:
    if not any(similar(r, d) for d in direct_dedup):
        direct_dedup.append(r)
print(f"Direct DT rules: {len(direct_dedup)}")

# ── Output ──
os.makedirs("output", exist_ok=True)

out = []
out.append(f"# XGBoost 规则蒸馏 — {LEVEL}")
out.append("")
out.append(f"## 方法: 二分类决策树 (Binary OvR)")
out.append("")
out.append(f"- 对每个常见音读, 训练一个二分类决策树")
out.append(f"- XGBoost 蒸馏版: 树学习 XGBoost 的预测行为")
out.append(f"- 直接版: 树直接学习真实音读标签")
out.append(f"- 树深度 ≤ {MAX_DEPTH}, 叶节点 ≥ {MIN_LEAF} 样本")
out.append(f"- 共 {len(common_onyomi)} 个音读类参与训练")
out.append("")

out.append("## XGBoost 蒸馏规则 (学习 XGBoost 的判别模式)\n")
out.append("| # | 条件 | →音读 | 覆盖 | 准确率 | 召回 | 正例 |")
out.append("|---|------|------|------|--------|------|------|")
for i, r in enumerate(deduped[:80]):
    rule_str = r["rule"].replace("|", "/")
    out.append(f"| {i+1} | {rule_str} | **{r['target_on']}** | "
               f"{r['coverage']}字 | {r['accuracy']:.0%} | {r['recall']:.0%} | "
               f"{r['examples'][:50]} |")

out.append(f"\n## 直接决策树规则 (从真实标签学习, {len(direct_dedup)}条)\n")
out.append("| # | 条件 | →音读 | 覆盖 | 准确率 | 正例 |")
out.append("|---|------|------|------|--------|------|")
for i, r in enumerate(direct_dedup[:60]):
    rule_str = r["rule"].replace("|", "/")
    out.append(f"| {i+1} | {rule_str} | **{r['target_on']}** | "
               f"{r['coverage']}字 | {r['accuracy']:.0%} | {r['examples'][:50]} |")

# High accuracy from both
h_xgb = [r for r in deduped if r["accuracy"] >= 0.75 and r["coverage"] >= 4]
h_dir = [r for r in direct_dedup if r["accuracy"] >= 0.75 and r["coverage"] >= 4]
h_both = []
seen_rules = set()
for r in deduped:
    key = (r["rule"], r["target_on"])
    if key not in seen_rules and r["accuracy"] >= 0.75 and r["coverage"] >= 4:
        h_both.append(r)
        seen_rules.add(key)
for r in direct_dedup:
    key = (r["rule"], r["target_on"])
    if key not in seen_rules and r["accuracy"] >= 0.75 and r["coverage"] >= 4:
        h_both.append(r)
        seen_rules.add(key)

out.append(f"\n## 高准确率规则汇总 (≥75%, {len(h_both)}条)\n")
out.append("| 条件 | →音读 | 覆盖 | 准确率 | 正例 | 反例 |")
out.append("|------|------|------|--------|------|------|")
for r in sorted(h_both, key=lambda x: -x["accuracy"])[:50]:
    rule_str = r["rule"].replace("|", "/")
    out.append(f"| {rule_str} | **{r['target_on']}** | {r['coverage']}字 | "
               f"{r['accuracy']:.0%} | {r['examples'][:40]} | {r['counter'][:40]} |")

with open(f"output/{FBASE}_xgboost_rules.md", "w") as f:
    f.write("\n".join(out))

# JSON
json_out = {
    "level": LEVEL,
    "method": "binary_ovr_decision_trees",
    "n_onyomi_classes": len(common_onyomi),
    "xgb_distilled_rules": [
        {"rule": r["rule"], "onyomi": r["target_on"], "coverage": r["coverage"],
         "accuracy": round(r["accuracy"], 3), "examples": r["examples"]}
        for r in deduped[:200]
    ],
    "direct_rules": [
        {"rule": r["rule"], "onyomi": r["target_on"], "coverage": r["coverage"],
         "accuracy": round(r["accuracy"], 3), "examples": r["examples"]}
        for r in direct_dedup[:200]
    ],
}
with open(f"output/{FBASE}_xgboost_rules.json", "w") as f:
    json.dump(json_out, f, indent=2, ensure_ascii=False)

# ── Summary ──
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")

print(f"\nXGBoost distilled rules: {len(deduped)}")
for thresh in [0.9, 0.8, 0.7, 0.6, 0.5]:
    n = len([r for r in deduped if r["accuracy"] >= thresh and r["coverage"] >= 4])
    print(f"  Acc ≥{thresh:.0%} & cov ≥4: {n}")

print(f"\nDirect DT rules: {len(direct_dedup)}")
for thresh in [0.9, 0.8, 0.7, 0.6, 0.5]:
    n = len([r for r in direct_dedup if r["accuracy"] >= thresh and r["coverage"] >= 4])
    print(f"  Acc ≥{thresh:.0%} & cov ≥4: {n}")

print(f"\nTop 20 XGBoost distilled rules:")
print("-" * 60)
for i, r in enumerate(deduped[:20]):
    print(f"\n[{i+1}] IF {r['rule']}")
    print(f"    → {r['target_on']}  覆盖{r['coverage']}字  准确率{r['accuracy']:.0%}  召回{r['recall']:.0%}")
    print(f"    ✓ {r['examples'][:80]}")
    if r['counter'] != '无':
        print(f"    ✗ {r['counter'][:60]}")

print(f"\nTop 15 Direct DT rules:")
print("-" * 60)
for i, r in enumerate(direct_dedup[:15]):
    print(f"\n[{i+1}] IF {r['rule']}")
    print(f"    → {r['target_on']}  覆盖{r['coverage']}字  准确率{r['accuracy']:.0%}")
    print(f"    ✓ {r['examples'][:80]}")

print(f"\nSaved: output/{FBASE}_xgboost_rules.md")
print(f"Saved: output/{FBASE}_xgboost_rules.json")
