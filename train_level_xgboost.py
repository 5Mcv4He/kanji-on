"""Train XGBoost for any JLPT level, compare with rules, output fused results."""
import pandas as pd
import numpy as np
import os
import sys
import pickle
import json
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb

level = sys.argv[1] if len(sys.argv) > 1 else "N5"
os.makedirs("models/saved", exist_ok=True)
os.makedirs("output", exist_ok=True)

print("=" * 60)
print(f"{level} XGBoost TRAINING")
print("=" * 60)

# Load data
df = pd.read_csv(f"dataset/{level.lower().replace('+','p')}_dataset.csv")
df = df[df["onyomi"].notna() & (df["onyomi"] != "")].copy()
df = df[df["observed_readings"].notna() & (df["observed_readings"] != "")].copy()
print(f"{level}: {len(df)} kanji with on'yomi and observed readings")

# Encode
df["radical_enc"] = LabelEncoder().fit_transform(df["radical"].fillna(""))
df["nonrad_enc"] = LabelEncoder().fit_transform(df["non_radical_part"].fillna(""))
df["init_enc"] = LabelEncoder().fit_transform(df["pinyin_initial"].fillna(""))
df["final_enc"] = LabelEncoder().fit_transform(df["pinyin_final"].fillna(""))
df["nasal_enc"] = LabelEncoder().fit_transform(df["nasal_coda"].fillna("none"))

le_on = LabelEncoder()
y = le_on.fit_transform(df["onyomi"])
print(f"On'yomi classes: {len(le_on.classes_)}")

feature_cols = [
    "radical_enc", "nonrad_enc", "init_enc", "final_enc",
    "pinyin_tone_num", "nasal_enc",
    "radical_strokes", "non_radical_strokes", "total_strokes",
    "is_entering_tone",
]
X = df[feature_cols].fillna(0).astype(float)

# Filter >=2
class_counts = pd.Series(y).value_counts()
valid_mask = np.isin(y, class_counts[class_counts >= 2].index)
X_valid = X.iloc[valid_mask].copy()
y_prefilter = y[valid_mask]
df_valid = df.iloc[valid_mask].copy()

le_train = LabelEncoder()
y_valid = le_train.fit_transform(y_prefilter)
n_train_classes = len(le_train.classes_)

on_to_train = {}
for orig_label in np.unique(y_prefilter):
    on_to_train[orig_label] = le_train.transform([orig_label])[0]

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_valid, y_valid, df_valid.index, test_size=0.2, random_state=42,
    stratify=y_valid,
)
print(f"Train: {len(X_train)}, Test: {len(X_test)} (classes: {n_train_classes})")

# Train
model = xgb.XGBClassifier(
    objective="multi:softprob", num_class=n_train_classes,
    max_depth=8, learning_rate=0.08, n_estimators=300,
    subsample=0.75, colsample_bytree=0.75, reg_alpha=0.5, reg_lambda=1.0,
    tree_method="hist", random_state=42, eval_metric="mlogloss",
    early_stopping_rounds=30,
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
acc = accuracy_score(y_test, y_pred)

def top_k_acc(y_true, y_proba, k):
    topk_idx = np.argpartition(-y_proba, min(k, y_proba.shape[1]-1), axis=1)[:, :k]
    correct = np.any(topk_idx == y_true.reshape(-1, 1), axis=1)
    return correct.sum() / len(y_true)

top1 = acc
top3 = top_k_acc(y_test, y_proba, 3)
top5 = top_k_acc(y_test, y_proba, 5)
print(f"XGBoost: Top-1={top1:.1%} Top-3={top3:.1%} Top-5={top5:.1%}")

# Feature importance
importance = model.feature_importances_
feat_imp = sorted(zip(feature_cols, importance), key=lambda x: -x[1])
print("Feature importance:")
for name, imp in feat_imp[:5]:
    print(f"  {name}: {imp:.4f}")

# Load rules
rules_df = pd.read_csv(f"rules/{level.lower().replace('+','p')}_rules.csv")
comp_rd, py_rd, fin_rd = {}, {}, {}
for _, r in rules_df.iterrows():
    rt = r["rule_type"]
    conf = float(r["confidence"].rstrip("%")) / 100
    parts = r["rule_desc"].split(" → ")
    if len(parts) < 2:
        continue
    exp_on = parts[1]
    if rt == "声符规则":
        ck = parts[0]
        if ck not in comp_rd or conf > comp_rd[ck][1]:
            comp_rd[ck] = (exp_on, conf, r["rule_strength"])
    elif rt == "拼音桥接":
        pk = r.get("pinyin_key", "")
        if pk not in py_rd or conf > py_rd[pk][1]:
            py_rd[pk] = (exp_on, conf, r["rule_strength"])
    elif rt == "韵母桥接":
        fk = r.get("pinyin_key", "")
        if fk not in fin_rd or conf > fin_rd[fk][1]:
            fin_rd[fk] = (exp_on, conf, r["rule_strength"])

def predict_rules(row):
    nrp = str(row["non_radical_part"]) if pd.notna(row["non_radical_part"]) else ""
    init = str(row["pinyin_initial"]) if pd.notna(row["pinyin_initial"]) else ""
    final = str(row["pinyin_final"]) if pd.notna(row["pinyin_final"]) else ""
    py_key = f"{init}+{final}" if init else ""

    for lookup, key, min_strength in [
        (comp_rd, nrp, "确定"), (py_rd, py_key, "确定"),
        (comp_rd, nrp, "强"), (py_rd, py_key, "强"),
        (comp_rd, nrp, "弱"), (fin_rd, final, None),
    ]:
        if key and key in lookup:
            exp, conf, strength = lookup[key]
            if min_strength is None or strength >= min_strength:
                if exp in le_on.classes_:
                    on_id_orig = le_on.transform([exp])[0]
                    if on_id_orig in on_to_train:
                        return on_to_train[on_id_orig], conf
                    else:
                        return None, 0
    return None, 0

# Strategy comparison
rule_correct = rule_covered = 0
xgb_correct = 0
fused_correct = 0

for i, (idx, row) in enumerate(df_valid.loc[idx_test].iterrows()):
    X_row = X_test.iloc[i:i+1]
    true_id = y_test[i]

    rp, rc = predict_rules(row)
    if rp is not None:
        rule_covered += 1
        if rp == true_id:
            rule_correct += 1

    xp = model.predict(X_row)[0]
    if xp == true_id:
        xgb_correct += 1

    if rp is not None and rc >= 0.80:
        if rp == true_id:
            fused_correct += 1
    else:
        if xp == true_id:
            fused_correct += 1

n_test = len(y_test)
rule_acc = rule_correct / max(rule_covered, 1)
print(f"\n--- Strategy Comparison ({n_test} test) ---")
print(f"  rule    : {rule_correct}/{rule_covered} = {rule_acc:.1%} (cover {rule_covered/n_test:.1%})")
print(f"  xgb     : {xgb_correct}/{n_test} = {xgb_correct/n_test:.1%}")
print(f"  fused   : {fused_correct}/{n_test} = {fused_correct/n_test:.1%}")

# Error analysis
err_cats = defaultdict(int)
for i, (idx, row) in enumerate(df_valid.loc[idx_test].iterrows()):
    true_id = y_test[i]
    X_row = X_test.iloc[i:i+1]
    rp, rc = predict_rules(row)
    xp = model.predict(X_row)[0]
    if rp is not None and rc >= 0.80:
        fp = rp
    else:
        fp = xp
    if fp == true_id:
        continue
    on_all = str(row["onyomi_all"]) if pd.notna(row["onyomi_all"]) else ""
    on_list = [x.strip() for x in on_all.split("|") if x.strip()]
    nrp = str(row["non_radical_part"]) if pd.notna(row["non_radical_part"]) else ""
    if len(on_list) > 1:
        err_cats["多音歧义"] += 1
    elif not nrp or nrp == "nan":
        err_cats["无声符"] += 1
    else:
        err_cats["声符不可靠/拼音例外"] += 1

print("Errors:")
for cat, cnt in sorted(err_cats.items(), key=lambda x: -x[1]):
    print(f"  {cat}: {cnt}")

# Save
model.save_model(f"models/saved/{level.lower().replace('+','p')}_xgboost.json")
with open(f"models/saved/{level.lower().replace('+','p')}_label_encoders.pkl", "wb") as f:
    pickle.dump({
        "le_on": le_on, "le_train": le_train,
        "le_on_labels": le_on.classes_.tolist(),
        "le_train_labels": le_train.classes_.tolist(),
        "on_to_train": on_to_train,
    }, f)

print(f"\nSaved: models/saved/{level.lower().replace('+','p')}_xgboost.json")

# Summary
results = {
    "level": level, "n_kanji": len(df), "n_onyomi_classes": len(le_on.classes_),
    "n_train": len(X_train), "n_test": n_test,
    "xgb_top1": top1, "xgb_top3": top3, "xgb_top5": top5,
    "rule_coverage": rule_covered / n_test,
    "rule_accuracy": rule_acc,
    "fused_accuracy": fused_correct / n_test,
    "n_rules": len(rules_df),
    "top_features": [(n, float(i)) for n, i in feat_imp[:3]],
}
with open(f"output/{level.lower().replace('+','p')}_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print(f"{level} SUMMARY: xgb={top1:.1%} rule={rule_acc:.1%} fused={fused_correct/n_test:.1%}")
print(f"{'='*60}")
