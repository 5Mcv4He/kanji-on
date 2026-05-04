"""Kanji lookup CLI: rules + XGBoost predictions + context by JLPT level.
Usage:
    python kanji_lookup.py <漢字> [--level N3]
If --level not specified, uses N5 for N5-only kanji, or the first level where the kanji appears.
"""
import pandas as pd
import numpy as np
import sys
import os
import json
import pickle
from collections import defaultdict

LEVELS = ["N5", "N4", "N3", "N2", "N1"]

# ── Lazy loading ──
_datasets = {}
_rules = {}
_models = {}
_encoders = {}
_distilled = {}
_loaded = set()

def _load_level(lv):
    if lv in _loaded:
        return
    fbase = lv.lower().replace("+", "p")
    try:
        _datasets[lv] = pd.read_csv(f"dataset/{fbase}_dataset.csv")
        _rules[lv] = pd.read_csv(f"rules/{fbase}_rules.csv")
        with open(f"models/saved/{fbase}_label_encoders.pkl", "rb") as f:
            _encoders[lv] = pickle.load(f)
        import xgboost as xgb
        m = xgb.XGBClassifier()
        m.load_model(f"models/saved/{fbase}_xgboost.json")
        _models[lv] = m
        # Load distilled rules if available
        try:
            with open(f"output/{fbase}_xgboost_rules.json") as f:
                distilled_data = json.load(f)
            _distilled[lv] = distilled_data.get("xgb_distilled_rules", [])
        except (FileNotFoundError, KeyError):
            _distilled[lv] = []
        _loaded.add(lv)
    except Exception as e:
        print(f"  [WARN] {lv} load failed: {e}", file=sys.stderr)


def _match_distilled_rule(rule_text, row):
    """Check if a kanji row satisfies a distilled rule's conditions."""
    conditions = [c.strip() for c in rule_text.split(" AND ")]
    try:
        for cond in conditions:
            if "<" in cond and "≤" in cond:
                # Range: lo<feat≤hi
                rest, hi_str = cond.rsplit("≤", 1)
                lo_str, feat = rest.split("<", 1)
                lo, hi = float(lo_str), float(hi_str)
            elif "≤" in cond:
                feat, val_str = cond.split("≤", 1)
                lo, hi = float("-inf"), float(val_str)
            elif ">" in cond:
                feat, val_str = cond.split(">", 1)
                lo, hi = float(val_str), float("inf")
            elif "=" in cond:
                feat, val = cond.split("=", 1)
            else:
                return False

            if feat == "声符":
                if str(row.get("non_radical_part", "")) != val:
                    return False
            elif feat == "部首":
                if str(row.get("radical", "")) != val:
                    return False
            elif feat == "声母":
                if str(row.get("pinyin_initial", "")) != val:
                    return False
            elif feat == "韵母":
                if str(row.get("pinyin_final", "")) != val:
                    return False
            elif feat == "鼻音":
                if str(row.get("nasal_coda", "")) != val:
                    return False
            elif feat == "声调":
                if not pd.notna(row.get("pinyin_tone_num")):
                    return False
                v = float(row["pinyin_tone_num"])
                if not (lo < v <= hi if hi != float("inf") else v > lo):
                    return False
            elif feat == "总画数":
                if not pd.notna(row.get("total_strokes")):
                    return False
                v = float(row["total_strokes"])
                if not (lo < v <= hi if hi != float("inf") else v > lo):
                    return False
            elif feat == "部首画数":
                if not pd.notna(row.get("radical_strokes")):
                    return False
                v = float(row["radical_strokes"])
                if not (lo < v <= hi if hi != float("inf") else v > lo):
                    return False
            elif feat == "声符画数":
                if not pd.notna(row.get("non_radical_strokes")):
                    return False
                v = float(row["non_radical_strokes"])
                if not (lo < v <= hi if hi != float("inf") else v > lo):
                    return False
            else:
                return False  # unknown feature
        return True
    except (ValueError, TypeError, IndexError):
        return False


def _find_level(kanji):
    """Find the first level where this kanji appears with observed readings."""
    for lv in ["N5", "N4", "N3", "N2", "N1"]:
        _load_level(lv)
        ds = _datasets[lv]
        rows = ds[(ds["kanji"] == kanji) & (ds["observed_readings"].notna()) & (ds["observed_readings"] != "")]
        if len(rows) > 0:
            return lv, rows.iloc[0]
    return None, None


def lookup(kanji, level=None):
    if level:
        _load_level(level)
        ds = _datasets.get(level)
        if ds is None:
            print(f"  Level {level} not available.")
            return
        rows = ds[ds["kanji"] == kanji]
        if len(rows) == 0:
            print(f"  '{kanji}' not found at {level}, trying auto-detect...")
            level, row = _find_level(kanji)
        else:
            row = rows.iloc[0]
    else:
        level, row = _find_level(kanji)

    if row is None:
        print(f"'{kanji}' not found in any level dataset.")
        return

    # Basic info
    on_all = str(row["onyomi_all"]).replace("|", " · ") if pd.notna(row["onyomi_all"]) else "?"
    kun_all = str(row["kunyomi_all"]).replace("|", " · ") if pd.notna(row["kunyomi_all"]) else "?"
    meaning = str(row["meaning"]) if pd.notna(row["meaning"]) else "?"
    radical = str(row["radical"]) if pd.notna(row["radical"]) else "?"
    nonrad = str(row["non_radical_part"]) if pd.notna(row["non_radical_part"]) else ""
    pinyin_s = str(row["pinyin"]) if pd.notna(row["pinyin"]) else "?"
    pinyin_init = str(row["pinyin_initial"]) if pd.notna(row["pinyin_initial"]) else ""
    pinyin_final = str(row["pinyin_final"]) if pd.notna(row["pinyin_final"]) else ""
    py_key = f"{pinyin_init}+{pinyin_final}" if pinyin_init else ""
    strokes = int(row["total_strokes"]) if pd.notna(row["total_strokes"]) else 0
    onyomi = str(row["onyomi"]) if pd.notna(row["onyomi"]) else ""

    print(f"""
╔══════════════════════════════════════════════════════════╗
║  {kanji}  拼音:{pinyin_s}  JLPT:{level}  画数:{strokes}
╠══════════════════════════════════════════════════════════╣
║  音読: {on_all}
║  訓読: {kun_all}
║  意味: {meaning}
║  部首: {radical}    声符(非部首部): {nonrad or '(なし)'}
║  拼音声母+韵母: {py_key}
╠══════════════════════════════════════════════════════════╣""")

    # Observed readings at this level
    obs_raw = str(row["observed_readings"]) if pd.notna(row["observed_readings"]) else ""
    obs_list = [x.strip() for x in obs_raw.split("|") if x.strip()]
    print(f"║  本层实测读法: {' · '.join(obs_list[:10])}")

    # Sample words
    sw_raw = row.get("sample_words", "")
    if pd.notna(sw_raw) and sw_raw:
        try:
            words = json.loads(sw_raw)
            print(f"║  ── 含此字的单词({len(words)}件) ──")
            for w in words[:6]:
                print(f"║  {w['word']}　→　{w['kana']}")
        except:
            pass

    # ── Rules ──
    rules_df = _rules.get(level)
    matched_rules = []
    if rules_df is not None and len(rules_df) > 0:
        for _, r in rules_df.iterrows():
            rt = r["rule_type"]
            parts = r["rule_desc"].split(" → ")
            if len(parts) < 2:
                continue
            key_part = parts[0]
            if rt == "声符规则" and nonrad and key_part == nonrad:
                matched_rules.append(r)
            elif rt == "拼音桥接" and r.get("pinyin_key", "") == py_key:
                matched_rules.append(r)
            elif rt == "韵母桥接" and r.get("pinyin_key", "") == pinyin_final:
                matched_rules.append(r)

    if matched_rules:
        print(f"║  ── 匹配规则({len(matched_rules)}条) ──")
        for r in matched_rules[:8]:
            strength = r["rule_strength"]
            conf = r["confidence"]
            desc = r["rule_desc"]
            examples = str(r.get("examples", ""))[:50]
            match_mark = "✓" if desc.split(" → ")[-1] == onyomi else "✗"
            print(f"║  [{strength}] {desc} ({conf}, 例:{examples}) {match_mark}")
    else:
        print(f"║  [无匹配规则] {level}层无规则覆盖此汉字")

    # ── XGBoost prediction ──
    model = _models.get(level)
    enc = _encoders.get(level)
    if model and enc:
        try:
            preds = _xgb_predict(row, level, model, enc)
            if preds:
                print(f"║  ── XGBoost Top-5预测 ──")
                for i, (label, prob) in enumerate(preds):
                    marker = "✓" if label == onyomi else " "
                    print(f"║  {i+1}. {label:8s} {prob:.0%} {marker}")
        except Exception as e:
            print(f"║  [预测失败]: {e}")

    # ── Distilled rules (XGBoost → Decision Tree) ──
    distilled_list = _distilled.get(level, [])
    if distilled_list:
        matched_distilled = []
        for dr in distilled_list:
            if _match_distilled_rule(dr["rule"], row):
                matched_distilled.append(dr)
        if matched_distilled:
            # Keep best per on'yomi
            best_per_on = {}
            for dr in matched_distilled:
                on = dr["onyomi"]
                if on not in best_per_on or dr["accuracy"] > best_per_on[on]["accuracy"]:
                    best_per_on[on] = dr
            matched_distilled = sorted(best_per_on.values(), key=lambda x: -x["accuracy"])

            print(f"║  ── 蒸馏规则({len(matched_distilled)}条匹配) ──")
            for dr in matched_distilled[:5]:
                match = "✓" if dr["onyomi"] == onyomi else "→"
                print(f"║  IF {dr['rule']}")
                print(f"║     → {dr['onyomi']}  覆盖{dr['coverage']}字  准确率{dr['accuracy']:.0%} {match}")
                if dr.get("examples"):
                    print(f"║     例: {dr['examples'][:60]}")

    # ── Same-reading kanji ──
    if onyomi:
        ds = _datasets.get(level)
        if ds is not None:
            same = ds[(ds["onyomi"] == onyomi) & (ds["kanji"] != kanji)]["kanji"].tolist()
            if same:
                print(f"║  ── 同音字({len(same)}字) ──")
                for i in range(0, min(len(same), 40), 10):
                    print(f"║  {' '.join(same[i:i+10])}")

    print(f"╚══════════════════════════════════════════════════════════╝")


def _xgb_predict(row, level, model, enc):
    """Get XGBoost top-5 predictions for a kanji."""
    from sklearn.preprocessing import LabelEncoder as LE

    ds = _datasets[level]
    # Fit encoders on dataset to match training
    le_radical = LE()
    le_nonrad = LE()
    le_init = LE()
    le_final = LE()
    le_nasal = LE()
    le_radical.fit(ds["radical"].fillna(""))
    le_nonrad.fit(ds["non_radical_part"].fillna(""))
    le_init.fit(ds["pinyin_initial"].fillna(""))
    le_final.fit(ds["pinyin_final"].fillna(""))
    le_nasal.fit(ds["nasal_coda"].fillna("none"))

    vals = {}
    # Use short names matching training: nonrad_enc, init_enc, final_enc, nasal_enc
    for col, le, short in [("radical", le_radical, "radical_enc"),
                            ("non_radical_part", le_nonrad, "nonrad_enc"),
                            ("pinyin_initial", le_init, "init_enc"),
                            ("pinyin_final", le_final, "final_enc"),
                            ("nasal_coda", le_nasal, "nasal_enc")]:
        raw = str(row[col]) if pd.notna(row[col]) else ""
        try:
            vals[short] = le.transform([raw])[0]
        except:
            vals[short] = 0

    # Detect model's actual feature set (some levels lack is_entering_tone)
    _model_feats = set(model.get_booster().feature_names)
    _all_feats = ["radical_enc", "nonrad_enc", "init_enc", "final_enc",
                  "pinyin_tone_num", "nasal_enc",
                  "radical_strokes", "non_radical_strokes", "total_strokes",
                  "is_entering_tone"]
    feature_cols = [f for f in _all_feats if f in _model_feats]
    vals["pinyin_tone_num"] = int(row["pinyin_tone_num"]) if pd.notna(row["pinyin_tone_num"]) else 0
    vals["radical_strokes"] = int(row["radical_strokes"]) if pd.notna(row["radical_strokes"]) else 0
    vals["non_radical_strokes"] = int(row["non_radical_strokes"]) if pd.notna(row["non_radical_strokes"]) else 0
    vals["total_strokes"] = int(row["total_strokes"]) if pd.notna(row["total_strokes"]) else 0
    vals["is_entering_tone"] = int(row["is_entering_tone"]) if pd.notna(row.get("is_entering_tone")) else 0

    X_row = pd.DataFrame([vals])[feature_cols].fillna(0).astype(float)
    probs = model.predict_proba(X_row)[0]

    # Map back to on'yomi labels
    le_train = enc["le_train"]
    le_on = enc["le_on"]
    on_to_train = enc.get("on_to_train", {})
    train_to_on = {v: k for k, v in on_to_train.items()}

    top_idx = np.argsort(-probs)[:5]
    results = []
    for idx in top_idx:
        if idx in train_to_on:
            on_label_id = train_to_on[idx]
            label = le_on.classes_[on_label_id]
        elif idx < len(le_train.classes_):
            label = str(le_train.classes_[idx])
        else:
            label = f"cls{idx}"
        results.append((label, float(probs[idx])))
    return results


if __name__ == "__main__":
    kanji = sys.argv[1] if len(sys.argv) > 1 else None
    if not kanji:
        print("Usage: python kanji_lookup.py <漢字> [--level N3]")
        print("Example: python kanji_lookup.py 学 --level N4")
        sys.exit(1)

    level = None
    if "--level" in sys.argv:
        idx = sys.argv.index("--level")
        if idx + 1 < len(sys.argv):
            level = sys.argv[idx + 1].upper()

    lookup(kanji, level)
