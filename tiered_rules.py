"""Exhaustive tiered rule discovery — ALL human-language-describable rules.

Tiers (by learning difficulty):
  1 (★): Single feature, easiest            e.g. f+ang → ホウ
  1.5 (★): MC single feature                 e.g. 見母 → カ行
  2 (★★): Single component/structure         e.g. 声符「包」→ ホウ
  3 (★★★): Two features, pairs               e.g. MC声母+韵摄 → カイ
  4 (★★★★): Mined compound, triple           e.g. 声母=j AND 韵母=ian AND 画数≤13
  5 (★★★★★): Complex distilled patterns

Confidence levels (NOT hard filters):
  确定 (>=80%), 大概率 (60-80%), 有时 (40-60%), 偶尔 (<40%)

Generates DUAL output:
  - exhaustive: ALL discovered rules with confidence tags
  - minimal:    greedy set-cover selected subset
"""

import pandas as pd
import numpy as np
import sys, json, os, re, pickle
from collections import defaultdict, Counter
from itertools import combinations
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder as LE

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from predict_go_kan import classify_rule_reading_type

LEVEL = sys.argv[1] if len(sys.argv) > 1 else "N1"
FBASE = LEVEL.lower().replace("+", "p")

print(f"{'='*60}")
print(f"Exhaustive Rule Discovery — {LEVEL} (ALL confidence levels)")
print(f"{'='*60}")

# ═══════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════
df = pd.read_csv(f"dataset/{FBASE}_dataset.csv")
df = df[df["onyomi"].notna() & (df["onyomi"] != "")].copy()
df = df[df["observed_readings"].notna() & (df["observed_readings"] != "")].copy()
df = df.reset_index(drop=True)
n_total = len(df)
print(f"Dataset: {n_total} kanji, {df['onyomi'].nunique()} on'yomi classes")

# ── Precompute columns ──
df["init"] = df["pinyin_initial"].fillna("")
df["final"] = df["pinyin_final"].fillna("")
df["final_nt"] = df["final"].str.replace(r"\d", "", regex=True)
df["inf"] = df["init"] + "+" + df["final_nt"]
df["ift"] = df["init"] + "+" + df["final"]
df["nrp"] = df["non_radical_part"].fillna("")
df["rad"] = df["radical"].fillna("")
df["nasal"] = df["nasal_coda"].fillna("none")
df["strokes_bkt"] = pd.cut(df["total_strokes"].fillna(0), bins=[0,8,12,16,200],
                           labels=["≤8", "9-12", "13-16", "17+"])
df["tone"] = df["pinyin_tone_num"].fillna(0).astype(int)
df["ent"] = df["is_entering_tone"].fillna(0).astype(int)
# MC features
df["mc_init"] = df["mc_initial"].fillna("")
df["mc_rhyme"] = df["mc_rhyme"].fillna("")
df["mc_grade"] = df["mc_grade"].fillna("")
df["mc_open"] = df["mc_openness"].fillna("")
df["mc_tone"] = df["mc_tone"].fillna("")
df["mc_voice"] = df["mc_voicing"].fillna("")
df["mc_rhyme_cat"] = df["mc_rhyme_cat"].fillna("")
df["mc_init_cat"] = df["mc_initial_cat"].fillna("")
df["mc_entering_coda"] = df["mc_entering_coda"].fillna("")
df["mc_divergent"] = df.get("mc_divergent", pd.Series([0]*n_total)).fillna(0).astype(int)
# Total strokes raw
df["strokes_raw"] = df["total_strokes"].fillna(0).astype(int)
onyomi_s = df["onyomi"]

# ═══════════════════════════════════════════════════════════════
# FEATURE DEFINITIONS
# ═══════════════════════════════════════════════════════════════

# Standalone feature groups
MC_SINGLE_FEATURES = {
    "mc_init":      ("MC声母",     "中古音·声母",     "mc_initial"),
    "mc_rhyme":     ("MC韵母",     "中古音·韵母",     "mc_rhyme"),
    "mc_tone":      ("MC声调",     "中古音·声调",     "mc_tone"),
    "mc_grade":     ("MC等",       "中古音·等",       "mc_grade"),
    "mc_open":      ("MC開合",     "中古音·開合",     "mc_openness"),
    "mc_voice":     ("MC清濁",     "中古音·清濁",     "mc_voicing"),
    "mc_rhyme_cat": ("MC韵摄",     "中古音·韵摄",     "mc_rhyme_cat"),
    "mc_init_cat":  ("MC声类",     "中古音·五音",     "mc_initial_cat"),
}

TIER2_FEATURES = {
    "nrp":          ("声符",       "声符·精确",       "non_radical_part"),
    "rad":          ("部首",       "结构·部首",       "radical"),
    "nasal":        ("鼻音",       "拼音·鼻音",       "nasal_coda"),
    "ent":          ("入声",       "拼音·入声",       "is_entering_tone"),
    "strokes_bkt":  ("画数组",     "结构·画数组",     "total_strokes"),
    "mc_entering_coda": ("入声韵尾", "中古音·入声韵尾", "mc_entering_coda"),
}

# MC feature columns for pair generation
MC_FEATURE_COLS = ["mc_init", "mc_rhyme", "mc_grade", "mc_open",
                   "mc_tone", "mc_voice", "mc_rhyme_cat", "mc_init_cat"]

MC_FEATURE_LABELS = {
    "mc_init": "MC声母", "mc_rhyme": "MC韵母", "mc_grade": "MC等",
    "mc_open": "MC開合", "mc_tone": "MC声调", "mc_voice": "MC清濁",
    "mc_rhyme_cat": "MC韵摄", "mc_init_cat": "MC声类",
}

# Cross-domain pair definitions: (col1, col2, label1, label2, source_label)
CROSS_DOMAIN_PAIRS = [
    # MC × Pinyin
    ("mc_init",   "final_nt",  "MC声母", "拼音韵母", "中古×拼音·声母+韵母"),
    ("mc_rhyme",  "init",      "MC韵母", "拼音声母", "中古×拼音·韵母+声母"),
    ("mc_grade",  "tone",      "MC等",   "拼音声调", "中古×拼音·等+声调"),
    ("mc_voice",  "final_nt",  "MC清濁", "拼音韵母", "中古×拼音·清濁+韵母"),
    ("mc_rhyme_cat","final_nt","MC韵摄", "拼音韵母", "中古×拼音·韵摄+韵母"),
    ("mc_open",   "final_nt",  "MC開合", "拼音韵母", "中古×拼音·開合+韵母"),
    ("mc_tone",   "final_nt",  "MC声调", "拼音韵母", "中古×拼音·声调+韵母"),
    ("mc_init_cat","final_nt", "MC声类", "拼音韵母", "中古×拼音·声类+韵母"),
    ("mc_init",   "init",      "MC声母", "拼音声母", "中古×拼音·声母+声母"),
    ("mc_rhyme",  "final_nt",  "MC韵母", "拼音韵母", "中古×拼音·韵母+韵母"),
    ("mc_voice",  "init",      "MC清濁", "拼音声母", "中古×拼音·清濁+声母"),
    ("mc_rhyme_cat","init",    "MC韵摄", "拼音声母", "中古×拼音·韵摄+声母"),
    # Component × MC
    ("nrp",       "mc_rhyme",  "声符",   "MC韵母",   "声符×中古·声符+韵母"),
    ("nrp",       "mc_voice",  "声符",   "MC清濁",   "声符×中古·声符+清濁"),
    ("nrp",       "mc_rhyme_cat","声符", "MC韵摄",   "声符×中古·声符+韵摄"),
    ("nrp",       "mc_tone",   "声符",   "MC声调",   "声符×中古·声符+声调"),
    ("nrp",       "mc_init",   "声符",   "MC声母",   "声符×中古·声符+声母"),
    # Component × Pinyin
    ("nrp",       "final_nt",  "声符",   "拼音韵母", "声符×拼音·声符+韵母"),
    ("nrp",       "init",      "声符",   "拼音声母", "声符×拼音·声符+声母"),
    ("nrp",       "nasal",     "声符",   "鼻音",     "声符×拼音·声符+鼻音"),
    # Radical × MC
    ("rad",       "mc_rhyme",  "部首",   "MC韵母",   "部首×中古·部首+韵母"),
    ("rad",       "mc_voice",  "部首",   "MC清濁",   "部首×中古·部首+清濁"),
    ("rad",       "mc_rhyme_cat","部首", "MC韵摄",   "部首×中古·部首+韵摄"),
    ("rad",       "mc_init",   "部首",   "MC声母",   "部首×中古·部首+声母"),
    # Radical × Pinyin
    ("rad",       "final_nt",  "部首",   "拼音韵母", "部首×拼音·部首+韵母"),
    ("rad",       "init",      "部首",   "拼音声母", "部首×拼音·部首+声母"),
    ("rad",       "nasal",     "部首",   "鼻音",     "部首×拼音·部首+鼻音"),
    ("rad",       "strokes_bkt","部首",  "画数组",   "部首×结构·部首+画数"),
    # MC × Structure
    ("mc_rhyme_cat","nasal",   "MC韵摄", "鼻音",     "中古×拼音·韵摄+鼻音"),
    ("mc_rhyme_cat","mc_entering_coda","MC韵摄","入声韵尾","中古音·韵摄+入声韵尾"),
    ("mc_tone",   "mc_entering_coda","MC声调","入声韵尾","中古音·声调+入声韵尾"),
    # Structure
    ("strokes_bkt","nasal",    "画数组",  "鼻音",    "结构×拼音·画数+鼻音"),
    ("strokes_bkt","ent",      "画数组",  "入声",    "结构×拼音·画数+入声"),
]

def col_is_valid(col):
    """Check if column exists and has non-empty values."""
    return col in df.columns and (df[col] != "").sum() > 0


# ═══════════════════════════════════════════════════════════════
# RULE GENERATION ENGINE
# ═══════════════════════════════════════════════════════════════
all_rules = []

def confidence_label(acc):
    if acc >= 0.80: return "确定"
    if acc >= 0.60: return "大概率"
    if acc >= 0.40: return "有时"
    return "偶尔"

def add_rule(text, on, src, tier, mask, feature_type="general", reading_type="general",
             correct_examples=None, exception_examples=None):
    """Add a rule — NO hard accuracy cutoff. All rules kept with confidence tag."""
    n_match = int(mask.sum())
    if n_match < 2:
        return
    n_correct = int((mask & (onyomi_s.values == on)).sum())
    if n_correct < 1:
        return  # No correct predictions = not a rule, just noise
    acc = n_correct / n_match

    correct_indices = set(np.where(mask & (onyomi_s.values == on))[0].tolist())
    conf = confidence_label(acc)

    # Collect example kanji
    if correct_examples is None:
        correct_kanji = df.iloc[list(correct_indices)]["kanji"].tolist()[:6]
        correct_examples = correct_kanji
    if exception_examples is None:
        wrong_mask = mask & (onyomi_s.values != on)
        if wrong_mask.sum() > 0:
            wrong_rows = df[wrong_mask]
            exception_examples = [
                {"kanji": str(r["kanji"]), "actual": str(r["onyomi"])}
                for _, r in wrong_rows.head(3).iterrows()
            ]
        else:
            exception_examples = []

    all_rules.append({
        "rule_text": text,
        "onyomi": on,
        "source": src,
        "tier": tier,
        "accuracy": acc,
        "matched": n_match,
        "correct": n_correct,
        "correct_indices": correct_indices,
        "confidence": conf,
        "feature_type": feature_type,
        "reading_type": reading_type,
        "correct_examples": correct_examples,
        "exception_examples": exception_examples,
        "weak_flag": acc < 0.40,
    })

def gen_single_feature_rules(col, label_template, source, tier, feature_type,
                            value_map=None):
    """Generate rules for a single feature column → onyomi.
    value_map: optional dict mapping raw values → display labels (e.g. {0: '非入声', 1: '入声'})
    """
    for val, grp in df.groupby(col):
        val_str = str(val)
        if val_str == "" or val_str == "nan" or val_str == "none" or len(grp) < 2:
            continue
        display_val = value_map.get(val, val_str) if value_map else val_str
        display_val = str(display_val)
        for on, ong in grp.groupby("onyomi"):
            text = label_template.format(val=display_val, on=on)
            add_rule(text, on, source, tier, (df[col] == val),
                     feature_type=feature_type)

def gen_pair_rules(col1, col2, label1, label2, source, tier, feature_type,
                   reading_type="general", min_support=3, value_maps=None,
                   reading_type_map=None):
    """Generate rules for a feature pair → onyomi.
    value_maps: optional dict {col: {raw_val: display_val}} for human-readable labels
    reading_type_map: optional dict {compound_key: reading_type} for Go/Kan annotation
    """
    if value_maps is None:
        value_maps = {}
    if reading_type_map is None:
        reading_type_map = {}
    compound = df[col1].astype(str) + "+" + df[col2].astype(str)
    for key, grp in df.groupby(compound):
        if "+" not in key or "++" in key or len(grp) < min_support:
            continue
        v1, v2 = key.split("+", 1)
        if not v1 or not v2 or v1 == "nan" or v2 == "nan":
            continue
        # Convert to display values
        d1 = str(value_maps.get(col1, {}).get(v1, v1)) if col1 in value_maps else v1
        d2 = str(value_maps.get(col2, {}).get(v2, v2)) if col2 in value_maps else v2
        # Also try typed keys for int-valued columns
        try:
            if col1 in value_maps and v1.isdigit():
                d1 = str(value_maps[col1].get(int(v1), v1))
        except: pass
        try:
            if col2 in value_maps and v2.isdigit():
                d2 = str(value_maps[col2].get(int(v2), v2))
        except: pass
        rt = reading_type_map.get(key, reading_type)
        for on, ong in grp.groupby("onyomi"):
            mask = (df[col1].astype(str) == v1) & (df[col2].astype(str) == v2)
            text = f"{label1}「{d1}」+{label2}「{d2}」→ {on}"
            add_rule(text, on, source, tier, mask,
                     feature_type=feature_type, reading_type=rt)


# ═══════════════════════════════════════════════════════════════
# TIER 1: Single pinyin feature (easiest)
# ═══════════════════════════════════════════════════════════════
print("\nTier 1: Single pinyin features...")
# final (no tone) → onyomi
for fnt, grp in df.groupby("final_nt"):
    if fnt == "" or len(grp) < 3: continue
    for on, ong in grp.groupby("onyomi"):
        add_rule(f"韵母 -{fnt} → {on}", on, "拼音·韵母", 1, (df["final_nt"] == fnt),
                 feature_type="pinyin_standalone")
# init+final (no tone) → onyomi
for inf, grp in df.groupby("inf"):
    if inf == "+" or len(grp) < 2: continue
    for on, ong in grp.groupby("onyomi"):
        add_rule(f"拼音 {inf} → {on}", on, "拼音·声+韵", 1, (df["inf"] == inf),
                 feature_type="pinyin_standalone")

# ═══════════════════════════════════════════════════════════════
# TIER 1.5: MC single feature
# ═══════════════════════════════════════════════════════════════
print("Tier 1.5: MC single feature rules...")
for col, (label, source, orig_col) in MC_SINGLE_FEATURES.items():
    gen_single_feature_rules(col, f"{label}「{{val}}」→ {{on}}", source, 1,
                             feature_type="mc_standalone")

# ═══════════════════════════════════════════════════════════════
# TIER 2: Single component/structure features
# ═══════════════════════════════════════════════════════════════
print("Tier 2: Component/structure features...")

# ── V1+V2 phonetic series ──
v1 = pd.read_excel(f'{HERE}/KanjiKensaku.xls', sheet_name='漢字一覧')
v2 = pd.read_excel(f'{HERE}/漢字検索V2.xlsm', sheet_name='漢字一覧')
kanji_comps = {}
for _, r in v1.iterrows():
    k = str(r['漢字']) if pd.notna(r['漢字']) else ''
    cr = str(r['構成文字']) if pd.notna(r['構成文字']) else ''
    if k and cr and cr != 'nan':
        kanji_comps[k] = [c for c in cr if '一' <= c <= '鿿' or '㐀' <= c <= '䶿']
for _, r in v2.iterrows():
    k = str(r['漢字']) if pd.notna(r['漢字']) else ''
    nrp = str(r['非部首部']) if pd.notna(r['非部首部']) else ''
    if k and nrp and nrp != 'nan' and nrp != '':
        cl = re.sub(r'^\d+', '', nrp).split('-')[0].strip()
        if cl:
            if k not in kanji_comps: kanji_comps[k] = []
            if cl not in kanji_comps[k]: kanji_comps[k].append(cl)

comp_onyomi = defaultdict(Counter)
for k, comps in kanji_comps.items():
    rows = df[df['kanji'] == k]
    if len(rows) == 0: continue
    on = rows.iloc[0]['onyomi']
    for c in comps: comp_onyomi[c][on] += 1

for comp, on_ctr in comp_onyomi.items():
    total = sum(on_ctr.values())
    if total < 2: continue
    for on, cnt in on_ctr.items():
        if cnt >= 1:
            mk = {k for k, cs in kanji_comps.items() if comp in cs}
            add_rule(f"声符「{comp}」→ {on}", on, "声符·V1V2", 2, df['kanji'].isin(mk),
                     feature_type="component")

# ── Exact non_radical_part ──
gen_single_feature_rules("nrp", "声符 {val} → {on}", "声符·精确", 2,
                         feature_type="component")

# ── Radical standalone ──
gen_single_feature_rules("rad", "部首「{val}」→ {on}", "结构·部首", 2,
                         feature_type="structure")

# ── Nasal coda ──
NASAL_MAP = {"-ng": "鼻音韵尾-ng", "-n": "鼻音韵尾-n", "none": "非鼻音"}
gen_single_feature_rules("nasal", "鼻音={val} → {on}", "拼音·鼻音", 2,
                         feature_type="structure", value_map=NASAL_MAP)

# ── Entering tone ──
ENT_MAP = {0: "非入声(舒声)", 1: "入声"}
gen_single_feature_rules("ent", "{val} → {on}", "拼音·入声", 2,
                         feature_type="structure", value_map=ENT_MAP)

# ── Stroke bucket ──
gen_single_feature_rules("strokes_bkt", "画数{val} → {on}", "结构·画数", 2,
                         feature_type="structure")

# ── Entering coda type ──
gen_single_feature_rules("mc_entering_coda", "入声韵尾「{val}」→ {on}",
                         "中古音·入声韵尾", 2, feature_type="structure")

# ═══════════════════════════════════════════════════════════════
# TIER 3: Two-feature pairs (ALL combinations)
# ═══════════════════════════════════════════════════════════════
print("Tier 3: Two-feature rules (pinyin tone, MC-internal, cross-domain)...")

# ── 3a: Pinyin init+final+tone ──
for ift, grp in df.groupby("ift"):
    if ift == "+" or len(grp) < 2: continue
    for on, ong in grp.groupby("onyomi"):
        add_rule(f"拼音 {ift} → {on}", on, "拼音·声+韵+调", 3, (df["ift"] == ift),
                 feature_type="pinyin_pair")

# ── 3b: Pinyin final+tone ──
for fin, grp in df.groupby("final"):
    if fin == "" or len(grp) < 3: continue
    for on, ong in grp.groupby("onyomi"):
        add_rule(f"韵母 -{fin} → {on}", on, "拼音·韵母+调", 3, (df["final"] == fin),
                 feature_type="pinyin_pair")

# ── 3c: ALL MC-internal pairs (28 combinations = C(8,2)) ──
print("  MC-internal pairs (all 28 combinations)...")
mc_pair_count = 0
# Map column names to the keys expected by classify_rule_reading_type
MC_COL_TO_KEY = {
    "mc_init": "initial", "mc_rhyme": "rhyme", "mc_grade": "grade",
    "mc_open": "openness", "mc_tone": "tone", "mc_voice": "voicing",
    "mc_rhyme_cat": "rhyme_category", "mc_init_cat": "initial_category",
}
for col1, col2 in combinations(MC_FEATURE_COLS, 2):
    label1 = MC_FEATURE_LABELS[col1]
    label2 = MC_FEATURE_LABELS[col2]
    src = f"中古音·{label1}+{label2}"
    # Build reading type lookup for each pair value from MC features
    # Pre-compute reading types per feature-value pair
    rt_map = {}
    compound = df[col1].astype(str) + "+" + df[col2].astype(str)
    for key, grp in compound.groupby(compound):
        if len(grp) == 0:
            continue
        v1, v2 = key.split("+", 1)
        mc_dict = {}
        if col1 in MC_COL_TO_KEY: mc_dict[MC_COL_TO_KEY[col1]] = v1
        if col2 in MC_COL_TO_KEY: mc_dict[MC_COL_TO_KEY[col2]] = v2
        rt_map[key] = classify_rule_reading_type(mc_dict)
    gen_pair_rules(col1, col2, label1, label2, src, 3, feature_type="mc_internal_pair",
                   reading_type_map=rt_map)
    mc_pair_count += 1
print(f"    Generated pairs for {mc_pair_count} MC feature combinations")

# ── 3d: Cross-domain pairs ──
print("  Cross-domain pairs...")
# Value maps for human-readable labels in pair rules
PAIR_VALUE_MAPS = {
    "nasal": {"-ng": "鼻音韵尾-ng", "-n": "鼻音韵尾-n", "none": "非鼻音"},
    "ent": {0: "非入声(舒声)", 1: "入声"},
}
cross_count = 0
for col1, col2, label1, label2, src in CROSS_DOMAIN_PAIRS:
    if not col_is_valid(col1) or not col_is_valid(col2):
        continue
    gen_pair_rules(col1, col2, label1, label2, src, 3,
                   feature_type="cross_domain_pair", min_support=3,
                   value_maps=PAIR_VALUE_MAPS)
    cross_count += 1
print(f"    Generated pairs for {cross_count} cross-domain feature combinations")

# ═══════════════════════════════════════════════════════════════
# TIER 4-5: Enhanced XGBoost mining
# ═══════════════════════════════════════════════════════════════
print("Tier 4-5: Enhanced mining...")

# Load XGBoost
print("  Loading XGBoost...")
with open(f"models/saved/{FBASE}_label_encoders.pkl", "rb") as f:
    enc = pickle.load(f)
model = xgb.XGBClassifier()
model.load_model(f"models/saved/{FBASE}_xgboost.json")

# Fit encoders
le_rad = LE(); le_rad.fit(df["rad"])
le_nrp = LE(); le_nrp.fit(df["nrp"])
le_ini = LE(); le_ini.fit(df["init"])
le_fin = LE(); le_fin.fit(df["final"])
le_nas = LE(); le_nas.fit(df["nasal"])

_feat_names = set(model.get_booster().feature_names)
_feat_all = ["radical_enc","nonrad_enc","init_enc","final_enc","pinyin_tone_num",
             "nasal_enc","radical_strokes","non_radical_strokes","total_strokes","is_entering_tone"]
feat_cols = [f for f in _feat_all if f in _feat_names]

X_all = []
for _, row in df.iterrows():
    v = {}
    for col, le, sn in [("rad", le_rad, "radical_enc"), ("nrp", le_nrp, "nonrad_enc"),
                         ("init", le_ini, "init_enc"), ("final", le_fin, "final_enc"),
                         ("nasal", le_nas, "nasal_enc")]:
        raw = str(row[col]) if pd.notna(row[col]) else ""
        try: v[sn] = le.transform([raw])[0]
        except: v[sn] = 0
    for fn in ["pinyin_tone_num","radical_strokes","non_radical_strokes","total_strokes","is_entering_tone"]:
        v[fn] = float(row[fn]) if pd.notna(row.get(fn)) else 0.0
    X_all.append(v)
X_all = pd.DataFrame(X_all)[feat_cols].fillna(0).astype(float)

probs = model.predict_proba(X_all)
preds = np.argmax(probs, axis=1)
le_train = enc["le_train"]; le_on = enc["le_on"]
on_to_train = enc.get("on_to_train", {})
train_to_on = {v: k for k, v in on_to_train.items()}

xgb_correct = np.zeros(n_total, dtype=bool)
for i, pred_id in enumerate(preds):
    if pred_id in train_to_on:
        on_id = train_to_on[pred_id]
        lbl = le_on.classes_[on_id]
        xgb_correct[i] = (lbl == df.iloc[i]["onyomi"])
print(f"  XGBoost Top-1 accuracy: {xgb_correct.sum()}/{n_total} = {xgb_correct.mean():.1%}")

# Target: ALL kanji where XGBoost is correct (not just uncovered)
# This allows Tier 4 to discover compound patterns even on "covered" kanji
xgb_correct_indices = set(np.where(xgb_correct)[0])
print(f"  XGBoost correct total: {len(xgb_correct_indices)}/{n_total}")
# Also add kanji not covered by any Tier 1-2 rule (higher priority)
t12_indices = set()
for r in all_rules:
    if r["tier"] <= 2:
        t12_indices |= r["correct_indices"]
uncovered_t12 = set(range(n_total)) - t12_indices
print(f"  Uncovered by Tiers 1-2: {len(uncovered_t12)}/{n_total}")
# Mine on: XGBoost-correct + Tier1-2-uncovered (higher weight), then all xgb-correct
xgb_correct_uncovered = (uncovered_t12 & xgb_correct_indices) | xgb_correct_indices

# Value maps for human-readable mining labels
FV_NASAL_MAP = {"-ng": "鼻音韵尾-ng", "-n": "鼻音韵尾-n", "none": "非鼻音"}
FV_ENT_MAP = {0: "非入声", 1: "入声字"}
FV_DIVERGENT_MAP = {0: "一致", 1: "有異"}

# Enhanced feature-value matrix (no len(nrp) <= 4 filter)
fv_matrix = []
for i in range(n_total):
    row = df.iloc[i]
    fvs = set()
    fvs.add(f"声母={row['init']}") if row['init'] else None
    fvs.add(f"韵母={row['final_nt']}") if row['final_nt'] else None
    fvs.add(f"韵母+调={row['final']}") if row['final'] else None
    fvs.add(f"鼻音={FV_NASAL_MAP.get(row['nasal'], row['nasal'])}")
    fvs.add(f"入声={FV_ENT_MAP.get(row['ent'], row['ent'])}")
    fvs.add(f"声调={row['tone']}")
    fvs.add(f"画数={row['strokes_bkt']}")
    fvs.add(f"画数_raw={row['strokes_raw']}")
    fvs.add(f"部首={row['rad']}") if row['rad'] and row['rad'] != 'nan' else None
    nrp_val = row['nrp']
    if nrp_val and nrp_val != 'nan':
        fvs.add(f"声符={nrp_val}")
    # All MC features
    fvs.add(f"MC声母={row['mc_init']}") if row['mc_init'] else None
    fvs.add(f"MC韵母={row['mc_rhyme']}") if row['mc_rhyme'] else None
    fvs.add(f"MC等={row['mc_grade']}") if row['mc_grade'] else None
    fvs.add(f"MC開合={row['mc_open']}") if row['mc_open'] else None
    fvs.add(f"MC声调={row['mc_tone']}") if row['mc_tone'] else None
    fvs.add(f"MC清濁={row['mc_voice']}") if row['mc_voice'] else None
    fvs.add(f"MC韵摄={row['mc_rhyme_cat']}") if row['mc_rhyme_cat'] else None
    fvs.add(f"MC声类={row['mc_init_cat']}") if row['mc_init_cat'] else None
    fvs.add(f"MC入声韵尾={row['mc_entering_coda']}") if row['mc_entering_coda'] else None
    fvs.add(f"呉漢異={FV_DIVERGENT_MAP.get(row['mc_divergent'], str(row['mc_divergent']))}")
    fv_matrix.append(fvs)

onyomi_uncovered = defaultdict(list)
for idx in xgb_correct_uncovered:
    on = df.iloc[idx]["onyomi"]
    onyomi_uncovered[on].append(idx)

# Col map for mining (expanded)
COL_MAP = {
    "声母": "init", "韵母": "final_nt", "韵母+调": "final",
    "鼻音": "nasal", "入声": "ent", "声调": "tone",
    "画数": "strokes_bkt", "画数_raw": "strokes_raw", "部首": "rad", "声符": "nrp",
    "MC声母": "mc_init", "MC韵母": "mc_rhyme", "MC等": "mc_grade",
    "MC開合": "mc_open", "MC声调": "mc_tone", "MC清濁": "mc_voice",
    "MC韵摄": "mc_rhyme_cat", "MC声类": "mc_init_cat",
    "MC入声韵尾": "mc_entering_coda", "呉漢異": "mc_divergent",
}

# Reverse value maps for mask creation (human-readable -> raw)
FV_REV_MAP = {
    "nasal": {"鼻音韵尾-ng": "-ng", "鼻音韵尾-n": "-n", "非鼻音": "none"},
    "ent": {"非入声": "0", "入声字": "1"},
    "mc_divergent": {"一致": "0", "有異": "1"},
}

def _mining_val(col_name, display_val):
    """Translate a human-readable value back to raw for mask creation."""
    if col_name in FV_REV_MAP:
        return FV_REV_MAP[col_name].get(display_val, display_val)
    return display_val

tier4_count = 0
tier5_count = 0

for on, indices in onyomi_uncovered.items():
    if len(indices) < 3:
        continue

    fv_counts = Counter()
    for idx in indices:
        for fv in fv_matrix[idx]:
            fv_counts[fv] += 1

    thresh = max(2, int(len(indices) * 0.35))
    common_fvs = {fv for fv, cnt in fv_counts.items() if cnt >= thresh}

    # Pairs (Tier 4)
    for fv1, fv2 in combinations(sorted(common_fvs), 2):
        ft1, ft2 = fv1.split("=")[0], fv2.split("=")[0]
        if ft1 == ft2:
            continue

        mask = np.ones(n_total, dtype=bool)
        for fv in [fv1, fv2]:
            feat, val = fv.split("=", 1)
            col = COL_MAP.get(feat)
            if col:
                mask &= (df[col].astype(str) == str(_mining_val(col, val)))

        if mask.sum() < 3:
            continue

        correct_mask = mask & (onyomi_s.values == on)
        acc = correct_mask.sum() / mask.sum()
        if correct_mask.sum() >= 2:
            rule_text = f"{fv1} AND {fv2} → {on}"
            src = "XGBoost·双层"
            add_rule(rule_text, on, src, 4, mask, feature_type="xgboost_mining")
            tier4_count += 1

    # Triples (Tier 5) — adaptive limit
    if len(indices) >= 5:
        fv_list = sorted(common_fvs)
        # Adaptive: use up to 15 features for triple mining
        limit = min(len(fv_list), 15)
        for fv1, fv2, fv3 in combinations(fv_list[:limit], 3):
            fts = {fv1.split("=")[0], fv2.split("=")[0], fv3.split("=")[0]}
            if len(fts) < 3:
                continue

            mask = np.ones(n_total, dtype=bool)
            for fv in [fv1, fv2, fv3]:
                feat, val = fv.split("=", 1)
                col = COL_MAP.get(feat)
                if col:
                    mask &= (df[col].astype(str) == str(_mining_val(col, val)))

            if mask.sum() < 4:
                continue
            correct_mask = mask & (onyomi_s.values == on)
            acc = correct_mask.sum() / mask.sum()
            if correct_mask.sum() >= 3:
                rule_text = f"{fv1} AND {fv2} AND {fv3} → {on}"
                add_rule(rule_text, on, "XGBoost·三层", 5, mask, feature_type="xgboost_mining")
                tier5_count += 1

print(f"  Tier 4 (2-condition): {tier4_count} rules")
print(f"  Tier 5 (3-condition): {tier5_count} rules")

# ── Distilled DT rules ──
print("Adding distilled DT rules...")
try:
    with open(f"output/{FBASE}_xgboost_rules.json") as f:
        dd = json.load(f)
    from kanji_lookup import _match_distilled_rule
    for dr in dd.get("xgb_distilled_rules", []):
        rt, on = dr["rule"], dr["onyomi"]
        mask = np.zeros(n_total, dtype=bool)
        for ki, (_, row) in enumerate(df.iterrows()):
            try:
                if _match_distilled_rule(rt, row): mask[ki] = True
            except: pass
        add_rule(f"IF {rt}", on, "蒸馏·DT", 5, pd.Series(mask),
                 feature_type="distilled")
except Exception as e:
    print(f"  (distilled rules skipped: {e})")

# ═══════════════════════════════════════════════════════════════
# SUMPTION CLEANUP: Remove T3 tone rules when T1 no-tone rule subsumes them
# ═══════════════════════════════════════════════════════════════
print(f"\nTotal raw rules: {len(all_rules)}")

# Build T1 pinyin lookup: "d+ong" → {onyomi → accuracy}
import re as re_sump
t1_pinyin_map = defaultdict(dict)  # (init+final_nt) → {onyomi: best_accuracy}
for r in all_rules:
    if r["tier"] == 1 and r["feature_type"] == "pinyin_standalone":
        m = re_sump.match(r'拼音 (.+) →', r["rule_text"])
        if m:
            key = m.group(1)
            on = r["onyomi"]
            if on not in t1_pinyin_map[key] or r["accuracy"] > t1_pinyin_map[key][on]:
                t1_pinyin_map[key][on] = r["accuracy"]

# Filter T3 tone rules subsumed by T1 no-tone rules
pruned = []
subsumed_count = 0
for r in all_rules:
    if r["tier"] == 3 and r["feature_type"] == "pinyin_pair" and r["rule_text"].startswith("拼音 "):
        m = re_sump.match(r'拼音 (.+) →', r["rule_text"])
        if m:
            key_with_tone = m.group(1)  # e.g. "d+ong1"
            key_no_tone = re_sump.sub(r'\d', '', key_with_tone)  # "d+ong"
            on = r["onyomi"]
            if key_no_tone in t1_pinyin_map and on in t1_pinyin_map[key_no_tone]:
                # T1 already covers this onyomi with same or better accuracy
                if t1_pinyin_map[key_no_tone][on] >= r["accuracy"] * 0.9:  # 10% tolerance
                    subsumed_count += 1
                    continue
    pruned.append(r)
print(f"  Subsumed T3 tone rules (covered by T1): {subsumed_count}")
all_rules = pruned

# ═══════════════════════════════════════════════════════════════
# DEDUP
# ═══════════════════════════════════════════════════════════════
seen = {}
for r in all_rules:
    key = (r["rule_text"], r["onyomi"])
    if key not in seen or r["accuracy"] > seen[key]["accuracy"]:
        seen[key] = r
all_rules = list(seen.values())
print(f"After dedup: {len(all_rules)}")

# Sort: by tier ascending, then by accuracy descending within tier
all_rules.sort(key=lambda r: (r["tier"], -r["accuracy"]))

# ═══════════════════════════════════════════════════════════════
# GREEDY SET COVER (for minimal subset)
# ═══════════════════════════════════════════════════════════════
print("\nRunning greedy set cover...")
uncovered_set = set(range(n_total))
selected = []
cumulative = []

# Work on a copy
remaining = list(all_rules)

while uncovered_set:
    best_r, best_new, best_idx, best_score = None, set(), -1, -1.0
    for i, r in enumerate(remaining):
        new = r["correct_indices"] & uncovered_set
        if len(new) == 0:
            continue
        score = len(new) * (r["accuracy"] ** 2)
        if score > best_score:
            best_new, best_r, best_idx, best_score = new, r, i, score
        elif abs(score - best_score) < 0.001 and best_r:
            if r["tier"] < best_r["tier"]:
                best_new, best_r, best_idx, best_score = new, r, i, score

    if len(best_new) == 0:
        break

    selected.append({**best_r, "new_covered": len(best_new),
                     "cumulative": n_total - len(uncovered_set) + len(best_new)})
    uncovered_set -= best_new
    cumulative.append(n_total - len(uncovered_set))
    del remaining[best_idx]

    if len(selected) % 50 == 0:
        print(f"  {len(selected)} rules → {cumulative[-1]}/{n_total} ({cumulative[-1]/n_total:.1%})")

# ═══════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"RESULTS: {len(selected)} rules cover {cumulative[-1]}/{n_total} ({cumulative[-1]/n_total:.1%})")
print(f"{'='*60}")

tier_names = {1: "★ 最易(单特征)", 2: "★★ 容易(声符/结构)", 3: "★★★ 中等(双特征)",
              4: "★★★★ 较难(挖掘双层)", 5: "★★★★★ 最难(三层+蒸馏)"}

# ── Stats by tier ──
tier_stats = defaultdict(lambda: {"count": 0, "total_new": 0, "accs": []})
for r in selected:
    ts = tier_stats[r["tier"]]
    ts["count"] += 1
    ts["total_new"] += r["new_covered"]
    ts["accs"].append(r["accuracy"])

print("\nTier breakdown (minimal set):")
for t in sorted(tier_stats.keys()):
    ts = tier_stats[t]
    print(f"  {tier_names[t]}: {ts['count']} rules, avg acc={np.mean(ts['accs']):.0%}, "
          f"new={ts['total_new']} ({ts['total_new']/n_total:.1%})")

# ── Stats by confidence (exhaustive set) ──
conf_stats = defaultdict(lambda: {"count": 0, "acc_min": 1.0, "acc_max": 0.0})
for r in all_rules:
    cs = conf_stats[r["confidence"]]
    cs["count"] += 1
    cs["acc_min"] = min(cs["acc_min"], r["accuracy"])
    cs["acc_max"] = max(cs["acc_max"], r["accuracy"])

print("\nConfidence distribution (exhaustive set):")
for conf in ["确定", "大概率", "有时", "偶尔"]:
    cs = conf_stats[conf]
    print(f"  {conf}: {cs['count']} rules, range {cs['acc_min']:.0%}-{cs['acc_max']:.0%}")

# ── Stats by feature type ──
ft_stats = Counter(r["feature_type"] for r in all_rules)
print("\nFeature type distribution:")
for ft, cnt in ft_stats.most_common():
    print(f"  {ft}: {cnt}")

# ── Coverage milestones ──
print("\nCoverage milestones:")
for m in [5, 10, 20, 30, 50, 100, 200]:
    if m <= len(cumulative):
        tier = selected[m-1]["tier"]
        print(f"  First {m:3d} rules: {cumulative[m-1]/n_total:.0%}  (last rule tier: {'★'*tier})")

# ── Top rules per tier ──
print("\nTop rules by tier:")
for t in sorted(tier_stats.keys()):
    ts = [r for r in selected if r["tier"] == t]
    if not ts: continue
    print(f"\n  {tier_names[t]} (top 5 by new_covered):")
    for i, r in enumerate(sorted(ts, key=lambda x: -x["new_covered"])[:5]):
        print(f"    {i+1}. [{r['confidence']}] {r['rule_text'][:60]} → {r['onyomi']} (+{r['new_covered']}字, {r['accuracy']:.0%})")

# ═══════════════════════════════════════════════════════════════
# OUTPUT — DUAL FORMAT
# ═══════════════════════════════════════════════════════════════
os.makedirs("output", exist_ok=True)

# ── Compute path coverage ──
t12 = [r for r in selected if r["tier"] <= 2]
t123 = [r for r in selected if r["tier"] <= 3]
cov12 = sum(r["new_covered"] for r in t12)
cov123 = sum(r["new_covered"] for r in t123)

# ── Build exhaustive rules list ──
exhaustive_rules = []
for r in all_rules:
    er = {
        "rule_text": r["rule_text"],
        "onyomi": r["onyomi"],
        "source": r["source"],
        "tier": r["tier"],
        "accuracy": round(r["accuracy"], 4),
        "matched": r["matched"],
        "correct": r["correct"],
        "confidence": r["confidence"],
        "feature_type": r["feature_type"],
        "reading_type": r.get("reading_type", "general"),
        "weak_flag": r["weak_flag"],
        "correct_examples": r["correct_examples"][:6],
        "exception_examples": r["exception_examples"][:3],
    }
    exhaustive_rules.append(er)

# ── Build minimal (selected) rules list ──
minimal_rules = []
for i, r in enumerate(selected):
    mr = {
        "rank": i + 1,
        "rule_text": r["rule_text"],
        "onyomi": r["onyomi"],
        "source": r["source"],
        "tier": r["tier"],
        "accuracy": round(r["accuracy"], 4),
        "new_covered": r["new_covered"],
        "cumulative": r["cumulative"],
        "confidence": r["confidence"],
        "feature_type": r["feature_type"],
        "reading_type": r.get("reading_type", "general"),
        "correct_examples": r["correct_examples"][:6],
        "exception_examples": r["exception_examples"][:3],
    }
    minimal_rules.append(mr)

# ── Exhaustive JSON ──
exhaustive_json = {
    "level": LEVEL,
    "generated": pd.Timestamp.now().isoformat(),
    "n_total": n_total,
    "n_rules_total": len(all_rules),
    "n_rules_selected": len(selected),
    "coverage_minimal": cumulative[-1] if cumulative else 0,
    "coverage_ratio": round(cumulative[-1]/n_total, 3) if cumulative else 0,
    "stats": {
        "by_confidence": {
            conf: {
                "count": conf_stats[conf]["count"],
                "min_acc": round(conf_stats[conf]["acc_min"], 3),
                "max_acc": round(conf_stats[conf]["acc_max"], 3),
            } for conf in ["确定", "大概率", "有时", "偶尔"]
        },
        "by_tier": {
            tier_names[t]: {
                "n_rules": tier_stats[t]["count"],
                "avg_accuracy": round(np.mean(tier_stats[t]["accs"]), 3) if tier_stats[t]["accs"] else 0,
                "total_new": tier_stats[t]["total_new"],
            } for t in sorted(tier_stats.keys())
        },
        "by_feature_type": dict(ft_stats.most_common()),
        "learning_paths": {
            "A_quick": {"tiers": "1-2", "n_rules": len(t12), "coverage": round(cov12/n_total, 3)},
            "B_deep": {"tiers": "1-3", "n_rules": len(t123), "coverage": round(cov123/n_total, 3)},
            "C_complete": {"tiers": "1-5", "n_rules": len(selected), "coverage": round(cumulative[-1]/n_total, 3) if cumulative else 0},
        },
    },
    "rules": exhaustive_rules,
}

with open(f"output/{FBASE}_tiered_rules_exhaustive.json", "w") as f:
    json.dump(exhaustive_json, f, indent=2, ensure_ascii=False)
print(f"\nSaved: output/{FBASE}_tiered_rules_exhaustive.json ({len(exhaustive_rules)} rules)")

# ── Minimal JSON ──
minimal_json = {
    "level": LEVEL,
    "generated": pd.Timestamp.now().isoformat(),
    "n_total": n_total,
    "n_rules": len(selected),
    "coverage": cumulative[-1] if cumulative else 0,
    "coverage_ratio": round(cumulative[-1]/n_total, 3) if cumulative else 0,
    "stats": exhaustive_json["stats"],
    "rules": minimal_rules,
}

with open(f"output/{FBASE}_tiered_rules.json", "w") as f:
    json.dump(minimal_json, f, indent=2, ensure_ascii=False)
print(f"Saved: output/{FBASE}_tiered_rules.json ({len(minimal_rules)} selected rules)")

# ── Markdown ──
out = []
out.append(f"# 分层规则集 — {LEVEL}")
out.append(f"掌握最少的东西，学习最多的单词")
out.append(f"\n- 全量规则: {len(all_rules)} 条（含所有置信度）")
out.append(f"- 贪心选中: {len(selected)} 条")
out.append(f"- 覆盖: {cumulative[-1]}/{n_total} = {cumulative[-1]/n_total:.1%}" if cumulative else "")
out.append(f"- 压缩比: {cumulative[-1]/len(selected):.1f}x\n" if cumulative else "")

out.append("## 置信度分布\n")
out.append("| 置信度 | 数量 | 精度范围 |")
out.append("|--------|------|---------|")
for conf in ["确定", "大概率", "有时", "偶尔"]:
    cs = conf_stats[conf]
    if cs["count"] > 0:
        out.append(f"| {conf} | {cs['count']} | {cs['acc_min']:.0%}-{cs['acc_max']:.0%} |")

out.append("\n## 学习路径\n")
out.append("### 路径A：快速入门（Tier 1-2，最易学）")
out.append(f"- {len(t12)} 条规则，覆盖 {cov12}/{n_total} = {cov12/n_total:.0%}")
out.append(f"- 单条件规则，利用已有的拼音知识或熟悉的声符\n")

out.append("### 路径B：深度学习（Tier 1-3，中等难度）")
out.append(f"- {len(t123)} 条规则，覆盖 {cov123}/{n_total} = {cov123/n_total:.0%}")
out.append(f"- 加入声调和双特征规则，覆盖范围更大\n")

out.append("### 路径C：追求极致（Tier 1-5，包含复杂规则）")
out.append(f"- {len(selected)} 条规则，覆盖 {cumulative[-1]}/{n_total} = {cumulative[-1]/n_total:.0%}" if cumulative else "")
out.append(f"- 包含XGBoost挖掘的复合规则\n")

# Coverage curve
out.append("## 覆盖曲线\n")
out.append("| 规则数 | 5 | 10 | 20 | 30 | 50 | 100 | 200 |")
out.append("|--------|----|----|----|----|----|-----|-----|")
row = []
for m in [5, 10, 20, 30, 50, 100, 200]:
    row.append(f"{cumulative[m-1]/n_total:.0%}" if m <= len(cumulative) else "-")
out.append(f"| 累计覆盖 | {' | '.join(row)} |")

# Full rule table
out.append(f"\n## 完整规则列表（按学习优先级排序）\n")
out.append("| # | 难度 | 置信 | 规则 | →音读 | 来源 | 类型 | 精度 | 新增 | 累计 |")
out.append("|---|------|------|------|------|------|------|------|------|------|")
for i, r in enumerate(selected[:300]):
    stars = "★" * r["tier"]
    rs = r["rule_text"].replace("|", "/")[:45]
    ft = r.get("feature_type", "")[:8]
    rt_label = {"go-on": "呉", "kan-on": "漢", "divergent": "異", "general": ""}.get(r.get("reading_type", ""), "")
    cp = r["cumulative"] / n_total
    out.append(f"| {i+1} | {stars} | {r['confidence']} | {rs} | **{r['onyomi']}** | {r['source'][:12]} | "
               f"{ft}{' '+rt_label if rt_label else ''} | {r['accuracy']:.0%} | +{r['new_covered']} | {r['cumulative']} |")

with open(f"output/{FBASE}_tiered_rules.md", "w") as f:
    f.write("\n".join(out))

print(f"Saved: output/{FBASE}_tiered_rules.md")
print("\nDone.")
