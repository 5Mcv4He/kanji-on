"""Phase 1: Build level-stratified kanji datasets (N5–N1++).
Reads word.xlsx for level→word→reading, V2 for structural features, pypinyin for pinyin.
Outputs dataset/n5_dataset.csv through dataset/n1pp_dataset.csv.
"""
import pandas as pd
import numpy as np
import json
import re
import os
from collections import defaultdict
from pypinyin import pinyin, Style

# ── Config ──────────────────────────────────────────────────────────
LEVELS = ["N5", "N4", "N3", "N2", "N1", "N1+", "N1++"]
OUT_DIR = "dataset"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Step 1: Extract word levels from word.xlsx MP3 paths ────────────
print("=" * 60)
print("Step 1: Extracting word levels from word.xlsx")
w = pd.read_excel("word.xlsx")

def extract_level_from_row(row):
    for col in w.columns:
        text = str(row[col]) if pd.notna(row[col]) else ""
        m = re.search(r"n([1-5])", text, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None

w["level_num"] = w.apply(extract_level_from_row, axis=1)
w["kana"] = w["假名"].apply(lambda x: str(x) if pd.notna(x) else "")
w["word_text"] = w["汉字/外文"].apply(lambda x: str(x) if pd.notna(x) else "")

# Filter to rows with actual words and levels
w_valid = w[w["level_num"].notna() & (w["word_text"] != "nan") & (w["word_text"] != "")].copy()
print(f"  Valid word rows: {len(w_valid)}")

# Count per level
for lv in range(1, 6):
    cnt = len(w_valid[w_valid["level_num"].astype(int) == lv])
    print(f"  N{lv}: {cnt} words")

# ── Step 2: Aggregate kanji → readings per level ────────────────────
print("\n" + "=" * 60)
print("Step 2: Aggregating kanji readings per level")

# For each level, collect: kanji → set of (reading, word, word_kana)
level_kanji_data = {f"N{lv}": defaultdict(lambda: {"words": [], "readings": set()})
                     for lv in range(1, 6)}

for _, row in w_valid.iterrows():
    lv = int(row["level_num"])
    level_key = f"N{lv}"
    word_text = row["word_text"]
    kana = row["kana"]
    kanji_chars = [c for c in word_text if "一" <= c <= "鿿"
                   or "㐀" <= c <= "䶿" or "豈" <= c <= "﫿"]

    for ch in kanji_chars:
        level_kanji_data[level_key][ch]["words"].append((word_text, kana))
        level_kanji_data[level_key][ch]["readings"].add(kana)

# Build cumulative sets: N5 = only word level 5 (basic first),
# N4 = word levels 5+4, ..., N1 = word levels 5+4+3+2+1 (all words).
# word.xlsx: level 5=N5 (basic, ~1000 words), level 1=N1 (advanced, ~3052 words).
spec_levels = [(5, "N5"), (4, "N4"), (3, "N3"), (2, "N2"), (1, "N1")]
cumulative = {}
for i, (lv, spec_key) in enumerate(spec_levels):
    cum_data = defaultdict(lambda: {"words": [], "readings": set()})
    # Accumulate from basic (5) upward through current level
    for j in range(i + 1):
        wl = spec_levels[j][0]  # word level number
        lkey = f"N{wl}"
        for k, v in level_kanji_data[lkey].items():
            cum_data[k]["words"].extend(v["words"])
            cum_data[k]["readings"] |= v["readings"]
    cumulative[spec_key] = cum_data
    print(f"  {spec_key} cumulative: {len(cum_data)} kanji (word levels 5..{lv})")

# ── Step 3: Load V2 structural data ─────────────────────────────────
print("\n" + "=" * 60)
print("Step 3: Loading V2 structural data")
v2 = pd.read_excel("漢字検索V2.xlsm", sheet_name="漢字一覧")
print(f"  V2 rows: {len(v2)}")

v2_lookup = {}
for _, r in v2.iterrows():
    k = str(r["漢字"]) if pd.notna(r["漢字"]) else ""
    if not k:
        continue
    on_raw = str(r["音"]) if pd.notna(r["音"]) else ""
    kun_raw = str(r["訓"]) if pd.notna(r["訓"]) else ""
    meaning_raw = str(r["意味"]) if pd.notna(r["意味"]) else ""
    comp_raw = str(r["構成文字"]) if pd.notna(r["構成文字"]) else ""
    radical = str(r["部首"]) if pd.notna(r["部首"]) else ""
    non_rad = str(r["非部首部"]) if pd.notna(r["非部首部"]) else ""
    rad_cd = str(r["部首CD"]) if pd.notna(r["部首CD"]) else ""
    rad_strokes = int(r["部首画数"]) if pd.notna(r["部首画数"]) else 0
    non_rad_strokes = int(r["非部首部画数"]) if pd.notna(r["非部首部画数"]) else 0
    total_strokes = int(r["総画数"]) if pd.notna(r["総画数"]) else 0

    # Parse on readings
    on_list = []
    if on_raw and on_raw != "nan":
        for part in on_raw.replace("、", ",").split(","):
            part = part.strip()
            if part and part != "nan":
                on_list.append(part)

    # Parse kun readings
    kun_list = []
    if kun_raw and kun_raw != "nan":
        for part in re.split(r"[、,]", kun_raw):
            part = part.strip()
            if part and part != "nan":
                kun_list.append(part)

    v2_lookup[k] = {
        "on_list": on_list,
        "kun_list": kun_list,
        "meaning": meaning_raw if meaning_raw != "nan" else "",
        "components": comp_raw,
        "radical": radical,
        "non_radical_part": non_rad,
        "radical_cd": rad_cd,
        "radical_strokes": rad_strokes,
        "non_radical_strokes": non_rad_strokes,
        "total_strokes": total_strokes,
    }

print(f"  V2 lookup: {len(v2_lookup)} kanji")

# ── Step 4: Load V1 for kanji not in V2 ─────────────────────────────
print("\n" + "=" * 60)
print("Step 4: Loading V1 fallback data")
v1 = pd.read_excel("KanjiKensaku.xls", sheet_name="漢字一覧")
v1_lookup = {}
for _, r in v1.iterrows():
    k = str(r["漢字"]) if pd.notna(r["漢字"]) else ""
    if not k:
        continue
    on_raw = str(r["音読み"]) if pd.notna(r["音読み"]) else ""
    kun_raw = str(r["訓読み"]) if pd.notna(r["訓読み"]) else ""
    meaning_raw = str(r["意味"]) if pd.notna(r["意味"]) else ""
    comp_raw = str(r["構成文字"]) if pd.notna(r["構成文字"]) else ""
    radical = str(r["部首"]) if pd.notna(r["部首"]) else ""
    strokes = int(r["画数"]) if pd.notna(r["画数"]) else 0

    on_list = []
    if on_raw and on_raw != "nan":
        for part in on_raw.replace("、", ",").replace("・", ",").split(","):
            part = part.strip()
            if part and part != "nan":
                on_list.append(part)

    kun_list = []
    if kun_raw and kun_raw != "nan":
        for part in re.split(r"[、,]", kun_raw):
            part = part.strip()
            if part and part != "nan":
                kun_list.append(part)

    v1_lookup[k] = {
        "on_list": on_list,
        "kun_list": kun_list,
        "meaning": meaning_raw if meaning_raw != "nan" else "",
        "components": comp_raw,
        "radical": radical,
        "non_radical_part": "",  # V1 doesn't have this
        "radical_cd": "",
        "radical_strokes": 0,
        "non_radical_strokes": 0,
        "total_strokes": strokes,
    }

print(f"  V1 lookup: {len(v1_lookup)} kanji")

# ── Step 5: Add pinyin ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 5: Adding pinyin features")

# Initial consonant categories (声母)
INITIAL_CATEGORIES = {
    "b": "b", "p": "p", "m": "m", "f": "f",
    "d": "d", "t": "t", "n": "n", "l": "l",
    "g": "g", "k": "k", "h": "h",
    "j": "j", "q": "q", "x": "x",
    "zh": "zh", "ch": "ch", "sh": "sh", "r": "r",
    "z": "z", "c": "c", "s": "s",
    "y": "y", "w": "w",
}

def get_initial(py_str):
    """Extract initial consonant from pinyin string."""
    if not py_str:
        return ""
    py = py_str.lower().strip()
    # Check multi-char initials first
    for ic in ["zh", "ch", "sh"]:
        if py.startswith(ic):
            return ic
    if py and py[0] in "bcdfghjklmnpqrstwxyz":
        return py[0]
    return ""  # zero initial (e.g., 安→an)

def get_final(py_str):
    """Extract final (韵母) from pinyin string."""
    if not py_str:
        return ""
    py = py_str.lower().strip()
    init = get_initial(py)
    if init:
        return py[len(init):]
    return py

def get_tone(py_str):
    """Extract tone number from pinyin string."""
    if not py_str:
        return 0
    # pypinyin style.TONE3 gives e.g. 'an1', 'zhong1'
    m = re.search(r"(\d)$", py_str)
    if m:
        return int(m.group(1))
    return 0

def get_nasal_coda(final):
    """Determine nasal coda type."""
    if not final:
        return "none"
    if final.endswith("n") and not final.endswith("ng"):
        return "n"
    if final.endswith("ng"):
        return "ng"
    return "none"


# Entering tone lookup: pinyin finals with ≥30% entering tone (-k/-t) rate
# Built from on'yomi analysis of full kanji dataset
ENTERING_TONE_FINALS = {
    # -k finals
    "e4": "-k", "e2": "-k", "e1": "-k", "e": "-k",
    "o4": "-k", "o1": "-k",
    "uo4": "-k", "uo2": "-k", "uo1": "-k",
    "ue4": "-k", "ue1": "-k",
    "ve4": "-k",
    "u4": "-k", "u2": "-k", "u1": "-k",
    "i1": "-k", "i4": "-k",
    "uai1": "-k",
    # -t finals
    "ue2": "-t", "ue3": "-t",
    "v4": "-t",
    "ie4": "-t", "ie2": "-t",
    "a2": "-t", "a1": "-t",
    "o3": "-t",
    "ua2": "-t",
}


def get_entering_tone(final):
    """Determine if a pinyin final historically corresponds to entering tone (入声).
    Returns (is_entering: int, coda_type: str).
    Entering tone characters had -p/-t/-k codas in Middle Chinese that disappeared
    in Mandarin but survive in Japanese on'yomi as ク/キ/ツ/チ/フ/ウ."""
    if not final:
        return 0, "none"
    if final in ENTERING_TONE_FINALS:
        return 1, ENTERING_TONE_FINALS[final]
    return 0, "none"

# Build pinyin dict for all kanji we'll need
all_kanji_needed = set()
for lv in range(1, 6):
    all_kanji_needed |= set(cumulative[f"N{lv}"].keys())
# Also add all V1 kanji for N1+
v1_set = set(v1_lookup.keys())
all_kanji_needed |= v1_set
# And all V2 kanji for N1++
all_kanji_needed |= set(v2_lookup.keys())

print(f"  Total unique kanji to pinyin: {len(all_kanji_needed)}")

pinyin_data = {}
no_pinyin = []
for ch in all_kanji_needed:
    try:
        py_result = pinyin(ch, style=Style.TONE3, heteronym=False)
        if py_result and py_result[0]:
            py_str = py_result[0][0]
            initial = get_initial(py_str)
            final = get_final(py_str)
            tone = get_tone(py_str)
            nasal = get_nasal_coda(final)
            py_clean = re.sub(r"\d", "", py_str)  # tone-less version
            is_ent, ent_coda = get_entering_tone(final)
            pinyin_data[ch] = {
                "pinyin": py_clean,
                "pinyin_tone": py_str,
                "pinyin_initial": initial,
                "pinyin_final": final,
                "pinyin_tone_num": tone,
                "nasal_coda": nasal,
                "is_entering_tone": is_ent,
                "entering_coda": ent_coda,
            }
        else:
            no_pinyin.append(ch)
            pinyin_data[ch] = {
                "pinyin": "", "pinyin_tone": "", "pinyin_initial": "",
                "pinyin_final": "", "pinyin_tone_num": 0, "nasal_coda": "none",
                "is_entering_tone": 0, "entering_coda": "none",
            }
    except Exception:
        no_pinyin.append(ch)
        pinyin_data[ch] = {
            "pinyin": "", "pinyin_tone": "", "pinyin_initial": "",
            "pinyin_final": "", "pinyin_tone_num": 0, "nasal_coda": "none",
            "is_entering_tone": 0, "entering_coda": "none",
        }

print(f"  No pinyin: {len(no_pinyin)}")
if no_pinyin:
    print(f"  Samples: {no_pinyin[:10]}")

# ── Step 6: Build level datasets ────────────────────────────────────
print("\n" + "=" * 60)
print("Step 6: Building level datasets")

def get_kanji_info(k):
    """Get structural info for a kanji, V2 first, V1 fallback."""
    if k in v2_lookup:
        return v2_lookup[k]
    if k in v1_lookup:
        return v1_lookup[k]
    return {
        "on_list": [], "kun_list": [], "meaning": "",
        "components": k, "radical": "", "non_radical_part": "",
        "radical_cd": "", "radical_strokes": 0,
        "non_radical_strokes": 0, "total_strokes": 0,
    }

def build_level_dataset(level_key, kanji_dict):
    """Build a DataFrame for a given level."""
    rows = []
    for kanji, data in sorted(kanji_dict.items()):
        info = get_kanji_info(kanji)
        py = pinyin_data.get(kanji, {})

        # All readings observed in words at this level
        observed_kana = sorted(data["readings"])

        # V2 readings that actually appear in observed kana (filtered)
        on_in_use = []
        kun_in_use = []

        def k2h(s):
            """Katakana to hiragana."""
            return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)

        for on_r in info["on_list"]:
            on_h = k2h(on_r)
            for obs in observed_kana:
                if on_r in obs or on_h in obs:
                    on_in_use.append(on_r)
                    break
        for kun_r in info["kun_list"]:
            stem = kun_r.split("・")[0] if "・" in kun_r else kun_r
            for obs in observed_kana:
                if stem in obs or obs.startswith(stem) or stem.startswith(obs):
                    kun_in_use.append(kun_r)
                    break

        # Primary on'yomi (most reliable single target for classification)
        primary_on = info["on_list"][0] if info["on_list"] else ""

        # Sample words
        sample_words = data["words"][:10]

        rows.append({
            "kanji": kanji,
            "onyomi": primary_on,
            "onyomi_all": "|".join(info["on_list"]),
            "kunyomi_all": "|".join(info["kun_list"]),
            "onyomi_in_use": "|".join(on_in_use),
            "kunyomi_in_use": "|".join(kun_in_use),
            "observed_readings": "|".join(observed_kana),
            "meaning": info["meaning"],
            "components": info["components"],
            "radical": info["radical"],
            "non_radical_part": info["non_radical_part"],
            "radical_cd": info["radical_cd"],
            "radical_strokes": info["radical_strokes"],
            "non_radical_strokes": info["non_radical_strokes"],
            "total_strokes": info["total_strokes"],
            "pinyin": py.get("pinyin", ""),
            "pinyin_tone": py.get("pinyin_tone", ""),
            "pinyin_initial": py.get("pinyin_initial", ""),
            "pinyin_final": py.get("pinyin_final", ""),
            "pinyin_tone_num": py.get("pinyin_tone_num", 0),
            "nasal_coda": py.get("nasal_coda", "none"),
            "is_entering_tone": py.get("is_entering_tone", 0),
            "entering_coda": py.get("entering_coda", "none"),
            "word_count": len(data["words"]),
            "sample_words": json.dumps(
                [{"word": w, "kana": r} for w, r in sample_words],
                ensure_ascii=False,
            ),
        })
    return pd.DataFrame(rows)

# Build N5–N1 cumulative datasets
datasets = {}
for key in ["N5", "N4", "N3", "N2", "N1"]:
    df = build_level_dataset(key, cumulative[key])
    datasets[key] = df
    print(f"  {key}: {len(df)} kanji, {df['word_count'].sum()} word occurrences")

# ── Step 7: Build N1+ (all V1 kanji) ────────────────────────────────
print("\n" + "=" * 60)
print("Step 7: Building N1+ (V1 all kanji)")

# N1+ uses all V1 kanji with cumulative N5-N1 word data
n1_cumulative = cumulative["N1"]
n1plus_data = {}
for k in v1_set:
    if k in n1_cumulative:
        n1plus_data[k] = n1_cumulative[k]
    else:
        n1plus_data[k] = {"words": [], "readings": set()}

df_n1plus = build_level_dataset("N1+", n1plus_data)
datasets["N1+"] = df_n1plus
print(f"  N1+: {len(df_n1plus)} kanji")

# ── Step 8: Build N1++ (all V2 kanji) ───────────────────────────────
print("\n" + "=" * 60)
print("Step 8: Building N1++ (V2 all kanji)")

n1pp_data = {}
for k in set(v2_lookup.keys()):
    if k in n1_cumulative:
        n1pp_data[k] = n1_cumulative[k]
    else:
        n1pp_data[k] = {"words": [], "readings": set()}

df_n1pp = build_level_dataset("N1++", n1pp_data)
datasets["N1++"] = df_n1pp
print(f"  N1++: {len(df_n1pp)} kanji")

# ── Step 9: Save all datasets ───────────────────────────────────────
print("\n" + "=" * 60)
print("Step 9: Saving datasets")

for key, df in datasets.items():
    fname = f"{key.lower().replace('+', 'p')}_dataset.csv"
    path = os.path.join(OUT_DIR, fname)
    df.to_csv(path, index=False, encoding="utf-8")
    # Also count kanji with observed readings (in-use kanji)
    has_readings = (df["observed_readings"] != "").sum()
    print(f"  {path} — {len(df)} kanji ({has_readings} with observed readings)")

# ── Step 10: Summary report ─────────────────────────────────────────
print("\n" + "=" * 60)
print("DATASET BUILD COMPLETE")
print("=" * 60)
for lv in ["N5", "N4", "N3", "N2", "N1", "N1+", "N1++"]:
    df = datasets[lv]
    has_on = (df["onyomi"] != "").sum()
    has_py = (df["pinyin"] != "").sum()
    has_nonrad = (df["non_radical_part"] != "").sum()
    has_obs = (df["observed_readings"] != "").sum()
    print(f"  {lv}: {len(df)}字 | 有音读:{has_on} | 有拼音:{has_py} | 有非部首部:{has_nonrad} | 有实测读法:{has_obs}")
