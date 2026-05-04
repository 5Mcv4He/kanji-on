"""Extract 音訓判別 rules: when is a kanji in a word read as on'yomi vs kun'yomi?
Outputs formal rules for word-first mode (N5-N4 beginner entry point).
"""
import pandas as pd
import numpy as np
import os
import re
from collections import defaultdict

os.makedirs("rules", exist_ok=True)
os.makedirs("output", exist_ok=True)

print("=" * 60)
print("音訓判別 RULE EXTRACTION")
print("=" * 60)

# Load word data with levels
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

# Filter to valid rows
w = w[w["level_num"].notna() & (w["word_text"] != "nan") & (w["word_text"] != "")].copy()
w["level_num"] = w["level_num"].astype(int)

# Load V2 for kanji reading info
v2 = pd.read_excel("漢字検索V2.xlsm", sheet_name="漢字一覧")
v2_lookup = {}
for _, r in v2.iterrows():
    k = str(r["漢字"]) if pd.notna(r["漢字"]) else ""
    if not k:
        continue
    on_raw = str(r["音"]) if pd.notna(r["音"]) else ""
    kun_raw = str(r["訓"]) if pd.notna(r["訓"]) else ""
    on_list = [x.strip() for x in on_raw.replace("、", ",").split(",") if x.strip() and x.strip() != "nan"]
    kun_list = [x.strip() for x in re.split(r"[、,]", kun_raw) if x.strip() and x.strip() != "nan"]
    v2_lookup[k] = {"on_list": on_list, "kun_list": kun_list}

# Load V1 fallback
v1 = pd.read_excel("KanjiKensaku.xls", sheet_name="漢字一覧")
for _, r in v1.iterrows():
    k = str(r["漢字"]) if pd.notna(r["漢字"]) else ""
    if not k or k in v2_lookup:
        continue
    on_raw = str(r["音読み"]) if pd.notna(r["音読み"]) else ""
    kun_raw = str(r["訓読み"]) if pd.notna(r["訓読み"]) else ""
    on_list = [x.strip() for x in on_raw.replace("、", ",").split(",") if x.strip() and x.strip() != "nan"]
    kun_list = [x.strip() for x in re.split(r"[、,]", kun_raw) if x.strip() and x.strip() != "nan"]
    v2_lookup[k] = {"on_list": on_list, "kun_list": kun_list}

def k2h(s):
    """Katakana to hiragana."""
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)

# Analyze each word
print("\nAnalyzing words...")

results = []
for _, row in w.iterrows():
    lv = row["level_num"]
    word = row["word_text"]
    kana = str(row["kana"])
    kanji_chars = [c for c in word if "一" <= c <= "鿿"
                   or "㐀" <= c <= "䶿" or "豈" <= c <= "﫿"]

    if len(kanji_chars) == 0:
        # Pure kana word
        results.append({"word": word, "kana": kana, "level": lv,
                       "kanji_count": 0, "rule": "纯仮名", "kanji": ""})
        continue

    if len(kanji_chars) == 1:
        ch = kanji_chars[0]
        info = v2_lookup.get(ch, {"on_list": [], "kun_list": []})

        # Check: does the word have okurigana? (kanji + trailing kana)
        okurigana = word[len(ch):] if word.startswith(ch) else word[:word.index(ch)]
        full_reading = kana

        # Determine reading type by aligning kana with on/kun lists
        is_on = False
        is_kun = False
        matched_reading = ""

        for on_r in info["on_list"]:
            on_h = k2h(on_r)
            if on_h in kana:
                is_on = True
                matched_reading = on_r
                break

        if not is_on:
            for kun_r in info["kun_list"]:
                stem = kun_r.split("・")[0] if "・" in kun_r else kun_r
                if stem in kana:
                    is_kun = True
                    matched_reading = kun_r
                    break

        # Heuristic: kanji followed by kana (okurigana) → kun
        has_okurigana = len(word) > 1 and (word.endswith(kana[-1]) if kana else False)

        if is_kun:
            rule = "訓讀"
        elif is_on:
            rule = "音讀"
        elif has_okurigana:
            rule = "送仮名→訓讀"
        else:
            rule = "不明"

        results.append({
            "word": word, "kana": kana, "level": lv,
            "kanji_count": 1, "kanji": ch,
            "rule": rule, "matched_reading": matched_reading,
            "has_okurigana": has_okurigana,
        })

    elif len(kanji_chars) >= 2:
        # Multi-kanji word: judge each kanji
        # Two-kanji compound → typically on'yomi
        # But could be jukujikun (special kun reading for whole compound)

        is_jukujikun = True  # Assume jukujikun unless on'yomi matches

        # Try to find on'yomi matches for all kanji
        all_on_matched = True
        for ch in kanji_chars:
            info = v2_lookup.get(ch, {"on_list": [], "kun_list": []})
            matched = False
            for on_r in info["on_list"]:
                on_h = k2h(on_r)
                if on_h in kana:
                    matched = True
                    break
            if not matched:
                all_on_matched = False
                break

        if all_on_matched and len(kanji_chars) >= 2:
            rule = "音讀（連続漢字）"
        elif len(kanji_chars) >= 2:
            rule = "熟字訓/混合"
        else:
            rule = "不明"

        results.append({
            "word": word, "kana": kana, "level": lv,
            "kanji_count": len(kanji_chars), "kanji": "".join(kanji_chars),
            "rule": rule, "all_on_match": all_on_matched,
        })

df_r = pd.DataFrame(results)
print(f"Total words analyzed: {len(df_r)}")

# Accuracy by rule type
print("\n" + "=" * 60)
print("RULE ACCURACY SUMMARY")
print("=" * 60)

# Rule 1: 送仮名 → 訓讀 (okurigana → kun'yomi)
# Check: single kanji words with okurigana, are they actually kun?
okurigana_mask = (df_r["kanji_count"] == 1) & (df_r["has_okurigana"] == True)
okurigana_words = df_r[okurigana_mask]
okurigana_correct = sum(okurigana_words["rule"] == "訓讀")
print(f"Rule「漢字+送仮名 → 訓讀」: {okurigana_correct}/{len(okurigana_words)} = {okurigana_correct/max(len(okurigana_words),1):.1%}")

# Rule 2: 連続漢字 → 音讀
jukugo_mask = (df_r["kanji_count"] >= 2)
jukugo_words = df_r[jukugo_mask]
jukugo_on = sum(jukugo_words["rule"] == "音讀（連続漢字）")
print(f"Rule「兩漢字以上連続 → 音讀」: {jukugo_on}/{len(jukugo_words)} = {jukugo_on/max(len(jukugo_words),1):.1%}")

# By level
print("\nBy JLPT level:")
for lv in [5, 4, 3, 2, 1]:
    lv_words = df_r[df_r["level"] == lv]
    lv_oku = lv_words[(lv_words["kanji_count"] == 1) & (lv_words["has_okurigana"] == True)]
    lv_jukugo = lv_words[lv_words["kanji_count"] >= 2]

    oku_acc = sum(lv_oku["rule"] == "訓讀") / max(len(lv_oku), 1)
    jukugo_acc = sum(lv_jukugo["rule"] == "音讀（連続漢字）") / max(len(lv_jukugo), 1)
    print(f"  N{lv}: 送仮名→訓讀 {oku_acc:.1%} ({len(lv_oku)}詞) | 連続漢字→音讀 {jukugo_acc:.1%} ({len(lv_jukugo)}詞) | 総詞数 {len(lv_words)}")

# Save
df_r.to_csv("output/onkun_analysis.csv", index=False, encoding="utf-8")
print(f"\nSaved: output/onkun_analysis.csv ({len(df_r)} rows)")

# Generate formal rules
print("\n" + "=" * 60)
print("FORMAL RULES")
print("=" * 60)

rules = [
    {
        "rule_id": "OKURIGANA-001",
        "rule_type": "音訓判別",
        "rule_name": "送仮名則",
        "rule_desc": "漢字の後に仮名が続く（送仮名）場合、その漢字は訓讀",
        "accuracy": f"{okurigana_correct/max(len(okurigana_words),1):.1%}",
        "applies_to": f"{len(okurigana_words)} words",
        "mechanism": "送仮名は日本語固有の語幹変化を示す。音讀は変化しないため送仮名を伴わない。",
        "examples": "食べる(たべる)·見る(みる)·大きい(おおきい)·話す(はなす)",
        "exceptions": "音讀+する動詞（勉強する→音讀）·当て字（美味しい→おいしい）",
    },
    {
        "rule_id": "JUKUGO-001",
        "rule_type": "音訓判別",
        "rule_name": "連続漢字則",
        "rule_desc": "漢字が2字以上連続する熟語は音讀",
        "accuracy": f"{jukugo_on/max(len(jukugo_words),1):.1%}",
        "applies_to": f"{len(jukugo_words)} words",
        "mechanism": "漢語は中国語由来のため漢字音で読む。2字以上の漢字連続は高い確率で漢語。",
        "examples": "学生(がくせい)·電車(でんしゃ)·日本語(にほんご)",
        "exceptions": "熟字訓（今日→きょう·明日→あした）·湯桶読み（夕刊→ゆうかん：訓+音）·重箱読み（場所→ばしょ：音+訓）",
    },
    {
        "rule_id": "TANKANJI-001",
        "rule_type": "音訓判別",
        "rule_name": "単漢字優先則",
        "rule_desc": "単独漢字は訓讀を優先確認、該当なければ音讀",
        "accuracy": None,
        "applies_to": None,
        "mechanism": "日本語の基本的な語彙は訓讀（和語）。一つの漢字が単独で現れる場合、まず訓讀を疑う。",
        "examples": "山(やま)·川(かわ)·心(こころ)·時(とき)",
        "exceptions": "数詞·単位（一→いち·円→えん）·抽象概念（愛→あい·禅→ぜん）",
    },
]

rules_df = pd.DataFrame(rules)
rules_df.to_csv("rules/onkun_rules.csv", index=False, encoding="utf-8")
print(f"Saved: rules/onkun_rules.csv ({len(rules_df)} rules)")
print("\nDone.")
