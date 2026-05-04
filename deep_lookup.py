"""Deep kanji analysis: explain every reading with rules, usage, and siblings.

Usage:
    python3 deep_lookup.py <漢字>           # human-readable text
    python3 deep_lookup.py <漢字> --json    # structured JSON for API

Answers for each kanji:
  音読: Why this reading? What rule? Or memorization-only?
  訓読: Why this reading? What other kanji share it? Semantic connection?
        When to use which reading (word examples)?
"""
import pandas as pd
import numpy as np
import sys
import json
import os
import re
from collections import defaultdict, Counter


def _split_on_readings(raw):
    if not raw or str(raw) == "nan":
        return []
    return [x.strip() for x in re.split(r"[|／\s]+", str(raw))
            if x.strip() and x.strip() != "?"]


def _split_kun_readings(raw):
    """Kun readings: |-separated groups, each may have · marking okurigana.
    e.g., 'い・きる|う・む' → ['いきる', 'うむ']
    """
    if not raw or str(raw) == "nan":
        return []
    result = []
    for group in re.split(r"[|／\s]+", str(raw)):
        group = group.strip()
        if not group or group == "?":
            continue
        full = group.replace("・", "").replace("·", "")
        if full:
            result.append(full)
    return result


def _katakana_to_hiragana(text):
    """Convert katakana to hiragana for matching word kana."""
    return "".join(
        chr(ord(ch) - 0x60) if 0x30A0 < ord(ch) < 0x30F7 else ch
        for ch in text
    )


def _find_rules_for_reading(on, ldf, inf, ift, final, final_nt, nonrad, nasal,
                           is_entering_tone, entering_coda, kanji,
                           mc_initial=None, mc_rhyme=None, mc_grade=None,
                           mc_openness=None, mc_tone=None, mc_voicing=None,
                           mc_rhyme_cat=None, mc_initial_cat=None,
                           mc_entering_coda_type=None, radical=None,
                           strokes=None):
    """Find all rules that predict this onyomi for this kanji."""
    rules = []

    if inf and inf != "+":
        grp = ldf[ldf["_inf"] == inf]
        if len(grp) >= 2:
            c = Counter(grp["onyomi"]); t = sum(c.values())
            if on in c and c[on] / t >= 0.5:
                ex = [k for k in grp[grp["onyomi"] == on]["kanji"].values if k != kanji]
                rules.append({"type": "拼音·声+韵", "description": f"拼音 {inf} → {on}",
                              "accuracy": c[on] / t, "examples": ex})

    if ift and ift != "+":
        grp = ldf[ldf["_ift"] == ift]
        if len(grp) >= 2:
            c = Counter(grp["onyomi"]); t = sum(c.values())
            if on in c and c[on] / t >= 0.5:
                ex = [k for k in grp[grp["onyomi"] == on]["kanji"].values if k != kanji]
                rules.append({"type": "拼音·声+韵+调", "description": f"拼音 {ift} → {on}",
                              "accuracy": c[on] / t, "examples": ex})

    if final:
        grp = ldf[ldf["_fin"] == final]
        if len(grp) >= 2:
            c = Counter(grp["onyomi"]); t = sum(c.values())
            if on in c and c[on] / t >= 0.5:
                ex = [k for k in grp[grp["onyomi"] == on]["kanji"].values if k != kanji]
                rules.append({"type": "拼音·韵母+调", "description": f"韵母 -{final} → {on}",
                              "accuracy": c[on] / t, "examples": ex})

    if final_nt:
        grp = ldf[ldf["_fin_nt"] == final_nt]
        if len(grp) >= 2:
            c = Counter(grp["onyomi"]); t = sum(c.values())
            if on in c and c[on] / t >= 0.5:
                ex = [k for k in grp[grp["onyomi"] == on]["kanji"].values if k != kanji]
                rules.append({"type": "拼音·韵母(无调)", "description": f"韵母 -{final_nt} → {on}",
                              "accuracy": c[on] / t, "examples": ex})

    if nonrad and nonrad != "nan" and nonrad != "":
        grp = ldf[ldf["_nrp"] == nonrad]
        if len(grp) >= 2:
            c = Counter(grp["onyomi"]); t = sum(c.values())
            if on in c and c[on] / t >= 0.5:
                ex = [k for k in grp[grp["onyomi"] == on]["kanji"].values if k != kanji]
                rules.append({"type": "声符·精确", "description": f"声符「{nonrad}」→ {on}",
                              "accuracy": c[on] / t, "examples": ex})

    if nasal and nasal != "none":
        grp = ldf[ldf["_nas"] == nasal]
        if len(grp) >= 3:
            grp_on = grp[grp["onyomi"] == on]
            desc_map = {"-ng": "鼻音韵尾-ng → 長音系 (オウ/ユウ/エイ等)",
                        "-n": "鼻音韵尾-n → 撥音系 (〜ン)"}
            desc = desc_map.get(nasal, f"鼻音韵尾{nasal}")
            rules.append({"type": "鼻音·韻尾",
                          "description": desc,
                          "accuracy": len(grp_on) / len(grp) if len(grp) > 0 else 0,
                          "examples": []})

    if is_entering_tone == 1 and entering_coda and entering_coda != "none":
        grp = ldf[ldf["_ent"] == entering_coda]
        if len(grp) >= 3:
            rules.append({"type": "入声·韵尾",
                          "description": f"入声{entering_coda} → 短促音节",
                          "accuracy": 1.0, "examples": []})

    # ── MC (中古音) rules ──
    if mc_initial:
        grp = ldf[ldf["_mc_init"] == mc_initial]
        if len(grp) >= 3:
            c = Counter(grp["onyomi"]); t = sum(c.values())
            if on in c and c[on] / t >= 0.5:
                ex = [k for k in grp[grp["onyomi"] == on]["kanji"].values if k != kanji]
                rules.append({"type": "中古音·声母", "description": f"MC声母「{mc_initial}」→ {on}",
                              "accuracy": c[on] / t, "examples": ex})

    if mc_rhyme:
        grp = ldf[ldf["_mc_rhyme"] == mc_rhyme]
        if len(grp) >= 3:
            c = Counter(grp["onyomi"]); t = sum(c.values())
            if on in c and c[on] / t >= 0.5:
                ex = [k for k in grp[grp["onyomi"] == on]["kanji"].values if k != kanji]
                rules.append({"type": "中古音·韵母", "description": f"MC韵母「{mc_rhyme}」→ {on}",
                              "accuracy": c[on] / t, "examples": ex})

    if mc_tone:
        grp = ldf[ldf["_mc_tone"] == mc_tone]
        if len(grp) >= 3:
            c = Counter(grp["onyomi"]); t = sum(c.values())
            if on in c and c[on] / t >= 0.5:
                rules.append({"type": "中古音·声调",
                              "description": f"MC声调「{mc_tone}」→ {on}",
                              "accuracy": c[on] / t, "examples": []})

    if mc_grade:
        grp = ldf[ldf["_mc_grade"] == mc_grade]
        if len(grp) >= 3:
            c = Counter(grp["onyomi"]); t = sum(c.values())
            if on in c and c[on] / t >= 0.5:
                rules.append({"type": "中古音·等",
                              "description": f"MC{mc_grade}等 → {on}",
                              "accuracy": c[on] / t, "examples": []})

    if mc_voicing:
        grp = ldf[ldf["_mc_voice"] == mc_voicing]
        if len(grp) >= 3:
            c = Counter(grp["onyomi"]); t = sum(c.values())
            if on in c and c[on] / t >= 0.5:
                rules.append({"type": "中古音·清濁",
                              "description": f"MC清濁「{mc_voicing}」→ {on}",
                              "accuracy": c[on] / t, "examples": []})

    if mc_openness:
        grp = ldf[ldf["_mc_open"] == mc_openness]
        if len(grp) >= 3:
            label = "開口" if mc_openness == "開" else "合口"
            rules.append({"type": "中古音·開合",
                          "description": f"MC{label}",
                          "accuracy": 1.0, "examples": []})

    if mc_rhyme_cat:
        grp = ldf[ldf["_mc_rcat"] == mc_rhyme_cat]
        if len(grp) >= 3:
            c = Counter(grp["onyomi"]); t = sum(c.values())
            if on in c and c[on] / t >= 0.5:
                ex = [k for k in grp[grp["onyomi"] == on]["kanji"].values if k != kanji]
                rules.append({"type": "中古音·韵摄", "description": f"MC{mc_rhyme_cat}攝 → {on}",
                              "accuracy": c[on] / t, "examples": ex})

    if mc_initial_cat:
        grp = ldf[ldf["_mc_init_cat"] == mc_initial_cat]
        if len(grp) >= 3:
            c = Counter(grp["onyomi"]); t = sum(c.values())
            if on in c and c[on] / t >= 0.5:
                ex = [k for k in grp[grp["onyomi"] == on]["kanji"].values if k != kanji]
                rules.append({"type": "中古音·五音", "description": f"MC{mc_initial_cat}音 → {on}",
                              "accuracy": c[on] / t, "examples": ex})

    if mc_entering_coda_type:
        grp = ldf[ldf["_mc_ent_coda"] == mc_entering_coda_type]
        if len(grp) >= 3:
            c = Counter(grp["onyomi"]); t = sum(c.values())
            if on in c and c[on] / t >= 0.5:
                ex = [k for k in grp[grp["onyomi"] == on]["kanji"].values if k != kanji]
                coda_label = {"ク": "-k (ク)", "ツ": "-t (ツ)", "フ": "-p (フ)"}.get(mc_entering_coda_type, mc_entering_coda_type)
                rules.append({"type": "中古音·入声韵尾", "description": f"MC入声韵尾{coda_label} → {on}",
                              "accuracy": c[on] / t, "examples": ex})

    if radical:
        grp = ldf[ldf["_rad"] == radical]
        if len(grp) >= 3:
            c = Counter(grp["onyomi"]); t = sum(c.values())
            if on in c and c[on] / t >= 0.5:
                ex = [k for k in grp[grp["onyomi"] == on]["kanji"].values if k != kanji]
                rules.append({"type": "结构·部首", "description": f"部首「{radical}」→ {on}",
                              "accuracy": c[on] / t, "examples": ex})

    # MC compound rules
    if mc_initial and mc_grade:
        key = mc_initial + "+" + mc_grade
        grp = ldf[(ldf["_mc_init"] == mc_initial) & (ldf["_mc_grade"] == mc_grade)]
        if len(grp) >= 3:
            c = Counter(grp["onyomi"]); t = sum(c.values())
            if on in c and c[on] / t >= 0.5:
                ex = [k for k in grp[grp["onyomi"] == on]["kanji"].values if k != kanji]
                rules.append({"type": "中古音·声母+等",
                              "description": f"MC声母+等「{mc_initial}+{mc_grade}等」→ {on}",
                              "accuracy": c[on] / t, "examples": ex})

    if mc_rhyme and mc_tone:
        key = mc_rhyme + "+" + mc_tone
        grp = ldf[(ldf["_mc_rhyme"] == mc_rhyme) & (ldf["_mc_tone"] == mc_tone)]
        if len(grp) >= 3:
            c = Counter(grp["onyomi"]); t = sum(c.values())
            if on in c and c[on] / t >= 0.5:
                ex = [k for k in grp[grp["onyomi"] == on]["kanji"].values if k != kanji]
                rules.append({"type": "中古音·韵+调",
                              "description": f"MC韵+调「{mc_rhyme}+{mc_tone}」→ {on}",
                              "accuracy": c[on] / t, "examples": ex})

    return rules


# Lazy-loaded centroid rules for kun semantic analysis
_kun_centroid_cache = None
_kun_subcluster_cache = None
_kun_llm_labels_cache = None


def _load_kun_data():
    global _kun_centroid_cache, _kun_subcluster_cache, _kun_llm_labels_cache
    if _kun_centroid_cache is not None:
        return
    try:
        with open("output/kun_semantic_rules.json") as f:
            data = json.load(f)
        _kun_centroid_cache = data.get("centroid_rules", {})
        _kun_subcluster_cache = data.get("subcluster_rules", {})
        _kun_llm_labels_cache = data.get("llm_labels", {})
    except Exception:
        _kun_centroid_cache = {}
        _kun_subcluster_cache = {}
        _kun_llm_labels_cache = {}


def _semantic_analysis(kun_reading, kanji, meaning, same_kun_meanings):
    """Semantic analysis using centroid rules, sub-clusters, and LLM labels.
    Returns dict with: features (list), label (str or None), sub_group_idx (int or None)
    """
    _load_kun_data()

    llm_info = _kun_llm_labels_cache.get(kun_reading, {})

    # Check sub-clusters first (most specific)
    if kun_reading in _kun_subcluster_cache:
        sc = _kun_subcluster_cache[kun_reading]
        # Find which sub-group contains the current kanji
        for idx, sg in enumerate(sc["sub_groups"]):
            if kanji in sg["kanji"]:
                feats = [w for w in sg["features"] if len(w) > 1]
                label = sg.get("_label")
                if not label and llm_info and idx < len(llm_info.get("labels", [])):
                    label = llm_info["labels"][idx]
                return {"features": feats[:4], "label": label, "sub_group_idx": idx}
        # Fallback: find the sub-group with most overlap with shared kanji
        shared_set = {x["kanji"] if isinstance(x, dict) else x for x in same_kun_meanings}
        best_sg = None
        best_idx = 0
        best_overlap = 0
        for idx, sg in enumerate(sc["sub_groups"]):
            overlap = len(set(sg["kanji"]) & shared_set)
            if overlap > best_overlap:
                best_overlap = overlap
                best_sg = sg
                best_idx = idx
        if best_sg:
            feats = [w for w in best_sg["features"] if len(w) > 1]
            label = best_sg.get("_label")
            if not label and llm_info and best_idx < len(llm_info.get("labels", [])):
                label = llm_info["labels"][best_idx]
            return {"features": feats[:4], "label": label, "sub_group_idx": best_idx}

    # Centroid rules
    if kun_reading in _kun_centroid_cache:
        features = _kun_centroid_cache[kun_reading].get("features", [])
        clean = [(w, s) for w, s in features if len(w) > 1]
        return {"features": [w for w, _ in clean[:4]], "label": None, "sub_group_idx": None}

    # Keyword fallback
    all_text = meaning + " " + " ".join(
        sm["meaning"] if isinstance(sm, dict) else sm for sm in same_kun_meanings
    )
    concepts = []
    if any(w in all_text for w in ["集", "会", "合", "寄", "聚", "纏", "揃", "併"]):
        concepts.append("集合/聚集")
    if any(w in all_text for w in ["遭", "遇", "逢", "邂逅", "出会", "出逢"]):
        concepts.append("相遇/遭遇")
    if any(w in all_text for w in ["生き", "生活", "生命", "誕生", "出生", "生誕"]):
        concepts.append("生命/生存")
    if any(w in all_text for w in ["生む", "生ま", "産", "誕生"]):
        concepts.append("产生/出生")
    if any(w in all_text for w in ["生え", "育", "成長", "生長"]):
        concepts.append("生长/成长")
    if any(w in all_text for w in ["合", "一致", "符合", "適"]):
        concepts.append("符合/一致")
    if any(w in all_text for w in ["面", "対", "向", "見"]):
        concepts.append("面对/会面")
    return {"features": concepts, "label": None, "sub_group_idx": None}


def _find_words_with_reading(reading, all_data, match_in_kana=True):
    """Find vocabulary words containing this reading."""
    words = []
    seen = set()
    reading_hira = _katakana_to_hiragana(reading)
    for lv_name, row_data in all_data:
        sw = row_data.get("sample_words", "")
        if pd.notna(sw) and sw:
            try:
                for w in json.loads(sw):
                    word = w["word"]
                    kana_str = w.get("kana", "")
                    if match_in_kana:
                        if (reading in kana_str or reading_hira in kana_str) and word not in seen:
                            seen.add(word)
                            words.append({"word": word, "kana": kana_str, "level": lv_name})
                    else:
                        raw_word = w.get("raw", word)
                        if reading in raw_word and word not in seen:
                            seen.add(word)
                            words.append({"word": word, "kana": kana_str, "level": lv_name})
            except Exception:
                pass
    return words


def _find_words_with_kun(kun, kanji, all_data):
    """Find vocabulary where this kanji uses this specific kun reading."""
    words = []
    seen = set()
    for lv_name, row_data in all_data:
        sw = row_data.get("sample_words", "")
        if pd.notna(sw) and sw:
            try:
                for w in json.loads(sw):
                    word = w["word"]
                    kana_str = w.get("kana", "")
                    if kanji in word and kun in kana_str:
                        if word not in seen:
                            seen.add(word)
                            words.append({"word": word, "kana": kana_str, "level": lv_name})
            except Exception:
                pass
    return words


def _load_level_datasets():
    """Load all JLPT level datasets into memory."""
    datasets = {}
    for lv in ["N5", "N4", "N3", "N2", "N1"]:
        try:
            datasets[lv] = pd.read_csv(f"dataset/{lv.lower()}_dataset.csv")
        except Exception:
            pass
    return datasets


def deep_lookup(kanji):
    """Build structured analysis data for a kanji. Returns a dict."""
    # ── Find kanji across all levels ──
    all_data = []
    for lv in ["N5", "N4", "N3", "N2", "N1"]:
        try:
            df = pd.read_csv(f"dataset/{lv.lower()}_dataset.csv")
            row = df[df["kanji"] == kanji]
            if len(row) > 0:
                all_data.append((lv, row.iloc[0]))
        except Exception:
            pass

    if not all_data:
        return None

    level, row = all_data[0]
    fbase = level.lower()

    # ── Parse readings ──
    on_readings = _split_on_readings(row.get("onyomi_all"))
    kun_readings = _split_kun_readings(row.get("kunyomi_all"))
    primary_on = str(row["onyomi"]) if pd.notna(row["onyomi"]) else ""

    # ── Basic attributes ──
    pinyin = str(row["pinyin"]) if pd.notna(row["pinyin"]) else "?"
    init = str(row["pinyin_initial"]) if pd.notna(row["pinyin_initial"]) else ""
    final = str(row["pinyin_final"]) if pd.notna(row["pinyin_final"]) else ""
    final_nt = re.sub(r"\d", "", final)
    inf = f"{init}+{final_nt}" if init else ""
    ift = f"{init}+{final}" if init else ""
    tone = int(row["pinyin_tone_num"]) if pd.notna(row.get("pinyin_tone_num")) else 0
    nasal = str(row.get("nasal_coda", ""))
    radical = str(row["radical"]) if pd.notna(row["radical"]) else "?"
    nonrad_raw = row.get("non_radical_part", "")
    nonrad = str(nonrad_raw) if pd.notna(nonrad_raw) and str(nonrad_raw) != "nan" else ""
    meaning = str(row["meaning"]) if pd.notna(row["meaning"]) else "?"
    strokes = int(row["total_strokes"]) if pd.notna(row.get("total_strokes")) else 0
    is_entering = int(row["is_entering_tone"]) if pd.notna(row.get("is_entering_tone")) else 0
    entering_coda = str(row.get("entering_coda", "")) if pd.notna(row.get("entering_coda")) else "none"
    # MC features
    mc_initial = str(row["mc_initial"]) if pd.notna(row.get("mc_initial")) else ""
    mc_rhyme = str(row["mc_rhyme"]) if pd.notna(row.get("mc_rhyme")) else ""
    mc_grade = str(row["mc_grade"]) if pd.notna(row.get("mc_grade")) else ""
    mc_openness = str(row["mc_openness"]) if pd.notna(row.get("mc_openness")) else ""
    mc_tone = str(row["mc_tone"]) if pd.notna(row.get("mc_tone")) else ""
    mc_voicing = str(row["mc_voicing"]) if pd.notna(row.get("mc_voicing")) else ""
    mc_rhyme_cat = str(row["mc_rhyme_cat"]) if pd.notna(row.get("mc_rhyme_cat")) else ""
    mc_initial_cat = str(row["mc_initial_cat"]) if pd.notna(row.get("mc_initial_cat")) else ""
    mc_entering_coda_type = str(row["mc_entering_coda"]) if pd.notna(row.get("mc_entering_coda")) else ""
    mc_reading_count = int(row["mc_reading_count"]) if pd.notna(row.get("mc_reading_count")) else 0
    mc_divergent = int(row["mc_divergent"]) if pd.notna(row.get("mc_divergent")) else 0
    # Parse multi-MC readings
    mc_all_readings = []
    mc_readings_json = str(row.get("mc_readings_json", ""))
    if mc_readings_json and mc_readings_json != "nan":
        try:
            mc_all_readings = json.loads(mc_readings_json)
        except Exception:
            pass
    # Parse reading types
    mc_reading_types = []
    mc_reading_types_raw = str(row.get("mc_reading_types", ""))
    if mc_reading_types_raw and mc_reading_types_raw != "nan":
        try:
            mc_reading_types = json.loads(mc_reading_types_raw)
        except Exception:
            pass
    # Parse Go-on/Kan-on primary
    mc_goon = {}
    mc_kanon = {}
    mc_goon_raw = str(row.get("mc_goon_primary", ""))
    if mc_goon_raw and mc_goon_raw != "nan":
        try: mc_goon = json.loads(mc_goon_raw)
        except: pass
    mc_kanon_raw = str(row.get("mc_kanon_primary", ""))
    if mc_kanon_raw and mc_kanon_raw != "nan":
        try: mc_kanon = json.loads(mc_kanon_raw)
        except: pass

    # ── Load level dataset for rule computation ──
    ldf = pd.read_csv(f"dataset/{fbase}_dataset.csv")
    ldf = ldf[ldf["onyomi"].notna() & (ldf["onyomi"] != "")]
    ldf["_inf"] = ldf["pinyin_initial"].fillna("") + "+" + \
                  ldf["pinyin_final"].fillna("").str.replace(r"\d", "", regex=True)
    ldf["_ift"] = ldf["pinyin_initial"].fillna("") + "+" + ldf["pinyin_final"].fillna("")
    ldf["_fin"] = ldf["pinyin_final"].fillna("")
    ldf["_fin_nt"] = ldf["pinyin_final"].fillna("").str.replace(r"\d", "", regex=True)
    ldf["_nrp"] = ldf["non_radical_part"].fillna("")
    ldf["_nas"] = ldf["nasal_coda"].fillna("none")
    ldf["_ent"] = ldf["entering_coda"].fillna("none")
    # MC features for rule matching
    ldf["_mc_init"] = ldf["mc_initial"].fillna("")
    ldf["_mc_rhyme"] = ldf["mc_rhyme"].fillna("")
    ldf["_mc_grade"] = ldf["mc_grade"].fillna("")
    ldf["_mc_open"] = ldf["mc_openness"].fillna("")
    ldf["_mc_tone"] = ldf["mc_tone"].fillna("")
    ldf["_mc_voice"] = ldf["mc_voicing"].fillna("")
    ldf["_mc_rcat"] = ldf["mc_rhyme_cat"].fillna("")
    ldf["_mc_init_cat"] = ldf["mc_initial_cat"].fillna("")
    ldf["_mc_ent_coda"] = ldf["mc_entering_coda"].fillna("")
    ldf["_rad"] = ldf["radical"].fillna("")

    # ── Build result dict ──
    result = {
        "kanji": kanji,
        "pinyin": pinyin,
        "pinyin_breakdown": {
            "initial": init,
            "final": final,
            "final_no_tone": final_nt,
            "tone": tone,
            "nasal": nasal,
        },
        "mc_breakdown": {
            "initial": mc_initial or None,
            "rhyme": mc_rhyme or None,
            "grade": mc_grade or None,
            "openness": mc_openness or None,
            "tone": mc_tone or None,
            "voicing": mc_voicing or None,
            "rhyme_category": mc_rhyme_cat or None,
            "initial_category": mc_initial_cat or None,
            "entering_coda": mc_entering_coda_type or None,
            "reading_count": mc_reading_count,
            "divergent": bool(mc_divergent),
            "all_readings": mc_all_readings if mc_all_readings else None,
            "reading_types": mc_reading_types if mc_reading_types else None,
            "goon_primary": mc_goon if mc_goon else None,
            "kanon_primary": mc_kanon if mc_kanon else None,
        },
        "radical": radical,
        "non_radical_part": nonrad or None,
        "strokes": strokes,
        "level": level,
        "meaning": meaning,
    }

    # ═══════════════════════ 音読 analysis ═══════════════════════
    result["on_readings"] = []
    for on in on_readings:
        entry = {
            "reading": on,
            "is_primary": on == primary_on,
        }

        # Find rules
        rules = _find_rules_for_reading(on, ldf, inf, ift, final, final_nt, nonrad, nasal,
                                         is_entering, entering_coda, kanji,
                                         mc_initial, mc_rhyme, mc_grade,
                                         mc_openness, mc_tone, mc_voicing,
                                         mc_rhyme_cat, mc_initial_cat=mc_initial_cat,
                                         mc_entering_coda_type=mc_entering_coda_type,
                                         radical=radical, strokes=strokes)
        entry["has_rules"] = len(rules) > 0
        entry["rules"] = [{"type": r["type"], "description": r["description"],
                           "accuracy": round(r["accuracy"], 4),
                           "reliability": "deterministic" if r["accuracy"] >= 0.9
                           else "strong" if r["accuracy"] >= 0.7 else "weak",
                           "examples": r["examples"][:6]}
                          for r in rules]

        # Words using this reading
        entry["words"] = _find_words_with_reading(on, all_data)

        # Same-reading kanji across all levels
        same_reading = []
        for lv_name in ["N5", "N4", "N3", "N2", "N1"]:
            try:
                tdf = pd.read_csv(f"dataset/{lv_name.lower()}_dataset.csv")
                same = tdf[(tdf["onyomi"] == on) & (tdf["kanji"] != kanji)]["kanji"].tolist()
                for k in same:
                    if k not in same_reading:
                        same_reading.append(k)
            except Exception:
                pass
        entry["same_reading_kanji"] = same_reading

        result["on_readings"].append(entry)

    # ═══════════════════════ 訓読 analysis ═══════════════════════
    result["kun_readings"] = []
    if kun_readings:
        for kun in kun_readings:
            entry = {"reading": kun}

            # Words using this kun reading
            entry["words"] = _find_words_with_kun(kun, kanji, all_data)

            # Fallback: search other levels
            if not entry["words"]:
                for lv_name in ["N5", "N4", "N3", "N2", "N1"]:
                    if lv_name == level:
                        continue
                    try:
                        tdf = pd.read_csv(f"dataset/{lv_name.lower()}_dataset.csv")
                        trow = tdf[tdf["kanji"] == kanji]
                        if len(trow) > 0:
                            sw = trow.iloc[0].get("sample_words", "")
                            if pd.notna(sw) and sw:
                                for w in json.loads(sw):
                                    if kanji in w["word"] and kun in w.get("kana", ""):
                                        entry["words"].append(
                                            {"word": w["word"], "kana": w["kana"], "level": lv_name})
                    except Exception:
                        pass

            # Other kanji sharing THIS kun reading
            shared_kanji = []
            for lv_name in ["N5", "N4", "N3", "N2", "N1"]:
                try:
                    tdf = pd.read_csv(f"dataset/{lv_name.lower()}_dataset.csv")
                    for _, kr in tdf.iterrows():
                        if kr["kanji"] == kanji:
                            continue
                        kk_kun_raw = str(kr["kunyomi_all"]) if pd.notna(kr["kunyomi_all"]) else ""
                        if not kk_kun_raw or kk_kun_raw == "nan":
                            continue
                        kk_kun_list = _split_kun_readings(kk_kun_raw)
                        if kun in kk_kun_list:
                            kk_meaning = str(kr["meaning"])[:120] if pd.notna(kr["meaning"]) else ""
                            if kr["kanji"] not in [x["kanji"] for x in shared_kanji]:
                                shared_kanji.append({
                                    "kanji": kr["kanji"],
                                    "meaning": kk_meaning,
                                    "level": lv_name,
                                })
                except Exception:
                    pass

            entry["shared_kanji"] = shared_kanji

            # Semantic analysis
            if shared_kanji:
                concepts = _semantic_analysis(kun, kanji, meaning, shared_kanji)
                entry["semantic_concepts"] = concepts
            else:
                entry["semantic_concepts"] = []

            result["kun_readings"].append(entry)

    # ═══════════════════════ Learning path ═══════════════════════
    ruled_on = []
    bare_on = []
    for on in on_readings:
        has = False
        # Simple check functions to avoid repetition
        def _check_group(col, val, on):
            if not val or val in ("", "none", "nan"):
                return False
            grp = ldf[ldf[col] == val]
            if len(grp) < 2:
                return False
            c = Counter(grp["onyomi"])
            return on in c and c[on] / sum(c.values()) >= 0.5
        if _check_group("_inf", inf, on): has = True
        if _check_group("_nrp", nonrad, on): has = True
        if _check_group("_mc_init", mc_initial, on): has = True
        if _check_group("_mc_rhyme", mc_rhyme, on): has = True
        if _check_group("_mc_rcat", mc_rhyme_cat, on): has = True
        if _check_group("_mc_init_cat", mc_initial_cat, on): has = True
        if _check_group("_mc_ent_coda", mc_entering_coda_type, on): has = True
        if _check_group("_rad", radical, on): has = True
        if nasal and nasal != "none":
            grp = ldf[ldf["_nas"] == nasal]
            if len(grp) >= 3: has = True
        if is_entering == 1 and entering_coda and entering_coda != "none":
            grp = ldf[ldf["_ent"] == entering_coda]
            if len(grp) >= 3: has = True
        if has:
            ruled_on.append(on)
        else:
            bare_on.append(on)

    result["learning_path"] = {
        "ruled_readings": ruled_on,
        "memorize_readings": bare_on,
        "kun_readings": kun_readings,
    }

    return result


# ═══════════════════════════════════════════════════════════════
# TEXT FORMATTER
# ═══════════════════════════════════════════════════════════════

def format_text(data):
    """Format analysis data as human-readable text."""
    lines = []

    kanji = data["kanji"]
    pinyin = data["pinyin"]
    radical = data["radical"]
    nonrad = data["non_radical_part"]
    strokes = data["strokes"]
    level = data["level"]
    meaning = data["meaning"]
    pb = data["pinyin_breakdown"]
    on_readings = [r["reading"] for r in data["on_readings"]]
    kun_readings = [r["reading"] for r in data["kun_readings"]]

    # Header
    lines.append("")
    lines.append("╔" + "═" * 64 + "╗")
    lines.append(f"║  {kanji}  拼音: {pinyin}  部首: {radical}"
                 f"  声符: {nonrad or '(なし)'}")
    lines.append(f"║  音読: {' · '.join(on_readings)}")
    lines.append(f"║  訓読: {' · '.join(kun_readings) if kun_readings else '(なし)'}")
    lines.append(f"║  画数: {strokes}  首次出现: {level}")
    lines.append(f"║  拼音拆解: 声母={pb['initial']} 韵母={pb['final']} 声调={pb['tone']} 鼻音={pb['nasal']}")
    mc = data.get("mc_breakdown", {})
    if mc and mc.get("initial"):
        mc_parts = []
        if mc.get("initial"): mc_parts.append(f"声母={mc['initial']}")
        if mc.get("initial_category"): mc_parts.append(f"({mc['initial_category']}音)")
        if mc.get("rhyme"): mc_parts.append(f"韵母={mc['rhyme']}")
        if mc.get("grade"): mc_parts.append(f"{mc['grade']}等")
        if mc.get("openness"): mc_parts.append(mc['openness'])
        if mc.get("tone"): mc_parts.append(f"声调={mc['tone']}")
        if mc.get("voicing"): mc_parts.append(f"清浊={mc['voicing']}")
        lines.append(f"║  中古音(廣韻): {' '.join(mc_parts)}")
        if mc.get("reading_count", 0) > 1:
            lines.append(f"║    └ {mc['reading_count']}个MC读音")
        if mc.get("divergent"):
            goon = mc.get("goon_primary", {})
            kanon = mc.get("kanon_primary", {})
            if goon:
                go_str = f"声={goon.get('initial','')} 韵={goon.get('rhyme','')} {goon.get('tone','')}"
                lines.append(f"║    呉音系: {go_str}")
            if kanon:
                kan_str = f"声={kanon.get('initial','')} 韵={kanon.get('rhyme','')} {kanon.get('tone','')}"
                lines.append(f"║    漢音系: {kan_str}")
    lines.append("╠" + "═" * 64 + "╣")
    # Truncate meaning
    meaning_lines = []
    rest = meaning
    while rest:
        if len(rest) <= 60:
            meaning_lines.append(f"║  {rest}")
            break
        split_at = rest.find("◆", 1)
        if split_at == -1 or split_at > 60:
            split_at = 60
        meaning_lines.append(f"║  {rest[:split_at]}")
        rest = rest[split_at:]
    for ml in meaning_lines[:4]:
        lines.append(ml)
    if len(meaning_lines) > 4:
        lines.append(f"║  …（意思过长，已截断）")
    lines.append("╚" + "═" * 64 + "╝")

    # Section 1: 音読
    s = "一、音読（おんよみ）—— 从中国传入的读音"
    lines.append("")
    lines.append("┌" + "─" * 64 + "┐")
    lines.append(f"│  {s}{' ' * (64 - len(s))}│")
    lines.append("└" + "─" * 64 + "┘")

    for entry in data["on_readings"]:
        on = entry["reading"]
        is_primary = " ★ 主要音读" if entry["is_primary"] else ""
        lines.append(f"\n  ▸ {on}{is_primary}")
        lines.append(f"    {'─'*52}")

        if entry["has_rules"]:
            lines.append(f"    【为什么读{on}？】")
            for r in entry["rules"]:
                rtype = r["type"]
                if rtype.startswith("入声") or rtype.startswith("鼻音"):
                    label = "◈ 構造"
                elif r["reliability"] == "deterministic":
                    label = "✓ 确定"
                elif r["reliability"] == "strong":
                    label = "○ 大概率"
                else:
                    label = "△ 有时"
                lines.append(f"    {label}  [{r['type']}] {r['description']}")
                if r["accuracy"] and not (rtype.startswith("入声") or rtype.startswith("鼻音")):
                    lines.append(f"           精度: {r['accuracy']:.0%}")
                if r["examples"]:
                    lines.append(f"           同规律的字: {' '.join(r['examples'][:6])}")
        else:
            lines.append(f"    【为什么读{on}？】")
            lines.append(f"    ⚠ 无规则可解释 → 这个读音需要死记")
            lines.append(f"    提示: 不是所有音读都有规律。{on}可能是历史上的")
            lines.append(f"          吴音(Go-on)或惯用音残留，不遵循常见对应规则。")

        if entry["words"]:
            lines.append(f"    【含此读音的单词（什么时候读{on}）】")
            for w in entry["words"][:6]:
                lines.append(f"    {w['word']}  →  {w['kana']}  [{w['level']}]")
        else:
            lines.append(f"    【含此读音的单词】（该级别数据中未收录）")

        if entry["same_reading_kanji"]:
            lines.append(f"    【同音字】{' '.join(entry['same_reading_kanji'][:20])}")

    # Section 2: 訓読
    if data["kun_readings"]:
        s = "二、訓読（くんよみ）—— 日语固有读音"
        lines.append("")
        lines.append("")
        lines.append("┌" + "─" * 64 + "┐")
        lines.append(f"│  {s}{' ' * (64 - len(s))}│")
        lines.append("└" + "─" * 64 + "┘")

        kun_display = " · ".join(kun_readings)
        lines.append(f"\n  （此汉字共有 {len(kun_readings)} 个訓読: {kun_display}）")

        for entry in data["kun_readings"]:
            kun = entry["reading"]
            lines.append(f"\n  ▸ {kun}")
            lines.append(f"    {'─'*52}")

            if entry["words"]:
                lines.append(f"    【使用场景】")
                for w in entry["words"][:4]:
                    lines.append(f"    {w['word']}  →  {w['kana']}  [{w['level']}]")

            if entry["shared_kanji"]:
                lines.append(f"    【共享此訓読的汉字】")
                for sk in entry["shared_kanji"][:8]:
                    sm_short = sk["meaning"][:70] + "…" if len(sk["meaning"]) > 70 else sk["meaning"]
                    lines.append(f"    {sk['kanji']} [{sk['level']}] — {sm_short}")

                sc = entry["semantic_concepts"]
                if sc and (sc.get("features") or sc.get("label")):
                    lines.append(f"    【语义关联】")
                    if sc.get("label"):
                        lines.append(f"    子类: {sc['label']}")
                    if sc.get("features"):
                        lines.append(f"    内核: {' / '.join(sc['features'])}")
                else:
                    lines.append(f"    【语义关联】较弱，可能只是读音偶合")

    # Section 3: Learning path
    s = "三、学习路径 —— 怎么记"
    lines.append("")
    lines.append("")
    lines.append("┌" + "─" * 64 + "┐")
    lines.append(f"│  {s}{' ' * (64 - len(s))}│")
    lines.append("└" + "─" * 64 + "┘")

    lp = data["learning_path"]
    lines.append("")
    if lp["ruled_readings"]:
        lines.append(f"  ✅ 有规则可循: {' · '.join(lp['ruled_readings'])}")
        lines.append(f"     → 记住拼音对应关系即可，看到同声母+韵母的字就能类推")
    else:
        lines.append(f"  ✅ 有规则可循: 无")

    if lp["memorize_readings"]:
        lines.append(f"")
        lines.append(f"  ⚠ 需要死记: {' · '.join(lp['memorize_readings'])}")
        for on in lp["memorize_readings"]:
            lines.append(f"     {on}: 无拼音或声符规则可覆盖")
            for entry in data["on_readings"]:
                if entry["reading"] == on and entry["words"]:
                    ex_words = [f"{w['word']}({w['kana']})" for w in entry["words"][:3]]
                    if ex_words:
                        lines.append(f"      通过单词记: {', '.join(ex_words)}")
                    break

    if lp["kun_readings"]:
        lines.append(f"")
        lines.append(f"  📖 訓読: {' · '.join(lp['kun_readings'])}")
        lines.append(f"     → 这是日语固有读法")
        lines.append(f"     → 策略: 通过单词学訓読，不要单独记")

    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    kanji = sys.argv[1] if len(sys.argv) > 1 else None
    if not kanji:
        print("Usage: python3 deep_lookup.py <漢字> [--json]")
        print("Example: python3 deep_lookup.py 会")
        print("         python3 deep_lookup.py 会 --json")
        sys.exit(1)

    data = deep_lookup(kanji)
    if data is None:
        print(f"'{kanji}' not found in any JLPT dataset.")
        sys.exit(1)

    if "--json" in sys.argv:
        output = json.dumps(data, ensure_ascii=False, indent=2)
        print(output)
    else:
        print(format_text(data))
