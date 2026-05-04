"""Build per-level web data from rules + datasets + kun semantic rules.

Output: web/data/{n5,n4,n3,n2,n1}.json — each ~200-500KB gzipped.

Each level file contains:
  - kanji: {kanji: {onyomi, kunyomi, words_by_reading, kun_semantic, ...}}
  - rules: enriched rules sorted by tier→accuracy
  - word_index: all 红宝书 words tagged on/kun
  - stats: coverage by path
"""

import json
import os
import re
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LEVELS = ["N5", "N4", "N3", "N2", "N1"]

# MC terms that appear in rule text
MC_TERMS = ["MC声母", "MC韵母", "MC韵摄", "MC声类", "MC等", "MC開合",
            "MC清濁", "MC声调", "MC入声韵尾", "呉漢異", "入声="]


def katakana_to_hiragana(text):
    return "".join(
        chr(ord(ch) - 0x60) if 0x30A0 < ord(ch) < 0x30F7 else ch
        for ch in text
    )


def reading_in_word_kana(reading, kana_str):
    reading_hira = katakana_to_hiragana(reading)
    return reading in kana_str or reading_hira in kana_str


def clean_kana(kana_str):
    """Remove leading/trailing spaces and normalize."""
    return kana_str.strip().replace("  ", " ")


def clean_meaning(raw):
    """Truncate 漢字林 academic meaning to a short summary."""
    if not raw or str(raw) == "nan":
        return ""
    # Take first ◆ segment as the primary meaning
    parts = str(raw).split("◆")
    if len(parts) > 1:
        # First part is empty (before first ◆), use the second
        first = parts[1].strip()
        # Truncate to ~100 chars
        if len(first) > 120:
            first = first[:120] + "..."
        return first
    return str(raw)[:120]


def needs_mc_knowledge(feature_type, rule_text):
    if feature_type in ("mc_standalone", "mc_internal_pair"):
        return True
    return any(term in rule_text for term in MC_TERMS)


def make_rule_human(rule_text, feature_type):
    """Make rule text accessible. Returns simplified version or None."""
    if feature_type in ("pinyin_standalone", "pinyin_pair", "component", "structure"):
        return rule_text
    # Try replacing MC prefixes with accessible terms
    replacements = {
        "MC韵母": "古代韵母", "MC声母": "古代声母",
        "MC韵摄": "古代韵部", "MC声类": "古代声类",
        "MC等": "等呼", "MC開合": "开合",
        "MC清濁": "清浊", "MC声调": "声调",
        "MC入声韵尾": "入声韵尾",
    }
    simplified = rule_text
    for old, new in replacements.items():
        simplified = simplified.replace(old, new)
    if simplified != rule_text:
        return simplified
    return None


def split_readings(raw):
    """Split pipe-separated readings into clean list."""
    if pd.isna(raw) or not raw or str(raw) == "nan":
        return []
    return [x.strip() for x in str(raw).split("|") if x.strip() and x.strip() != "?"]


def split_kun(raw):
    """Split kun readings, removing okurigana markers (· or ・)."""
    readings = split_readings(raw)
    result = []
    for r in readings:
        # Remove okurigana dot markers: た・べる → たべる
        clean = r.replace("・", "").replace("·", "")
        if clean:
            result.append(clean)
    return result


def parse_words(raw):
    """Parse sample_words JSON. Returns list of {word, kana}."""
    try:
        if raw is None:
            return []
        s = str(raw)
        if s in ("[]", "", "nan"):
            return []
        return json.loads(s)
    except (json.JSONDecodeError, TypeError, ValueError):
        return []


def load_kun_semantic():
    """Load kun semantic groups. Returns dict: kun_reading → {features, kanji}."""
    path = f"{HERE}/output/kun_semantic_rules.json"
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    # Use l1_balanced_rules as primary source
    return data.get("l1_balanced_rules", {})


def load_rules(level):
    path = f"{HERE}/output/{level.lower()}_tiered_rules.json"
    with open(path) as f:
        return json.load(f)["rules"]


def load_dataset(level):
    return pd.read_csv(f"{HERE}/dataset/{level.lower()}_dataset.csv")


def classify_word_reading(word, kana, kanji, on_readings, kun_readings):
    """Determine if a word uses on'yomi or kun'yomi for this kanji.

    Returns ("on"|"kun", matched_reading).
    """
    kana_clean = clean_kana(kana)
    kana_hira = katakana_to_hiragana(kana_clean)

    # Check on readings first (katakana)
    for on in on_readings:
        on_hira = katakana_to_hiragana(on)
        if on in kana_clean or on_hira in kana_hira:
            return ("on", on)

    # Check kun readings (hiragana)
    for kun in kun_readings:
        if kun in kana_hira:
            return ("kun", kun)
        # Also try stem (remove last 1-2 okurigana chars): たべる→たべ
        if len(kun) > 2:
            for trim in (1, 2):
                stem = kun[:-trim]
                if len(stem) >= 2 and stem in kana_hira:
                    return ("kun", kun)

    # Fallback
    return ("unknown", "")


def build_level(level):
    """Build web data for one JLPT level."""
    rules_raw = load_rules(level)
    df = load_dataset(level)
    kun_sem = load_kun_semantic()

    # ── Build kanji entries ──
    kanji_entries = {}
    kanji_order = []  # Preserve order for prev/next navigation

    for _, row in df.iterrows():
        k = str(row["kanji"])
        kanji_order.append(k)

        on_readings = split_readings(row.get("onyomi_all", ""))
        kun_readings = split_kun(row.get("kunyomi_all", ""))
        words_raw = parse_words(row.get("sample_words", ""))

        # Classify each word as on/kun
        words_tagged = []
        for w in words_raw:
            kana_clean = clean_kana(w["kana"])
            rtype, matched = classify_word_reading(
                w["word"], kana_clean, k, on_readings, kun_readings
            )
            words_tagged.append({
                "word": w["word"],
                "kana": kana_clean,
                "reading_type": rtype,  # "on" | "kun" | "unknown"
                "reading_used": matched,
            })

        # Group words by reading
        words_by_reading = {}
        for wt in words_tagged:
            key = wt["reading_used"] if wt["reading_used"] else "__unknown__"
            if key not in words_by_reading:
                words_by_reading[key] = []
            words_by_reading[key].append(wt)

        # Build kun semantic data
        kun_with_semantic = []
        for kun in kun_readings:
            sem = kun_sem.get(kun) if kun_sem else None
            entry = {"reading": kun}
            if sem:
                # Get features (semantic description)
                features = [f[0] for f in sem.get("features", [])[:5]]
                # Get related kanji (same kun reading, same semantic group)
                related = [c for c in sem.get("kanji", []) if c != k][:8]
                entry["semantic_features"] = features
                entry["related_kanji"] = related
            kun_with_semantic.append(entry)

        kanji_entries[k] = {
            "kanji": k,
            "onyomi": on_readings,
            "kunyomi": kun_readings,
            "pinyin": str(row.get("pinyin", "")),
            "pinyin_initial": str(row.get("pinyin_initial", "")),
            "pinyin_final": str(row.get("pinyin_final", "")),
            "pinyin_tone": int(row.get("pinyin_tone_num", 0)) if pd.notna(row.get("pinyin_tone_num")) else 0,
            "radical": str(row.get("radical", "")),
            "non_radical_part": str(row.get("non_radical_part", "")),
            "total_strokes": int(row.get("total_strokes", 0)) if pd.notna(row.get("total_strokes")) else 0,
            "meaning": clean_meaning(row.get("meaning", "")),
            "nasal_coda": str(row.get("nasal_coda", "")),
            "mc_initial": str(row.get("mc_initial", "")),
            "mc_rhyme": str(row.get("mc_rhyme", "")),
            "mc_rhyme_cat": str(row.get("mc_rhyme_cat", "")),
            "mc_voicing": str(row.get("mc_voicing", "")),
            "mc_divergent": int(row.get("mc_divergent", 0)) if pd.notna(row.get("mc_divergent")) else 0,
            "words": words_tagged,
            "words_by_reading": words_by_reading,
            "kun_semantic": kun_with_semantic,
            "matched_rules": [],  # Will be filled below
        }

    # ── Fill same-reading kanji links ──
    on_to_kanji = {}
    for k, entry in kanji_entries.items():
        for on in entry["onyomi"]:
            on_to_kanji.setdefault(on, []).append(k)
    kun_to_kanji = {}
    for k, entry in kanji_entries.items():
        for kun_info in entry["kun_semantic"]:
            kun_to_kanji.setdefault(kun_info["reading"], []).append(k)

    # Inject related kanji into each entry
    for k, entry in kanji_entries.items():
        # Same-on-reading kanji
        entry["same_on"] = {}
        for on in entry["onyomi"]:
            entry["same_on"][on] = [c for c in on_to_kanji.get(on, []) if c != k][:12]
        # Same-kun kanji (from kun_semantic data or kun_to_kanji index)
        for kun_info in entry["kun_semantic"]:
            kun = kun_info["reading"]
            if not kun_info.get("related_kanji"):
                kun_info["related_kanji"] = [c for c in kun_to_kanji.get(kun, []) if c != k][:12]

    # ── Enrich rules ──
    enriched_rules = []
    for rule in rules_raw:
        onyomi = rule["onyomi"]
        feature_type = rule.get("feature_type", "")
        tier = rule.get("tier", 1)
        kanji_list = rule.get("correct_examples", [])

        needs_mc = needs_mc_knowledge(feature_type, rule["rule_text"])
        rule_human = make_rule_human(rule["rule_text"], feature_type)
        if rule_human == rule["rule_text"] and needs_mc:
            rule_human = None

        # Find words for this rule
        rule_words = []
        seen_wk = set()
        for c in kanji_list:
            if c in kanji_entries:
                for wt in kanji_entries[c]["words"]:
                    wk = (wt["word"], wt["kana"])
                    if wk not in seen_wk and reading_in_word_kana(onyomi, wt["kana"]):
                        seen_wk.add(wk)
                        rule_words.append({
                            "word": wt["word"],
                            "kana": wt["kana"],
                            "kanji": c,
                        })

        enriched_rules.append({
            "rank": rule["rank"],
            "rule_text": rule["rule_text"],
            "rule_human": rule_human,
            "onyomi": onyomi,
            "source": rule.get("source", ""),
            "tier": tier,
            "accuracy": rule["accuracy"],
            "confidence": rule["confidence"],
            "feature_type": feature_type,
            "needs_mc": needs_mc,
            "reading_type": rule.get("reading_type", "general"),
            "kanji_matches": kanji_list,
            "exception_kanji": rule.get("exception_examples", []),
            "words": rule_words,
            "word_count": len(rule_words),
        })

    # Sort rules: tier first, then accuracy desc
    enriched_rules.sort(key=lambda r: (r["tier"], -r["accuracy"]))

    # ── Fill matched_rules for each kanji ──
    for rule in enriched_rules:
        for c in rule["kanji_matches"]:
            if c in kanji_entries and rule["onyomi"] in kanji_entries[c]["onyomi"]:
                kanji_entries[c]["matched_rules"].append(rule["rank"])

    # ── Build word index ──
    word_index = []
    seen_words = set()
    for k, entry in kanji_entries.items():
        for wt in entry["words"]:
            key = (wt["word"], wt["kana"])
            if key not in seen_words:
                seen_words.add(key)
                word_index.append({
                    "word": wt["word"],
                    "kana": wt["kana"],
                    "reading_type": wt["reading_type"],
                    "kanji": k,
                })

    # ── Stats ──
    path_a = [r for r in enriched_rules if not r["needs_mc"] and r["tier"] <= 2]
    path_b = [r for r in enriched_rules if not r["needs_mc"] and r["tier"] <= 3]

    def count_coverage(ruleset):
        covered = set()
        for r in ruleset:
            for c in r["kanji_matches"]:
                covered.add(c)
        return len(covered)

    total_k = len(kanji_entries)
    cov_a = count_coverage(path_a)
    cov_b = count_coverage(path_b)
    cov_c = count_coverage(enriched_rules)

    stats = {
        "total_kanji": total_k,
        "total_words": len(word_index),
        "total_rules": len(enriched_rules),
        "path_a": {
            "name": "快速入门",
            "description": "只用拼音和声符知识",
            "tiers": "1-2",
            "rules": len(path_a),
            "coverage_kanji": cov_a,
            "coverage_pct": round(cov_a / total_k * 100, 1),
        },
        "path_b": {
            "name": "深度学习",
            "description": "拼音+声符+双特征配对",
            "tiers": "1-3",
            "rules": len(path_b),
            "coverage_kanji": cov_b,
            "coverage_pct": round(cov_b / total_k * 100, 1),
        },
        "path_c": {
            "name": "追求极致",
            "description": "全部规则，含古汉语推导",
            "tiers": "1-5",
            "rules": len(enriched_rules),
            "coverage_kanji": cov_c,
            "coverage_pct": round(cov_c / total_k * 100, 1),
        },
    }

    return {
        "level": level,
        "stats": stats,
        "kanji": kanji_entries,
        "kanji_order": kanji_order,
        "rules": enriched_rules,
        "word_index": sorted(word_index, key=lambda x: x["kana"]),
    }


def main():
    out_dir = f"{HERE}/web/data"
    os.makedirs(out_dir, exist_ok=True)

    for lv in LEVELS:
        print(f"Building {lv}...", end=" ")
        ldata = build_level(lv)
        out_path = f"{out_dir}/{lv.lower()}.json"
        with open(out_path, "w") as f:
            json.dump(ldata, f, ensure_ascii=False)
        size_kb = os.path.getsize(out_path) / 1024
        s = ldata["stats"]
        print(f"{size_kb:.0f}KB | {s['total_rules']} rules, {s['total_words']} words, "
              f"{s['total_kanji']} kanji")
        print(f"  Path A: {s['path_a']['rules']} rules, {s['path_a']['coverage_pct']}%")
        print(f"  Path B: {s['path_b']['rules']} rules, {s['path_b']['coverage_pct']}%")
        print(f"  Path C: {s['path_c']['rules']} rules, {s['path_c']['coverage_pct']}%")

    # Also write an index file
    index = {
        "levels": LEVELS,
        "default_level": "N5",
    }
    with open(f"{out_dir}/index.json", "w") as f:
        json.dump(index, f, ensure_ascii=False)
    print(f"\nDone. Files in {out_dir}/")


if __name__ == "__main__":
    main()
