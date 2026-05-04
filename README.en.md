# Kanji-On

> Exhaustive on'yomi rule system for Chinese JLPT learners
>
> A rule discovery system bridging Modern Chinese (Pinyin), Middle Chinese (Guangyun), and kanji components to exhaustively find all human-interpretable "feature → Japanese on'yomi" correspondence rules. Every rule is annotated with JLPT level, difficulty tier, and confidence score. It's not about memorizing readings — it's about **understanding why**.

[中文](README.md) | [日本語](README.ja.md)

---

## Quick Start

```bash
pip install pandas numpy xgboost scikit-learn openpyxl pypinyin

# Rules are pre-generated in output/ — query any JLPT kanji
python deep_lookup.py 鬱
```

Output: basic info (pinyin / radical / phonetic component / strokes / meaning) → full Middle Chinese decomposition with go-on/kan-on annotations → matched rules per reading → kun'yomi semantic analysis → learning path suggestions.

---

## Scale

| Level | Kanji | Words | Exhaustive | Confident | Selected | Coverage |
|-------|-------|-------|-----------|-----------|----------|----------|
| N5 | 653 | 831 | 26,377 | 938 | 334 | 100% |
| N4 | 1,074 | 1,766 | 42,249 | 1,917 | 470 | 100% |
| N3 | 1,512 | 3,403 | 58,179 | 2,825 | 568 | 100% |
| N2 | 1,875 | 5,127 | 70,958 | 3,511 | 665 | 100% |
| N1 | 2,179 | 6,990 | 79,832 | 4,218 | 746 | 100% |

> Confident = precision ≥ 80%. Selected = greedy set-cover at 107:1 compression, maintaining 100% kanji coverage.

---

## Core Design

### Triple Bridge

```
Pinyin ──→ Japanese on'yomi        Knowledge all Chinese speakers already have
Middle Chinese (Guangyun) ──→ On'yomi   Finer-grained phoneme mapping, enables go/kan separation
Phonetic components ──→ On'yomi     See this component → read this sound
```

### Five Iron Laws vs System Rules — Benchmarked

Real-data comparison using all 紅宝書 N5-N1 vocabulary:

| Approach | Rules | N5 Coverage | N1 Coverage | Notes |
|----------|-------|-------------|-------------|-------|
| Hujiang 2 rules (init→row + nasal→long) | 2 | 85.8% | 85.3% | Common online method |
| **Five Iron Laws** (+ component→on + entering tone + go/kan) | 5 | 96.0% | 91.4% | Minimal rule set from this project |
| **System selected** (Tier 1-3, high precision) | 94-298 | 62% | 84% | Precision ≥ 98%, safe to use |
| **System selected** (all 5 Tiers) | 334-746 | **100%** | **100%** | Exhaustive coverage |

> Five iron laws beat the common online method by 6-11% with only 3 extra rules. System rules achieve 100% but need 334-746 rules. **Recommended**: start with five iron laws (91-96%), then add tiers incrementally.

Full data: [`output/iron_laws_analysis.json`](output/iron_laws_analysis.json).

### On/Kun Discrimination — Which reading for which word?

Three formal rules extracted from the full 紅宝書 vocabulary:

| Rule | Description | Measured Accuracy |
|------|-------------|-------------------|
| Okurigana Rule | Kanji + trailing kana → kun'yomi (食べる·見る) | **99%+** |
| Compound Rule | 2+ consecutive kanji → on'yomi (学生·電車) | **98%+** |
| Single Kanji Heuristic | Solo kanji → try kun'yomi first, then on | Heuristic |

Exceptions: jukujikun (今日→きょう), mixed readings (場所→ばしょ, 夕刊→ゆうかん).

### Four Confidence Levels

| Confidence | Precision | N1 Count |
|-----------|-----------|----------|
| Confident | ≥ 80% | 4,218 |
| Probable | 60-80% | 3,864 |
| Possible | 40-60% | 5,065 |
| Rare | < 40% | 66,685 |

### Three Learning Paths

| Path | Tiers | N5 Coverage | N1 Coverage | Best For |
|------|-------|-------------|-------------|----------|
| A·Quick Start | 1-2 | 38% | 36% | Pinyin + components only |
| B·Deep Study | 1-3 | 47% | 44% | Adding MC dual-feature pairs |
| C·Maximum | 1-5 | 96% | 85% | MC derivation + go/kan separation |

### Go-on / Kan-on Separation

Based on the MC → Japanese phonological derivation engine (`predict_go_kan.py`):

| MC Feature | Go-on | Kan-on |
|-----------|-------|--------|
| Voiced initials (b/d/g/dz) | Voiced (バ/ダ/ガ/ザ行) | Unvoiced (ハ/タ/カ/サ行) |
| Ming initial (m-) | マ行 | バ行 |
| Ri initial (ny-) | ナ/ニャ行 | ザ/ジャ行 |
| Geng she + non-entering | -ャウ | -エイ |

---

## ML Experiments

Beyond rule discovery, the project includes a full ML experimentation pipeline:

| Model | Method |
|-------|--------|
| Baseline (statistical) | P(on \| component) + Laplace smoothing |
| M0-M3 (neural) | Component embeddings + pinyin/radical/stroke features |
| Improved (residual) | Deeper MLP + Label Smoothing + AdamW + Cosine Annealing |
| XGBoost → distillation | Multi-class → decision tree → if-then rules |

Models: `models/`. Pretrained: `models/saved/`. Distilled rules: `output/n*_xgboost_rules.json`.

---

## Web App

`web/index.html` — interactive kanji lookup with N5-N1 switching and MC feature toggles.

```bash
python build_web_data.py   # generate web/data/*.json
# then open web/index.html with any HTTP server
```

---

## Commands

```bash
# Lookup
python deep_lookup.py 鬱              # Deep analysis
python kanji_lookup.py 学 --level N5  # Quick lookup

# Rule generation
python tiered_rules.py N5             # Exhaustive + selected + MD report

# Model training (optional)
python train_level_xgboost.py N1      # XGBoost
python distill_xgboost_rules.py N1    # Decision tree distillation
python models/baseline.py             # Statistical baseline
python models/train.py                # Neural M0-M3
python models/train_v2.py             # Improved neural

# Utilities
python extract_onkun_rules.py         # On/kun discrimination rules
python merge_cross_level_rules.py     # Cross-level consistency
python build_web_data.py              # Web data generation
```

---

## Docs

- [Full Project Report (Chinese)](docs/PROJECT_REPORT.md)
- [Spec (Chinese)](docs/SPEC.md)

---

## License

Code: MIT License. Data redistribution requires source attribution (K'sBookshelf + 紅宝書). Free for personal study.
