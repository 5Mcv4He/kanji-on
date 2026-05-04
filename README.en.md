<div align="center">
  <h1>Kanji-On</h1>
  <samp>学 → ガク · カク · まなぶ</samp>
</div>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue?style=flat-square&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/data-N5~N1-red?style=flat-square">
  <img src="https://img.shields.io/badge/kanji-2,179-orange?style=flat-square">
  <img src="https://img.shields.io/badge/rules-79,832-purple?style=flat-square">
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square">
</p>

<p align="center">
  <b>Derive Japanese on'yomi from your native Chinese intuition, not rote memorization.</b>
  <br><br>
  An exhaustive rule discovery system for "Chinese feature → Japanese on'yomi."
  <br>
  Triple bridge: Modern Pinyin · Middle Chinese (Guangyun) · Kanji components.
  <br>
  <b>100% kanji coverage across JLPT N5-N1 — all 2,179 characters.</b>
</p>

<br>

---

```
╔════════════════════════════════════════════════════════════════╗
║  学  Pinyin: xue  Radical: 子  Phonetic: 05労-力
║  On: カク · ガク   Kun: まなぶ   Strokes: 8
║  MC (Guangyun): Initial=匣(Guttural) Rhyme=江 Tone=入 Voiced
╠════════════════════════════════════════════════════════════════╣
║  ◆ Acquire knowledge/skill through study and imitation
║  ◆ A place established for learning
╚════════════════════════════════════════════════════════════════╝

  ▸ ガク  Why ガク?
    ✓ Confident  [Pinyin·Final+Tone] -ue2 → カク  Prec:100%  Also: 覚
    ✓ Confident  [Component·Exact] 「05労-力」→ カク  Prec:100%  Also: 覚
    ◈ Structural  [Entering Tone·Coda] -t → short syllable

    学生(がくせい) · 大学(だいがく) · 大学生(だいがくせい)
    小学生(しょうがくせい) · 入学式(にゅうがくしき)
```

<br>

---

## Why "derive" instead of "memorize"

As a Chinese native speaker, you already have a weapon no Japanese native has — **Chinese pronunciation**.

| What you already know | On'yomi you can derive |
|:--|:--|
| Pinyin initial `j` | カ行 (健·建·見·鍵·剣…) |
| Pinyin final `-ou` | Long ウ (有·由·友·右·油…) |
| Component 「方」 | ホウ (坊·妨·房·放·肪…) |
| MC "清" rhyme + ing | セイ (清·晴·静·精·情…) |

> This is not mnemonics. This is **historical linguistics** — Japanese kan-on derives from Tang-dynasty Chang'an pronunciation. Modern Chinese dialects and Middle Chinese reconstruction bridge the correspondence in reverse.

<br>

---

## How reliable? Benchmarked on real data

Tested against all 6,990 words from 紅宝書 N5-N1:

<p align="center">

| | Rules | N5 Coverage | N1 Coverage |
|:--|:--:|:--:|:--:|
| Common "Hujiang method" | 2 | 85.8% | 85.3% |
| **Our "Five Iron Laws"** | **5** | **96.0%** | **91.4%** |
| System selected rules (all) | 334-746 | **100%** | **100%** |

</p>

- Five iron laws beat the common method by **6-11%** with only 3 more rules
- System selected: greedy set cover at **107:1** compression, 100% kanji coverage
- Every rule has a confidence label (confident / probable / possible / rare) — you decide what to use

See [`output/iron_laws_analysis.json`](output/iron_laws_analysis.json).

<br>

---

## Five Iron Laws (use immediately)

1. **Initial determines row** — `j` → カ行, `zh` → サ行, `b` → ハ行 … (accuracy ~78%)
2. **Nasal determines length** — `-n/-ng` → long vowel or ン (accuracy 98%+)
3. **Component determines sound** — see 「方」→ ホウ, see 「包」→ ホウ (accuracy ~57-78%)
4. **Entering tone is short** — MC entering-tone characters → clipped syllable (ツ·ク·キ·チ·フ)
5. **Go-on / Kan-on split** — same character, two readings? Go-on preserves voicing, Kan-on devoices → both カク/ガク are explained

> Five iron laws ≈ your middle-school Chinese knowledge projected onto Japanese. Learn N5 pinyin mappings, and the rest is just expanding components and MC features.

<br>

---

## Quick Start

```bash
pip install pandas numpy xgboost scikit-learn openpyxl pypinyin

# Query any JLPT kanji (rules pre-generated, works out of the box)
python deep_lookup.py 鬱
python deep_lookup.py 学 --json    # JSON output
```

<br>

---

## Project Overview

```
Triple Bridge ──→ Exhaustive Enum ──→ Rule Discovery ──→ Confidence ──→ Greedy Select ──→ Output
     │                 │                    │                │               │
     │    Pinyin (initial/final/tone/nasal)            │      XGBoost distillation
     │    MC (initial/rhyme/grade/open/voiced/entering)│      Go/Kan separation
     │    Component (phonetic/radical/strokes)         │      4 confidence tiers
     │                                                 │
     └───────── 46,848-character MC map ───────────→ Go-on/Kan-on engine
```

**ML pipeline**: Statistical baseline → Neural (M0-M3) → Improved (residual + label smoothing) → XGBoost → decision tree distillation into if-then rules. Code in `models/`.

**Web app**: `web/index.html` — interactive kanji lookup with N5-N1 switching and MC feature toggles.

<br>

---

## Scale

| Level | Kanji | Words | Exhaustive | Selected | Coverage |
|:--|--:|--:|--:|--:|:--:|
| N5 | 653 | 831 | 26,377 | 334 | 100% |
| N4 | 1,074 | 1,766 | 42,249 | 470 | 100% |
| N3 | 1,512 | 3,403 | 58,179 | 568 | 100% |
| N2 | 1,875 | 5,127 | 70,958 | 665 | 100% |
| N1 | 2,179 | 6,990 | 79,832 | 746 | 100% |

<br>

---

## Data Sources

- **Kanji Rin** — K'sBookshelf 漢字検索V2 (Kobayashi Yoshio), 46,848 characters
- **Guangyun** — qieyun Python library (MIT), 18,212 entries
- **JLPT Vocabulary** — 紅宝書 (ECUST Press), ~9,500 words
- **Pinyin** — pypinyin library

<br>

---

<p align="center">
  <a href="docs/PROJECT_REPORT.md">Project Report</a> ·
  <a href="docs/SPEC.md">Spec</a> ·
  <a href="docs/rules-reference.md">Rules Reference</a> ·
  <a href="README.md">中文</a> ·
  <a href="README.ja.md">日本語</a>
</p>

<p align="center">
  <sub>MIT License · Data redistribution requires source attribution (K'sBookshelf + 紅宝書) · Free for personal study</sub>
</p>
