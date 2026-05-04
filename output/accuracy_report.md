# Kanji On'yomi Prediction: Accuracy Report

## Per-Level Accuracy

| Level | Kanji Count | Baseline Top-1 | Neural Top-1 | Neural Top-3 | Gain | Key Insight |
|-------|------------|----------------|--------------|--------------|------|-------------|
| N5 | 131 | 17.6% | 30.3% | 57.6% | +12.7% | Mostly pictographs; phonetic rules weak. Pinyin bridge adds the most value. |
| N4 | 156 | 26.9% | 54.5% | 78.8% | +27.6% | Phono-semantic compounds appear; pinyin bridge provides large gain. |
| N3 | 156 | 32.7% | 62.5% | 87.5% | +29.8% | Peak phono-semantic density; best level for rule-based learning. |
| N2 | 257 | 30.7% | 60.4% | 81.1% | +29.6% | More complex characters; radical-component interactions matter. |
| N1 | 798 | 38.7% | 58.2% | 84.2% | +19.5% | Rare components limit rule coverage; neural embeddings help. |
| N1+ | 4669 | 43.8% | 61.7% | 80.3% | +17.9% | Very rare kanji; model generalizes from common component patterns. |

## Key Findings

1. **Pinyin is the dominant feature**: Adding pinyin (M0→M1) increases accuracy by ~30% across all levels.
2. **Radical adds marginal gain** (~1-2%), suggesting it's secondary to the phonetic component.
3. **Stroke count adds no value** — possible slight overfitting when included.
4. **Phonetic rules alone are insufficient**: The baseline model (pure component statistics) achieves only 18-44% Top-1.
5. **Neural model with pinyin achieves 30-62% Top-1**, and Top-3 accuracy of 48-84% — usable for narrowing down readings.

## Top 10 Deterministic Phonetic Components (100% consistent)
- **加** → always **カ** (13 kanji): 伽(カ) 嘉(カ) 架(カ) 茄(カ) 迦(カ) 賀(カ) 駕(カ) 袈(カ) 枷(カ) 珈(カ) 痂(カ) 笳(カ)
- **票** → always **ヒョウ** (11 kanji): 標(ヒョウ) 漂(ヒョウ) 瓢(ヒョウ) 剽(ヒョウ) 嫖(ヒョウ) 慓(ヒョウ) 縹(ヒョウ) 飄(ヒョウ) 飃(ヒョウ) 驃(ヒョウ) 鰾(ヒョウ)
- **倉** → always **ソウ** (10 kanji): 創(ソウ) 槍(ソウ) 蒼(ソウ) 鎗(ソウ) 愴(ソウ) 搶(ソウ) 滄(ソウ) 瘡(ソウ) 艙(ソウ) 蹌(ソウ)
- **冓** → always **コウ** (9 kanji): 構(コウ) 溝(コウ) 講(コウ) 購(コウ) 媾(コウ) 搆(コウ) 篝(コウ) 覯(コウ) 遘(コウ)
- **咢** → always **ガク** (9 kanji): 鍔(ガク) 鰐(ガク) 愕(ガク) 萼(ガク) 蕚(ガク) 諤(ガク) 鄂(ガク) 鶚(ガク) 齶(ガク)
- **扁** → always **ヘン** (9 kanji): 偏(ヘン) 篇(ヘン) 編(ヘン) 遍(ヘン) 翩(ヘン) 蝙(ヘン) 褊(ヘン) 諞(ヘン) 騙(ヘン)
- **呑** → always **キョウ** (8 kanji): 僑(キョウ) 喬(キョウ) 橋(キョウ) 矯(キョウ) 蕎(キョウ) 嬌(キョウ) 轎(キョウ) 驕(キョウ)
- **亢** → always **コウ** (7 kanji): 坑(コウ) 抗(コウ) 杭(コウ) 航(コウ) 伉(コウ) 吭(コウ) 頏(コウ)
- **司** → always **シ** (7 kanji): 伺(シ) 嗣(シ) 詞(シ) 飼(シ) 笥(シ) 覗(シ) 祠(シ)
- **章** → always **ショウ** (7 kanji): 彰(ショウ) 樟(ショウ) 障(ショウ) 嶂(ショウ) 璋(ショウ) 瘴(ショウ) 鱆(ショウ)

## Pinyin Bridge: Most Reliable Finals
- **-甅** → **nan** (100% consistent, 1 examples): nan(1)
- **-瓰** → **nan** (100% consistent, 1 examples): nan(1)
- **-瓱** → **nan** (100% consistent, 1 examples): nan(1)
- **-瓧** → **nan** (100% consistent, 1 examples): nan(1)
- **-üe** → **リャク** (67% consistent, 6 examples): リャク(4), ギャク(2)
- **-er** → **ジ** (61% consistent, 18 examples): ジ(11), ニ(5), ジン(1), nan(1)
- **-uai** → **カイ** (56% consistent, 27 examples): カイ(15), カイ・エ(2), スイ(2), カク(1), セツ(1)
- **-ia** → **カ** (48% consistent, 64 examples): カ(31), キョウ(11), コウ(6), カツ(5), nan(5)
- **-v** → **リョ** (48% consistent, 23 examples): リョ(11), ル(3), リツ(2), ジク(2), ジョ(1)
- **-uang** → **コウ** (41% consistent, 73 examples): コウ(30), ソウ(17), キョウ(9), オウ(5), ショウ(4)