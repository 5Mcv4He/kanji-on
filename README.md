<p align="center">
  <samp>学 → ガク · カク · まなぶ</samp>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue?style=flat-square&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/data-N5~N1-red?style=flat-square">
  <img src="https://img.shields.io/badge/kanji-2,179-orange?style=flat-square">
  <img src="https://img.shields.io/badge/rules-79,832-purple?style=flat-square">
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square">
</p>

# Kanji-On

<p align="center">
  <b>以中国人的母语直觉推导日语音读，而非死记硬背。</b>
  <br><br>
  一个「汉语特征 → 日语音读」规则穷尽发现系统。
  <br>
  三重桥接：汉语拼音 · 中古音（廣韻） · 汉字部件。
  <br>
  <b>覆盖 JLPT N5-N1 全部 2,179 个汉字，字覆盖率 100%。</b>
</p>

<br>

---

```
╔════════════════════════════════════════════════════════════════╗
║  学  拼音: xue  部首: 子  声符: 05労-力
║  音読: カク · ガク   训読: まなぶ   画数: 8
║  中古音(廣韻): 声母=匣(喉音) 韵母=江 声调=入 全濁
╠════════════════════════════════════════════════════════════════╣
║  ◆ 知識や技術を身に付ける、習得する
║  ◆ 教えを受け知識や技術を習得するために設けられた場所
╚════════════════════════════════════════════════════════════════╝

  ▸ ガク  为什么读ガク？
    ✓ 确定  [拼音·韵母+调] 韵母 -ue2 → カク  精度100%  同规律: 覚
    ✓ 确定  [声符·精确] 声符「05労-力」→ カク  精度100%  同规律: 覚
    ◈ 構造  [入声·韵尾] 入声-t → 短促音节

    学生(がくせい) · 大学(だいがく) · 大学生(だいがくせい)
    小学生(しょうがくせい) · 入学式(にゅうがくしき)
```

<br>

---

## 为什么不是「背」而是「推导」

作为中文母语者，你天然掌握了日本人没有的武器——**汉语发音**。

| 你已有的直觉 | 可以推导的音读 |
|:--|:--|
| 拼音声母 `j` | カ行（健·建·見·鍵·剣…） |
| 拼音韵母 `-ou` | ウ段長音（有·由·友·右·油…） |
| 声符「方」 | ホウ（坊·妨·房·放·肪…） |
| 中古音「清」韵 + ing | セイ（清·晴·静·精·情…） |

> 这不是联想记忆法。这是**历史语言学规律**——日语汉音源自唐代长安音，现代汉语方言和中古音重构可以反向桥接这个对应关系。

<br>

---

## 规则有多可靠？实测数据说话

用紅宝書 N5-N1 全部 6,990 个词汇实测：

<p align="center">

| | 规则数 | N5 覆盖 | N1 覆盖 |
|:--|:--:|:--:|:--:|
| 网上流传「沪江方法」 | 2 | 85.8% | 85.3% |
| **本项目「五条铁律」** | **5** | **96.0%** | **91.4%** |
| 系统精选规则（全部） | 334-746 | **100%** | **100%** |

</p>

- 五条铁律比沪江方法多覆盖 **6-11%**，只多记 3 条规则
- 系统精选：贪婪集合覆盖压缩 **107:1**，字覆盖率 100%
- 每条规则标注置信度（确定 / 大概率 / 有时 / 偶尔），你决定用哪些

详见 [`output/iron_laws_analysis.json`](output/iron_laws_analysis.json)。

<br>

---

## 五条铁律（看一眼就能用）

1. **声母定行** — 拼音 `j` → カ行、`zh` → サ行、`b` → ハ行 … （准确率 ~78%）
2. **鼻音定长** — 韵尾 `-n/-ng` → 对应长音或拨音（准确率 98%+）
3. **声符定音** — 看到「方」→ ホウ、看到「包」→ ホウ（准确率 ~57-78%）
4. **入声定短** — 中古入声字 → 日语音读为短促音节（ツ·ク·キ·チ·フ）
5. **呉漢分層** — 同一汉字有两种音读？呉音保留濁音、漢音清化 → カク/ガク 都说得通

> 五条铁律 ≈ 中学语文知识的日语版投射。学会 N5 拼音对应关系，剩下的只是扩展声符和 MC 特征。

<br>

---

## 快速开始

```bash
# 安装
pip install pandas numpy xgboost scikit-learn openpyxl pypinyin

# 查询任意 JLPT 汉字（规则已预生成，开箱即用）
python deep_lookup.py 鬱
python deep_lookup.py 学 --json    # JSON 输出
```

<br>

---

## 项目全景

```
三重桥接 ──→ 穷尽枚举 ──→ 规则发现 ──→ 置信度标注 ──→ 贪婪精选 ──→ 输出
    │            │              │              │              │
    │    拼音特征 (声母/韵母/声调/鼻尾)       │         XGBoost 蒸馏
    │    MC 特征 (声母/韵母/等/開合/清濁/入声) │         呉漢分離引擎
    │    部件特征 (声符/部首/非部首/画数)      │         4档置信度
    │                                         │
    └───────── 46,848 汉字 MC 映射表 ──────→ 呉音/漢音 推导
```

**ML 实验线**：统计基线 → 神经网络 (M0-M3) → 改进版 (residual + label smoothing) → XGBoost → 决策树蒸馏为 if-then 规则。代码在 `models/`。

**Web 应用**：`web/index.html` — 交互式汉字查询，支持 N5-N1 切换和 MC 特征筛选。

<br>

---

## 规模

| 层级 | 汉字 | 词汇 | 穷尽规则 | 精选规则 | 覆盖率 |
|:--|--:|--:|--:|--:|:--:|
| N5 | 653 | 831 | 26,377 | 334 | 100% |
| N4 | 1,074 | 1,766 | 42,249 | 470 | 100% |
| N3 | 1,512 | 3,403 | 58,179 | 568 | 100% |
| N2 | 1,875 | 5,127 | 70,958 | 665 | 100% |
| N1 | 2,179 | 6,990 | 79,832 | 746 | 100% |

<br>

---

## 数据来源

- **漢字林** — K'sBookshelf 漢字検索V2（小林芳雄），46,848 字
- **廣韻** — qieyun Python 库 (MIT)，18,212 条
- **JLPT 词汇** — 紅宝書（华东理工大学出版社），~9,500 词
- **拼音** — pypinyin 库

<br>

---

## 文章 & 视觉资产

公众号文章、HTML 设计稿、截图脚本——全部开源在 `docs/`：

| 文件 | 说明 |
|:--|:--|
| [`docs/article.md`](docs/article.md) | 公众号文章原文：「日语音读规律，两条真的够吗？」（[已发表](https://mp.weixin.qq.com/s/7aOmrAOLhf4fSOVdqU9f1w)） |
| [`docs/rules-reference.md`](docs/rules-reference.md) | 746 条精选规则速查手册 |
| [`docs/images/`](docs/images/) | 9 张 HTML 设计源文件（Playwright 截图 → PNG） |
| [`docs/scripts/screenshot.py`](docs/scripts/screenshot.py) | HTML → PNG 批量截图工具 |
| [`docs/scripts/gen_rule_cards.py`](docs/scripts/gen_rule_cards.py) | 规则 JSON → HTML 卡片生成器 |

<p align="center">
  <a href="docs/PROJECT_REPORT.md">项目报告</a> ·
  <a href="docs/SPEC.md">规格书</a> ·
  <a href="docs/article.md">公众号文章</a> ·
  <a href="README.en.md">English</a> ·
  <a href="README.ja.md">日本語</a>
</p>

<p align="center">
  <sub>MIT License · 数据再发布需标注来源（K'sBookshelf + 紅宝書）· 个人学习自由使用</sub>
</p>
