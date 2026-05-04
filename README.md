# Kanji-On

> 日语汉字音读规律 · 穷尽规则体系
>
> 以现代汉语拼音、中古音（廣韻）、汉字部件为三重桥接，穷尽发现「特征 → 日语音读」对应规则。面向中国 JLPT 学习者，每一条规则标注层级、难度和置信度。不是记读音，是**理解为什么这样读**。

[English](README.en.md) | [日本語](README.ja.md)

---

## 快速开始

```bash
pip install pandas numpy xgboost scikit-learn openpyxl pypinyin

# 规则已预生成在 output/，直接查任意 JLPT 汉字
python deep_lookup.py 鬱
```

输出：拼音/部首/声符/画数 → 中古音完整分解（呉漢标注）→ 每条音读的匹配规则 → 训读语义分析 → 学习路径建议。

---

## 规模

| 层级 | 汉字 | 词汇 | 穷尽规则 | 确定规则 | 精选规则 | 覆盖率 |
|------|------|------|---------|---------|---------|--------|
| N5 | 653 | 831 | 26,377 | 938 | 334 | 100% |
| N4 | 1,074 | 1,766 | 42,249 | 1,917 | 470 | 100% |
| N3 | 1,512 | 3,403 | 58,179 | 2,825 | 568 | 100% |
| N2 | 1,875 | 5,127 | 70,958 | 3,511 | 665 | 100% |
| N1 | 2,179 | 6,990 | 79,832 | 4,218 | 746 | 100% |

> 确定规则 = 精度 ≥ 80%。精选规则 = 贪婪集合覆盖压缩 107:1，保持 100% 字覆盖率。

---

## 核心设计

### 三重桥接

```
汉语拼音 ──→ 日语音读        中国小学就掌握
中古音（廣韻）──→ 日语音读   更细粒度的音位映射，支撑呉漢分離
汉字声符/部件 ──→ 日语音读   看到这个部件就读这个音
```

### 五条铁律 vs 系统精选 —— 实测对比

用红宝书 N5-N1 全部词汇实测，三种方案的可覆盖汉字比例：

| 方案 | 规则数 | N5 覆盖 | N1 覆盖 | 说明 |
|------|--------|---------|---------|------|
| 沪江 2 规则（声母→行 + 鼻音→长音） | 2 | 85.8% | 85.3% | 网络上流传的方法 |
| **五条铁律**（+ 部件→音 + 入声 + 呉漢） | 5 | 96.0% | 91.4% | 本项目提炼的最简规则集 |
| **系统精选规则**（Tier 1-3，高精度） | 94-298 | 62% | 84% | 精度 ≥ 98%，可放心使用 |
| **系统精选规则**（全部 5 Tiers） | 334-746 | **100%** | **100%** | 穷尽覆盖 |

> 五条铁律比沪江方法多覆盖 6-11%，只需多记 3 条规则。系统精选规则实现 100% 覆盖，但需要 334-746 条规则。**推荐路径**：先用五条铁律快速入门（91-96%），再按 Tier 逐级补充。

详见 [`output/iron_laws_analysis.json`](output/iron_laws_analysis.json)。

### 音訓判別 —— 看到一个词，怎么判断读法？

从红宝书全量词汇中提取的 3 条形式化规则：

| 规则 | 说明 | N5-N1 实测精度 |
|------|------|--------------|
| 送仮名則 | 汉字+送假名 → 訓讀（食べる·見る） | **99%+** |
| 連続漢字則 | 2字以上汉字连续 → 音讀（学生·電車） | **98%+** |
| 単漢字優先則 | 单汉字 → 优先訓讀，无訓則音 | 启发式 |

例外：熟字訓（今日→きょう）、重箱読み（場所→ばしょ）、湯桶読み（夕刊→ゆうかん）。

### 四档置信度

| 置信度 | 精度 | 含义 | N1 数量 |
|--------|------|------|---------|
| 确定 | ≥ 80% | 语言学规律，推荐记忆 | 4,218 |
| 大概率 | 60-80% | 多数情况适用 | 3,864 |
| 有时 | 40-60% | 可参考，反例较多 | 5,065 |
| 偶尔 | < 40% | 弱参考，仅穷尽检索保留 | 66,685 |

### 五级难度 × 三条学习路径

| 路径 | Tier | N5 规则 | N5 覆盖 | N1 覆盖 | 适合 |
|------|------|---------|---------|---------|------|
| A·快速入门 | 1-2 | 156 | 38% | 36% | 只用拼音 + 声符 |
| B·深度学习 | 1-3 | 197 | 47% | 44% | 加入 MC 双特征 |
| C·追求极致 | 1-5 | 334 | 96% | 85% | 中古音推导 + 呉漢 |

### 呉音・漢音分離

基于 MC → 日语音系推导引擎（`predict_go_kan.py`），规则标注 `reading_type`：

| MC 特征 | 呉音 | 漢音 |
|---------|------|------|
| 全濁声母 (b/d/g/dz) | 保留濁音（バ/ダ/ガ/ザ行） | 变清音（ハ/タ/カ/サ行） |
| 明母 (m-) | マ行 | バ行 |
| 日母 (ny-) | ナ/ニャ行 | ザ/ジャ行 |
| 梗攝 + 非入声 | -ャウ | -エイ |

---

## ML 模型实验

除了规则发现，项目还包含完整的 ML 实验管线，用于验证「从部件+拼音预测音读」的可行性：

| 模型 | 方法 | N1 Top-1 精度 |
|------|------|-------------|
| 基线 (统计) | P(on | component) + Laplace 平滑 | — |
| M0-M3 (神经网络) | 部件嵌入 + 拼音/部首/笔画特征 | — |
| 改进版 (residual) | 更深 MLP + Label Smoothing + AdamW | — |
| XGBoost | 多分类 + 决策树蒸馏 → if-then 规则 | — |

模型文件在 `models/`，预训练权重在 `models/saved/`，蒸馏规则在 `output/n*_xgboost_rules.json`。

> XGBoost 蒸馏生成的 if-then 规则（Tier 4-5）进一步补全了手工规则的盲区。

---

## Web 应用

`web/index.html` — 交互式汉字查询，支持 N5-N1 切换、中古音特征筛选。

```bash
python build_web_data.py   # 生成 web/data/*.json
# 然后用任意 HTTP server 打开 web/index.html
```

---

## 全部命令

```bash
# 查询
python deep_lookup.py 鬱              # 深度分析
python deep_lookup.py 鬱 --json       # JSON 输出
python kanji_lookup.py 学 --level N5  # 快速查询
python kanji_lookup.py 日本語 --level N3  # 批量

# 规则生成
python tiered_rules.py N5             # 穷尽规则 + 精选 + MD 报告

# 模型训练（可选）
python train_level_xgboost.py N1      # XGBoost 训练
python distill_xgboost_rules.py N1    # 决策树蒸馏
python models/baseline.py             # 统计基线
python models/train.py                # 神经网络 M0-M3
python models/train_v2.py             # 改进版神经网络

# 辅助
python extract_onkun_rules.py         # 音訓判別规则
python merge_cross_level_rules.py     # 跨层规则一致性
python build_web_data.py              # Web 数据构建
```

---

## 项目结构

```
kanji-on/
├── tiered_rules.py              # ★ 穷尽规则生成 + 贪婪精选
├── deep_lookup.py               # ★ 深度汉字分析（主 CLI）
├── kanji_lookup.py              # 快速汉字查询
├── predict_go_kan.py            # MC → 呉音/漢音 音系推导引擎
├── extract_onkun_rules.py       # 音訓判別规则提取
│
├── build_dataset.py             # 全量数据集构建
├── build_level_dataset.py       # N5-N1 分层数据集构建
├── build_mc_data.py             # 廣韻 → 中古音映射表构建
├── enrich_mc_features.py        # MC 特征增强 + 呉漢标注
├── build_web_data.py            # Web 数据生成
│
├── train_level_xgboost.py       # XGBoost 多分类器
├── distill_xgboost_rules.py     # 决策树蒸馏（XGBoost → if-then）
├── merge_cross_level_rules.py   # 跨层规则一致性分析
│
├── models/
│   ├── baseline.py              # 统计基线模型
│   ├── neural_net.py            # PyTorch 神经网络（M0-M3）
│   ├── train.py                 # 模型训练 + 评估
│   ├── train_v2.py              # 改进版（residual + label smoothing）
│   └── saved/                   # 预训练权重
│
├── dataset/                     # N5-N1 累计数据集 (CSV)
├── output/                      # 规则输出 (JSON + MD)
├── web/                         # 交互式汉字查询 Web 应用
└── docs/                        # 项目报告 + 规格书
```

---

## 数据来源

| 数据 | 来源 | 规模 |
|------|------|------|
| 漢字林 | K'sBookshelf 漢字検索V2.xlsm（小林芳雄） | 46,848 字 |
| 廣韻 | qieyun Python 库 (MIT) | 18,212 条 |
| JLPT 词汇 | 紅宝書 word.xlsx（华东理工大学出版社） | ~9,500 词 |
| 拼音 | pypinyin 库 | — |
| KANJIDIC2 | Electronic Dictionary Research Group | 参考读音 |

---

## 文档

- [完整项目报告](docs/PROJECT_REPORT.md) — 设计哲学、系统架构、规则体系、性能评估
- [规格书](docs/SPEC.md) — 技术规格

---

## 许可证

代码：MIT License

数据再发布需标注来源（K'sBookshelf + 紅宝書）。个人学习自由使用。
