---
name: github-readme
description: 按高星 GitHub 项目规律生成项目 README。当用户说"写README"、"README怎么写"、"开源项目首页"、"项目介绍"、"整理README"、"GitHub项目页面"时使用此技能。
---

# GitHub README 生成器

基于对 15 个顶级开源项目（合计 1000 万+ star）的 README 分析，按照已验证的高星模式生成项目 README。

## 何时使用

- 用户要创建新项目的 README
- 用户要改进/重写现有 README
- 用户问"README 怎么写"或"开源项目怎么展示"
- 用户准备把项目上传到 GitHub

## 执行流程

### Step 0: 收集信息

如果用户没有提供完整信息，逐一询问以下内容（每次最多 4 个问题）：

**必问项：**
1. 项目名称
2. 项目一句话定位（是什么，解决什么问题，目标用户是谁）
3. 技术栈/语言
4. 安装方式（一条命令）

**选问项：**
- 有 Logo 吗？路径是什么？
- 核心特性 3-5 条
- 需要截图/GIF 吗？路径是什么？
- 文档站链接？
- License 类型？（不指定默认 MIT）
- 你认为用户为什么应该选你的项目而不是竞品？（差异点）

### Step 1: 读取参考模式库

**必须先用 Read 读取 `./references/readme-patterns.md`**，确保生成内容符合已验证的高星模式。

### Step 2: 生成 README

根据收集到的信息，按以下结构生成 README。**以下模板是结构骨架，具体风格必须遵守「风格化规则」一节。**

```markdown
<p align="center">
  [迷你 demo 行或 logo，用 <samp> 或图片]
</p>

<p align="center">
  [Badge 行，用 shields.io，含实际项目数据]
</p>

# 项目名

<p align="center">
  <b>[一句话价值主张]</b>
  <br>
  [补充定位描述，2-3 行]
</p>

---

[**Hero Demo**: 终端截图/输出/ GIF — 让用户不安装就知道这工具长什么样]

---

## [hook 章节名，形如「为什么不是 X 而是 Y」]

[用一个小表或 3-4 条对比，展示核心价值]

## [量化对比 / benchmark]

[表格：本项目 vs 竞品/替代方案，用数字说话]

## [最小可用知识 / quick wins]

[3-5 条「看一眼就能用」的规则或要点，每条 1 行]

## 快速开始

```bash
# 安装命令（单行优先）
```

## 项目全景

[架构流程或模块关系，用 ASCII 图或短列表]

## 规模 (可选，数据密集型项目放)

[1 张表，不超过 6 行]

## 数据来源 (可选)

## 文档

- [链接](url)

---

<p align="center">
  [文档链接] · [语言切换] · [其他链接]
</p>

<p align="center">
  <sub>[License 信息]</sub>
</p>
```

## 风格化规则

以下规则来自实际 README 迭代中验证有效的模式，优先级高于基础模板。

### 视觉层次

- **Hero Demo 前置**：在标题和 badge 之后、安装命令之前，放终端截图/输出/GIF。用户不安装就知道这工具长什么样、输出什么。CLI 工具直接贴真实终端输出（含 box-drawing 字符），Web 工具放截图。
- **居中布局**：标题区、badge 行、一句话定位、benchmark 表、底部链接——这些用 `<p align="center">` 包裹。不用 HTML 做复杂布局，只做居中。
- **`<br>` 呼吸**：每个 `---` 分节前加 `<br>`，让分节之间有空隙。GitHub 默认行距很紧。
- **单文件单语言**：多语言 README 拆成独立文件（`README.md` / `README.en.md` / `README.ja.md`），用底部链接切换。不要塞进一个文件，又长又丑。

### 内容结构

- **Hook 章节命名**：不要叫"特性"或"核心设计"。用对比句式——「为什么不是 X 而是 Y」「规则有多可靠？实测数据说话」「看一眼就能用」。标题本身就是信息。
- **Head-to-head benchmark 必须放**：如果你有竞品/替代方案，拉表比数字。规则数、覆盖率、准确率——数据比形容词可信 10 倍。表不超过 4 列 5 行。
- **Minimum Viable Knowledge**：给用户一个「看完就能用」的段落（3-5 条，每条 1 行）。不要铺方法论，先给可操作的知识。细节放文档。
- **项目全景用 ASCII 流程图**：`──→` 和 `│` 画一行流程图比 20 行文件树更有信息量。文件树太长，放文档不放 README。

### 砍掉什么

以下内容**不要出现在 README 里**，属于文档范畴：
- 详细置信度/难度分级大表 → 一句话带过，链接到项目报告
- 多级学习路径详解 → 同上
- 完整项目文件树 → 只列 5-8 个核心文件
- 全部 CLI 命令 → 只列 3-5 个最常用的
- FAQ → 永远不放
- 贡献指南 → 独立 CONTRIBUTING.md

README 是 landing page，不是文档目录。超过 200 行就要砍。

### Badge 规范

- 用 `shields.io` 的 `flat-square` 风格
- 除了 CI / License / Python版本，**用 badge 展示项目数据**：`data-N5~N1-red`、`kanji-2,179-orange`、`rules-79,832-purple`。这比一句话描述更快建立信任
- 新项目不放过多的 badge（显得空），有数据的 badge 优先于状态 badge

### Emoji 纪律

- 只用功能性标记：✓ 确定、✗ 否定、✅ 推荐、📖 参考
- **禁止**：装饰性 emoji（✨ 🚀 💪 🔥 ⭐ 🎉），这些降低专业感
- 每条规则或要点前最多 1 个 emoji

### 语言

- 中文 README 用口语化、有观点的中文。不要翻译腔。「以中国人的母语直觉推导日语音读，而非死记硬背」比「面向中国 JLPT 学习者的汉字音读规则发现系统」好 10 倍。
- 不用说「欢迎贡献」「如有问题请提 issue」——这些是废话，没人看了这个就去提 issue。

### Step 3: 对照检查清单自查

生成后逐项检查：

- [ ] Hero Demo 在首屏（终端输出/截图在 badge 和 install 之间）
- [ ] 安装命令在首屏可视区域（不滚动就能看到）
- [ ] 开头一句话精确描述了项目是什么 + 解决什么问题
- [ ] Badge 行包含项目数据 badge（not just CI/License）
- [ ] 有 head-to-head benchmark 或量化对比表
- [ ] 每条特性有具体数字，没有空话形容词
- [ ] 代码示例复制后能直接运行
- [ ] 没有裸 URL
- [ ] 没有大段文字墙（用标题/列表/代码/表格打断）
- [ ] 没有装饰性 emoji（✨ 🚀 🔥）
- [ ] License 在最后，居中 `<sub>` 小字
- [ ] 总行数 < 200，超了就砍

### Step 4: 对已有 README 的改进策略

1. 如果用户说「重新设计」「从零开始」「太丑了」→ 不参照原文件，从风格化规则重建
2. 如果是小幅优化 → 优先修复：缺少 hero demo、安装命令位置、一句话描述空泛、缺少数据 badge、没有 benchmark 对比
3. 效果：改后行数应显著减少（目标 < 200 行），视觉留白增加

## 关键原则

- **首屏即决策**：打开 GitHub 看到的前半屏决定用户是否继续。Hero Demo + Badge + 一句话定位必须在这一屏
- **Hero Demo 优先**：在安装命令之前展示工具长什么样。CLI 工具贴终端输出，Web 工具放截图
- **示例可运行**：代码块复制粘贴就能跑，不依赖外部上下文
- **数据比形容词可信**：benchmark 对比表、覆盖率数字、规则数——硬数字取代「高性能」「易扩展」
- **一句话说清楚**：写不好一句话描述就不往下写，这是 README 最重要的 30 个字
- **README 是 landing page，不是文档**：超过 200 行就砍，细节指向 docs/

## 一句话描述的黄金公式

```
[项目名] is a [技术类别] for [目标用户] that [核心差异化价值].
```

坏例子：
- "A powerful web framework" → 空洞
- "Another tool for developers" → 没差异化
- "Built with love" → 没有信息量

好例子：
- "React is a JavaScript library for building user interfaces."
- "FastAPI is a modern, fast web framework for building APIs with Python based on standard Python type hints."
- "Crawl4AI turns the web into clean, LLM-ready Markdown for RAG, agents, and data pipelines."

**最后更新**: 2026-05-04（风格化规则来自 kanji-on README 3 次迭代验证）

---

> 基于 15 个顶级开源项目的 README 分析生成 | 持续校准模式库
