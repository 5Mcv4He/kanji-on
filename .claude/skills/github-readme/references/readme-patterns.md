# 高星 GitHub 项目 README 模式库

基于对 15 个顶级开源项目（React、Vue、FastAPI、TensorFlow、LangChain、browser-use、crawl4ai、shadcn/ui 等）的 README 分析。

## 核心模板结构

```markdown
[LOGO 图片居中]

# 项目名称

一句话价值主张（<30 字，说清楚"这是什么"+"解决什么问题"）

[Badge 行：CI | 版本 | License | 下载量]

3-8 条特性要点（bullet points）

## 快速开始 / 安装            ← 必须是第一个实质性章节
## 使用示例                    ← 紧跟安装
## 文档                        ← 指向外部文档（可选）
## 贡献
## 许可证
```

## 各章节规则

### 开篇（必须 < 20 行）
- Logo 在上，标题在下
- 一句话描述，不允许写"一个 XXX 框架"这种空泛表述
- 必须包含具体技术栈关键词（Python/TypeScript/Rust 等）
- 量化收益优先：说"200% 提速"不说"很快"
- 可以加一句说明解决什么痛点

### Badge 决策
- 必选：CI 状态、版本号（npm/PyPI）、License
- 强烈推荐：下载量/star 数（社交证明）
- 可选：Discord/Twitter、测试覆盖率、PRs Welcome
- 新项目不要放过多的 badge（显得空），老项目 badge 必须保持准确

### 安装
- 单行命令优先（pip install / npm install / curl 一键脚本）
- 复杂安装提供多种途径但默认推荐放最前面
- 如有系统依赖先列 Prerequisites

### 使用示例
- 放最能体现核心能力的 2-3 个最小示例
- 代码必须可以复制即运行
- 复杂项目可以只放最小示例，其余链接到文档

### 文档布局
- 链接到文档站、API 参考、教程、FAQ
- 用描述性链接文字，禁止裸 URL
- 大型项目：README 当 landing page，其他外链
- 小型项目：README 即完整文档

### 视觉决策树
- 有品牌 → LOGO 居中顶部
- 输出是 UI/终端 → 放截图
- 输出涉及动态交互 → 放 GIF/视频
- 增长数据好 → 放 star history 图
- 架构不直观 → 放架构图
- 贡献者多 → 放 contrib.rocks 头像墙
- 什么都没有 → 不放图也比放一张无关图强

## 绝对禁止
- 大段文字墙（无列表/代码/标题打断）
- 裸 URL（必须用 []() 包裹）
- 安装命令不在首屏
- 空泛描述（"一个强大的框架"）
- 全大写标题
- FAQ 放在顶部
- 截图替代文字说明（两者都要有）

## 高质量一句话描述示例
- "React is a JavaScript library for building user interfaces." — 简洁精确
- "FastAPI is a modern, fast (high-performance), web framework for building APIs with Python based on standard Python type hints." — 含技术栈+关键词
- "Crawl4AI turns the web into clean, LLM-ready Markdown for RAG, agents, and data pipelines." — 问题+方案+目标用户
- "fabric is an open-source framework for augmenting humans using AI." — 一句话说清定位
