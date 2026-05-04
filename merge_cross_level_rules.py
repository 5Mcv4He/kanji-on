"""Merge XGBoost distilled rules across all levels into a consolidated report.
Also identifies rules that are consistent (appear at early levels, hold at later levels)
vs. rules that degrade (high accuracy early, drop later).
"""
import json
import os
from collections import defaultdict

LEVELS = ["N5", "N4", "N3", "N2", "N1"]

# Load all
all_data = {}
for lv in LEVELS:
    fbase = lv.lower().replace("+", "p")
    try:
        with open(f"output/{fbase}_xgboost_rules.json") as f:
            all_data[lv] = json.load(f)
    except FileNotFoundError:
        print(f"  Missing: {fbase}_xgboost_rules.json")
        all_data[lv] = {"xgb_distilled_rules": [], "direct_rules": []}

# ── Merge all XGBoost distilled rules ──
# Key: (rule_text, target_on) — track first level seen, accuracy per level
rule_tracker = defaultdict(lambda: {"first_level": None, "levels": {}, "examples": ""})

for lv in LEVELS:
    for r in all_data[lv].get("xgb_distilled_rules", []):
        key = (r["rule"], r["onyomi"])
        tracker = rule_tracker[key]
        if tracker["first_level"] is None:
            tracker["first_level"] = lv
            tracker["examples"] = r.get("examples", "")
        tracker["levels"][lv] = {
            "accuracy": r["accuracy"],
            "coverage": r["coverage"],
        }

# Also add direct rules
for lv in LEVELS:
    for r in all_data[lv].get("direct_rules", []):
        key = ("DIRECT: " + r["rule"], r["onyomi"])
        tracker = rule_tracker[key]
        if tracker["first_level"] is None:
            tracker["first_level"] = lv
            tracker["examples"] = r.get("examples", "")
        tracker["levels"][lv] = {
            "accuracy": r["accuracy"],
            "coverage": r["coverage"],
        }

# ── Classify rules ──
consistent = []   # appears ≥2 levels, accuracy stable or improving
degrading = []    # appears ≥2 levels, accuracy dropping
single_level = [] # only at one level
new_discovery = [] # first seen at N3+

for key, tracker in rule_tracker.items():
    rule_text, onyomi = key
    levels_seen = sorted(tracker["levels"].keys())
    if len(levels_seen) >= 2:
        # Check if accuracy is stable
        accs = [tracker["levels"][l]["accuracy"] for l in levels_seen]
        if min(accs) >= 0.7:
            consistent.append((key, tracker))
        elif accs[0] > accs[-1] + 0.15:  # dropped >15%
            degrading.append((key, tracker))
        else:
            consistent.append((key, tracker))  # stable-ish
    else:
        single_level.append((key, tracker))
        if tracker["first_level"] in ("N3", "N2", "N1"):
            new_discovery.append((key, tracker))

# Sort consistent by best accuracy * max coverage
consistent.sort(key=lambda x: -max(v["accuracy"] for v in x[1]["levels"].values()))
degrading.sort(key=lambda x: -(x[1]["levels"][x[1]["first_level"]]["accuracy"] -
                                x[1]["levels"][sorted(x[1]["levels"].keys())[-1]]["accuracy"]))
new_discovery.sort(key=lambda x: -max(v["accuracy"] for v in x[1]["levels"].values()))

# ── Output ──
out = []
out.append("# XGBoost 规则蒸馏 — 跨层分析报告")
out.append("")
out.append("## 方法")
out.append("对每个常见音读训练二分类决策树代理模型，从 XGBoost 和真实标签中提取 if-then 规则。")
out.append("")

out.append("## 跨层一致性规则 (在多个层级保持高准确率)\n")
out.append("| 规则 | →音读 | 首次出现 | N5 | N4 | N3 | N2 | N1 | 正例 |")
out.append("|------|------|----------|----|----|----|----|----|------|")
for (rule_text, onyomi), tracker in consistent[:60]:
    levels_info = tracker["levels"]
    cols = []
    for lv in LEVELS:
        if lv in levels_info:
            info = levels_info[lv]
            cols.append(f"{info['accuracy']:.0%}/{info['coverage']}字")
        else:
            cols.append("-")
    rule_str = rule_text.replace("|", "/")
    out.append(f"| {rule_str} | **{onyomi}** | {tracker['first_level']} | "
               f"{' | '.join(cols)} | {tracker['examples'][:40]} |")

out.append(f"\n## 规则退化 (准确率随层级下降>15%)\n")
out.append("| 规则 | →音读 | 首次 | N5 | N4 | N3 | N2 | N1 |")
out.append("|------|------|------|----|----|----|----|----|")
for (rule_text, onyomi), tracker in degrading[:30]:
    levels_info = tracker["levels"]
    cols = []
    for lv in LEVELS:
        if lv in levels_info:
            info = levels_info[lv]
            cols.append(f"{info['accuracy']:.0%}")
        else:
            cols.append("-")
    rule_str = rule_text.replace("|", "/")
    out.append(f"| {rule_str} | **{onyomi}** | {tracker['first_level']} | "
               f"{' | '.join(cols)} |")

out.append(f"\n## 高层级新发现规则 (N3+首次出现, 按准确率排序)\n")
out.append("| 规则 | →音读 | 首次 | 准确率 | 覆盖 | 正例 |")
out.append("|------|------|------|--------|------|------|")
for (rule_text, onyomi), tracker in new_discovery[:80]:
    lv = tracker["first_level"]
    info = tracker["levels"][lv]
    rule_str = rule_text.replace("|", "/")
    out.append(f"| {rule_str} | **{onyomi}** | {lv} | {info['accuracy']:.0%} | "
               f"{info['coverage']}字 | {tracker['examples'][:50]} |")

# ── Summary stats ──
out.append(f"\n## 统计\n")
out.append(f"- 跨层一致规则: {len(consistent)} 条")
out.append(f"- 退化规则: {len(degrading)} 条")
out.append(f"- 高层新发现规则: {len(new_discovery)} 条")
out.append(f"- 总规则数: {len(rule_tracker)} 条")

# Count 100% rules per level
out.append(f"\n## 各层级100%准确率规则数\n")
out.append(f"| 层级 | 蒸馏规则 | 直接规则 | 示例 |")
out.append(f"|------|---------|---------|------|")
for lv in LEVELS:
    xgb_100 = [r for r in all_data[lv].get("xgb_distilled_rules", []) if r["accuracy"] >= 0.99]
    dir_100 = [r for r in all_data[lv].get("direct_rules", []) if r["accuracy"] >= 0.99]
    example = xgb_100[0]["rule"][:50] if xgb_100 else "-"
    out.append(f"| {lv} | {len(xgb_100)} | {len(dir_100)} | {example} |")

with open("output/cross_level_rules.md", "w") as f:
    f.write("\n".join(out))

print("Saved: output/cross_level_rules.md")
print(f"Consistent: {len(consistent)}, Degrading: {len(degrading)}, New: {len(new_discovery)}")

# ── Print highlights ──
print("\n=== HIGHLIGHTS ===")
print("\nTop 10 cross-level consistent rules:")
for (rule_text, onyomi), tracker in consistent[:10]:
    levels = " → ".join(f"{l}({tracker['levels'][l]['accuracy']:.0%})"
                         for l in sorted(tracker["levels"].keys()))
    print(f"  {rule_text[:70]}")
    print(f"    → {onyomi}  [{levels}]")

print("\nTop 10 rules first discovered at N3+:")
for (rule_text, onyomi), tracker in new_discovery[:10]:
    lv = tracker["first_level"]
    info = tracker["levels"][lv]
    print(f"  [{lv}] {rule_text[:70]}")
    print(f"    → {onyomi}  acc={info['accuracy']:.0%}  cov={info['coverage']}字")
