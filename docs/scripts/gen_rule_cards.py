"""Generate HTML cards for 746 rules. Split law1/law2 into MC vs pinyin. Only 确定+大概率."""
import json
import os
from collections import defaultdict

JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "output", "n1_tiered_rules.json")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images", "2026-05-04")

with open(JSON_PATH) as f:
    data = json.load(f)
rules = data["rules"]

# Categorize into 5 iron laws
cats_raw = {
    "law1_shengmu": [],
    "law2_yunwei": [],
    "law3_shengfu": [],
    "law4_rusheng": [],
    "law5_wuhan": [],
}

for r in rules:
    rt = r["rule_text"]
    ft = r.get("feature_type", "")
    if ft == "component" or "声符" in rt:
        cats_raw["law3_shengfu"].append(r)
    elif ft == "structure" or ("入声" in rt and "声符" not in rt and "鼻音" not in rt and "入声韵尾" not in rt):
        cats_raw["law4_rusheng"].append(r)
    elif "鼻音" in rt or "入声韵尾" in rt:
        cats_raw["law2_yunwei"].append(r)
    elif r.get("reading_type") in ("go-on", "kan-on"):
        cats_raw["law5_wuhan"].append(r)
    else:
        cats_raw["law1_shengmu"].append(r)

# Split law1 and law2 into MC vs pinyin sub-categories
def is_mc_rule(r):
    rt = r["rule_text"]
    ft = r.get("feature_type", "")
    return ("MC" in rt or "中古音" in rt or
            ft in ("mc_internal_pair", "mc_standalone", "cross_domain_pair"))

law1_all = cats_raw["law1_shengmu"]
law2_all = cats_raw["law2_yunwei"]

cats = {
    "law1_pinyin":  [r for r in law1_all if not is_mc_rule(r)],
    "law1_mc":      [r for r in law1_all if is_mc_rule(r)],
    "law2_pinyin":  [r for r in law2_all if not is_mc_rule(r)],
    "law2_mc":      [r for r in law2_all if is_mc_rule(r)],
    "law3_shengfu": cats_raw["law3_shengfu"],
    "law4_rusheng": cats_raw["law4_rusheng"],
    "law5_wuhan":   cats_raw["law5_wuhan"],
}

# Filter to only 确定 + 大概率
KEEP_CONF = {"确定", "大概率"}
for k in list(cats.keys()):
    cats[k] = [r for r in cats[k] if r["confidence"] in KEEP_CONF]

# ----- Config -----
CONF_ORDER = ["确定", "大概率"]
CONF_COLORS = {
    "确定": ("#e0a040", "#c0b0a0"),
    "大概率": ("#8ab4d8", "#a09080"),
}

LAW_META = {
    "law1_pinyin":  ("铁律一 · 声母定行（拼音）", "拼音声母+韵母 → 音读，直接可用"),
    "law1_mc":      ("铁律一 · 声母定行（中古音）", "MC声母+韵母+韵摄 → 音读，音韵学层面"),
    "law2_pinyin":  ("铁律二 · 韵尾定长短（拼音）", "鼻音韵尾 → 撥音/長音"),
    "law2_mc":      ("铁律二 · 韵尾定长短（中古音）", "MC入声韵尾+声类 → 音读尾部锁定"),
    "law3_shengfu": ("铁律三 · 声符定读音", "形声字声旁 → 字族音读"),
    "law4_rusheng": ("铁律四 · 入声见端", "入声字 ク/キ/ツ/チ 尾部标记"),
    "law5_wuhan":   ("铁律五 · 呉漢分層", "Go-on vs Kan-on 待KANJIDIC2补齐"),
}

CSS = """* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:"PingFang SC","Noto Serif SC","SF Mono","Menlo",monospace; background:#0b0908; color:#c0b0a0; }
.card { padding:40px 44px; }
h2 { font-family:"Noto Serif SC","Songti SC",serif; font-size:32px; font-weight:900; color:#efe4d0; letter-spacing:0.06em; margin-bottom:6px; }
.stats { font-size:16px; color:#6a5a4a; letter-spacing:0.04em; margin-bottom:24px; }
.stats span { margin-right:18px; }
.stats .c1 { color:#e0a040; } .stats .c2 { color:#8ab4d8; }
.section-title { font-family:"Noto Serif SC","Songti SC",serif; font-size:20px; font-weight:700; color:#a09070; letter-spacing:0.05em; margin:18px 0 10px 0; padding-bottom:8px; border-bottom:1px solid rgba(255,255,255,0.04); }
.grid { display:grid; grid-template-columns:1fr 1fr; gap:6px 16px; }
.rule { font-size:15px; line-height:1.55; padding:3px 0; letter-spacing:0.02em; word-break:break-all; }
.rule .acc { font-size:13px; margin-left:4px; }
.rule.c1 .acc { color:#e0a040; } .rule.c2 .acc { color:#8ab4d8; }
.rule.c1 { color:#c0b0a0; } .rule.c2 { color:#a09080; }
.footer { margin-top:20px; font-size:14px; color:#4a3a2a; letter-spacing:0.06em; text-align:right; border-top:1px solid rgba(255,255,255,0.03); padding-top:12px; }
.empty-note { font-size:20px; color:#6a5a4a; line-height:1.8; letter-spacing:0.04em; padding:30px 0; }
.empty-note em { color:#d47a3a; font-style:normal; }
"""

def build_html(cat_key, rules_list):
    title, subtitle = LAW_META[cat_key]

    conf_rank = {c: i for i, c in enumerate(CONF_ORDER)}
    rules_list.sort(key=lambda r: (conf_rank.get(r["confidence"], 99), -r["accuracy"]))

    counts = defaultdict(int)
    for r in rules_list:
        counts[r["confidence"]] += 1

    stats_html = "".join(
        f'<span class="c{i+1}">{c} {counts[c]}</span>'
        for i, c in enumerate(CONF_ORDER)
    )

    groups = defaultdict(list)
    for r in rules_list:
        groups[r["confidence"]].append(r)

    body_parts = []
    for conf in CONF_ORDER:
        grp = groups.get(conf, [])
        if not grp:
            continue
        color_name = f"c{CONF_ORDER.index(conf)+1}"
        body_parts.append(f'<div class="section-title">{conf} ({len(grp)} 条)</div>')
        body_parts.append(f'<div class="grid">')
        for r in grp:
            acc_pct = f'{r["accuracy"]:.0%}' if r["accuracy"] >= 0.01 else f'{r["accuracy"]:.1%}'
            rt = r["rule_text"]
            body_parts.append(
                f'<div class="rule {color_name}">'
                f'{rt}<span class="acc">{acc_pct}</span>'
                f'</div>'
            )
        body_parts.append('</div>')

    body = "\n".join(body_parts)
    total = len(rules_list)

    # Estimate height - rules can wrap now so use taller per-row estimate
    use_grid = total > 24
    if not use_grid:
        body = body.replace('grid-template-columns:1fr 1fr;', 'grid-template-columns:1fr;')
        est_rows = total * 1.5 + sum(1 for c in CONF_ORDER if groups.get(c))
    else:
        est_rows = total * 0.7 + sum(1 for c in CONF_ORDER if groups.get(c))

    est_height = 140 + est_rows * 30 + 50 + 30 * sum(1 for c in CONF_ORDER if groups.get(c))
    height = max(int(est_height) + 40, 360)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<style>
{CSS}
</style>
</head>
<body>
<div class="card">
<h2>{title}</h2>
<div class="stats">{stats_html}</div>
<div class="subtitle" style="font-size:17px;color:#8a7a6a;margin-bottom:6px;letter-spacing:0.04em;">{subtitle}</div>
{body}
<div class="footer">kanji-on · N1 2,179 汉字 · {total} 条规则（确定+大概率）</div>
</div>
</body>
</html>"""
    return html, height

def main():
    generated = []
    for cat_key in ["law1_pinyin", "law1_mc", "law2_pinyin", "law2_mc",
                     "law3_shengfu", "law4_rusheng", "law5_wuhan"]:
        rules_list = cats[cat_key]
        if not rules_list:
            title, subtitle = LAW_META[cat_key]
            html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><style>{CSS}</style></head>
<body>
<div class="card">
<h2>{title}</h2>
<div class="stats"><span class="c2">0 条规则（确定+大概率）</span></div>
<div class="empty-note">
{ '计划引入 <em>KANJIDIC2</em> 的 goon/kanon 标注后补齐。' if 'wuhan' in cat_key else '该类别暂无确定或大概率规则。<br>有时/偶尔规则已过滤。' }
</div>
<div class="footer">kanji-on · 仅保留确定+大概率</div>
</div>
</body>
</html>"""
            height = 380
        else:
            html, height = build_html(cat_key, rules_list)

        fname = f"rules-{cat_key}.html"
        fpath = os.path.join(OUT_DIR, fname)
        with open(fpath, "w") as f:
            f.write(html)
        generated.append((fname, height, len(rules_list)))
        print(f"OK: {fname} ({len(rules_list)} rules, ~{height}px)")

    print("\n# Screenshot entries:")
    for fname, height, n in generated:
        png = fname.replace(".html", ".png")
        print(f'    ("{fname}", 900, {height}),  # {n} rules')

if __name__ == "__main__":
    main()
