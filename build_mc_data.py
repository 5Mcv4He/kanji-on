"""Build Middle Chinese (中古音) feature mapping from 廣韻 rhyme_book.csv.
Parses the qieyun library's rhyme_book.csv → kanji→MC features dict.

最簡描述 format: 聲母+開合+等+韻+聲
Example: 從開三支平 → 聲母=從, 開合=開, 等=三, 韻=支, 聲=平
"""

import csv
import json
import os
import re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

# ── MC category constants (from QieyunEncoder/常量.py) ──

ALL_INITIALS = '幫滂並明端透定泥來知徹澄孃精清從心邪莊初崇生俟章昌常書船日見溪羣疑影曉匣云以'
ALL_RHYMES = '東冬鍾江支脂之微魚虞模齊祭泰佳皆夬灰咍廢眞臻文欣元魂痕寒刪山仙先蕭宵肴豪歌麻陽唐庚耕清青蒸登尤侯幽侵覃談鹽添咸銜嚴凡'
ALL_TONES = '平上去入'
ALL_GRADES = '一二三四'
ALL_OPENNESS = '開合'

# Voicing categories
VOICING_MAP = {
    # 脣音 (labial)
    '幫': '全清', '滂': '次清', '並': '全濁', '明': '次濁',
    # 舌音 (lingual)
    '端': '全清', '透': '次清', '定': '全濁', '泥': '次濁', '來': '次濁',
    '知': '全清', '徹': '次清', '澄': '全濁', '孃': '次濁',
    # 齒音 (dental/sibilant)
    '精': '全清', '清': '次清', '從': '全濁', '心': '全清', '邪': '全濁',
    '莊': '全清', '初': '次清', '崇': '全濁', '生': '全清', '俟': '全濁',
    '章': '全清', '昌': '次清', '常': '全濁', '書': '全清', '船': '全濁',
    '日': '次濁',
    # 牙音 (velar)
    '見': '全清', '溪': '次清', '羣': '全濁', '疑': '次濁',
    # 喉音 (laryngeal)
    '影': '全清', '曉': '全清', '匣': '全濁', '云': '次濁', '以': '次濁',
}

# Initial categories (for grouping)
INITIAL_CATEGORY = {}
for c in '幫滂並明': INITIAL_CATEGORY[c] = '脣'
for c in '端透定泥': INITIAL_CATEGORY[c] = '舌'
for c in '來': INITIAL_CATEGORY[c] = '舌'
for c in '知徹澄孃': INITIAL_CATEGORY[c] = '舌'
for c in '精清從心邪': INITIAL_CATEGORY[c] = '齒'
for c in '莊初崇生俟': INITIAL_CATEGORY[c] = '齒'
for c in '章昌常書船': INITIAL_CATEGORY[c] = '齒'
for c in '日': INITIAL_CATEGORY[c] = '齒'
for c in '見溪羣疑': INITIAL_CATEGORY[c] = '牙'
for c in '影曉匣云以': INITIAL_CATEGORY[c] = '喉'

# Rhyme → expected Japanese on'yomi pattern (for Go-on and Kan-on)
# This is a known mapping from MC rhyme groups to Japanese readings
RHYME_CATEGORY = {
    '東': '通', '冬': '通', '鍾': '通',
    '江': '江',
    '支': '止', '脂': '止', '之': '止', '微': '止',
    '魚': '遇', '虞': '遇', '模': '遇',
    '齊': '蟹', '祭': '蟹', '泰': '蟹', '佳': '蟹', '皆': '蟹', '夬': '蟹', '灰': '蟹', '咍': '蟹', '廢': '蟹',
    '眞': '臻', '臻': '臻', '文': '臻', '欣': '臻', '元': '臻', '魂': '臻', '痕': '臻',
    '寒': '山', '刪': '山', '山': '山', '仙': '山', '先': '山',
    '蕭': '效', '宵': '效', '肴': '效', '豪': '效',
    '歌': '果', '麻': '假',
    '陽': '宕', '唐': '宕',
    '庚': '梗', '耕': '梗', '清': '梗', '青': '梗',
    '蒸': '曾', '登': '曾',
    '尤': '流', '侯': '流', '幽': '流',
    '侵': '深',
    '覃': '咸', '談': '咸', '鹽': '咸', '添': '咸', '咸': '咸', '銜': '咸', '嚴': '咸', '凡': '咸',
}

# MC tone → Japanese entering tone coda mapping
# 入聲 (entering tone) has specific coda consonants in MC that map to Japanese
ENTERING_CODA_MAP = {
    # 通攝: -k → ク/キ
    '東': 'ク', '冬': 'ク', '鍾': 'ク',
    # 江攝: -k → ク
    '江': 'ク',
    # 臻攝: -t → ツ/チ
    '眞': 'ツ', '臻': 'ツ', '文': 'ツ', '欣': 'ツ', '元': 'ツ', '魂': 'ツ', '痕': 'ツ',
    # 山攝: -t → ツ/チ
    '寒': 'ツ', '刪': 'ツ', '山': 'ツ', '仙': 'ツ', '先': 'ツ',
    # 宕攝: -k → ク/キ
    '陽': 'ク', '唐': 'ク',
    # 梗攝: -k → ク/キ
    '庚': 'ク', '耕': 'ク', '清': 'ク', '青': 'ク',
    # 曾攝: -k → ク/キ
    '蒸': 'ク', '登': 'ク',
    # 深攝: -p → フ/ウ
    '侵': 'フ',
    # 咸攝: -p → フ/ウ
    '覃': 'フ', '談': 'フ', '鹽': 'フ', '添': 'フ', '咸': 'フ', '銜': 'フ', '嚴': 'フ', '凡': 'フ',
}


def parse_description(desc):
    """Parse a 最簡描述 string like '從開三支平' into structured MC features.

    Returns dict with: initial, openness, grade, rhyme, tone,
        initial_category, voicing, rhyme_category, entering_coda
    """
    if not desc or desc == 'nan':
        return None

    desc = desc.strip()

    # Parse by character categories
    # Pattern: 聲母(1-2 chars) + 開合(0-1) + 等(0-1) + 韻(1-2) + 聲(1)

    # Find initial consonant (always 1 character from the set)
    initial = None
    for c in ALL_INITIALS:
        if desc.startswith(c):
            initial = c
            break

    if not initial:
        return None

    rest = desc[len(initial):]

    # Check for openness
    openness = None  # None = 開合中立
    if rest and rest[0] == '開':
        openness = '開'
        rest = rest[1:]
    elif rest and rest[0] == '合':
        openness = '合'
        rest = rest[1:]

    # Check for grade
    grade = None
    if rest and rest[0] in ALL_GRADES:
        grade = rest[0]
        rest = rest[1:]

    # Check for rhyme (longest match first, max 2 chars)
    rhyme = None
    rhymes_by_len = sorted(ALL_RHYMES, key=len, reverse=True)
    for r in rhymes_by_len:
        if rest.startswith(r):
            rhyme = r
            rest = rest[len(r):]
            break

    if not rhyme:
        return None

    # Remaining is tone
    tone = rest if rest in ALL_TONES else None

    result = {
        'initial': initial,
        'initial_category': INITIAL_CATEGORY.get(initial, '?'),
        'voicing': VOICING_MAP.get(initial, '?'),
        'openness': openness,
        'grade': grade,
        'rhyme': rhyme,
        'rhyme_category': RHYME_CATEGORY.get(rhyme, '?'),
        'tone': tone,
        'is_entering': tone == '入',
    }

    # Determine entering tone coda
    if tone == '入' and rhyme in ENTERING_CODA_MAP:
        result['entering_coda_type'] = ENTERING_CODA_MAP[rhyme]
    else:
        result['entering_coda_type'] = None

    return result


def build_mc_mapping():
    """Parse rhyme_book.csv → kanji→MC features mapping.

    Handles multiple readings per kanji (polysemy).
    """
    csv_path = '/opt/homebrew/lib/python3.14/site-packages/qieyun/rhyme_book.csv'

    # kanji → list of MC feature dicts
    mc_map = defaultdict(list)

    with open(csv_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 字頭覈校前 is the primary character field; 字頭 is often empty
            char = row.get('字頭覈校前', '').strip()
            if not char:
                char = row.get('字頭', '').strip()
            desc = row.get('最簡描述', '').strip()

            if not char or not desc or len(char) != 1:
                continue

            features = parse_description(desc)
            if features:
                # Add source info
                features['source'] = row.get('資料名稱', '')
                features['fanqie'] = row.get('反切', '')

                # Avoid duplicates (same source + same description)
                exists = False
                for existing in mc_map[char]:
                    if (existing.get('initial') == features['initial'] and
                        existing.get('rhyme') == features['rhyme'] and
                        existing.get('tone') == features['tone']):
                        exists = True
                        break
                if not exists:
                    mc_map[char].append(features)

    return dict(mc_map)


def select_primary_mc(mc_readings):
    """For kanji with multiple MC readings, select the most relevant one.

    Priority:
    1. 廣韻 (Guangyun) source preferred over 王一 (Wang Yi)
    2. Most common tone: 平 > 去 > 上 > 入
    3. If still multiple, first one
    """
    if not mc_readings:
        return None

    if len(mc_readings) == 1:
        return mc_readings[0]

    # Prefer 廣韻 over 王一
    gy = [r for r in mc_readings if '廣韻' in r.get('source', '')]
    candidates = gy if gy else mc_readings

    # Prefer 平聲 as it's usually the primary reading
    tone_order = {'平': 0, '去': 1, '上': 2, '入': 3}
    candidates.sort(key=lambda r: tone_order.get(r.get('tone', ''), 9))

    return candidates[0]


if __name__ == '__main__':
    print("Building MC mapping from 廣韻 data...")
    mc_map = build_mc_mapping()
    print(f"  Mapped {len(mc_map)} unique characters to MC readings")

    # Stats
    multi = sum(1 for v in mc_map.values() if len(v) > 1)
    print(f"  Characters with multiple readings: {multi}")

    # Test examples
    test_chars = ['東', '學', '神', '日', '本', '中', '大', '小', '人', '一',
                  '漢', '字', '読', '見', '聞', '話', '語', '書', '食', '飲']

    print("\n  Examples:")
    for c in test_chars:
        if c in mc_map:
            primary = select_primary_mc(mc_map[c])
            num = len(mc_map[c])
            extras = f" (+{num-1} more)" if num > 1 else ""
            if primary:
                desc = f"{primary['initial']}{primary['openness'] or ''}{primary['grade'] or ''}{primary['rhyme']}{primary['tone']}"
                print(f"    {c}: {desc}  [{primary['initial_category']}音 {primary['voicing']} {primary['rhyme_category']}攝{extras}]")
        else:
            print(f"    {c}: NOT FOUND")

    # Save
    out_path = os.path.join(HERE, 'output', 'mc_reading_map.json')

    # Convert to serializable format (keep all readings)
    serializable = {}
    for kanji, readings in mc_map.items():
        serializable[kanji] = []
        for r in readings:
            serializable[kanji].append({
                'initial': r['initial'],
                'initial_category': r['initial_category'],
                'voicing': r['voicing'],
                'openness': r['openness'],
                'grade': r['grade'],
                'rhyme': r['rhyme'],
                'rhyme_category': r['rhyme_category'],
                'tone': r['tone'],
                'is_entering': r['is_entering'],
                'entering_coda_type': r['entering_coda_type'],
                'source': r['source'],
                'fanqie': r['fanqie'],
            })

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {out_path}")
    print(f"File size: {os.path.getsize(out_path) / 1024:.0f} KB")
