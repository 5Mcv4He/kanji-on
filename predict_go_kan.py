"""Go-on (呉音) vs Kan-on (漢音) inference from MC features.

Key phonological contrasts between the two historical layers:
  呉音 (Go-on): 5-6th century, via Korean peninsula (Baekje). Older.
  漢音 (Kan-on): 7-8th century, via Tang envoys. Newer. Became standard.

When MC features imply DIFFERENT reflexes, we can tag rules by reading type.
When both layers agree, the rule is "general".
"""

# ── Key Go-on vs Kan-on contrasts ──

# 1. 全濁 initials: Go-on preserves voicing, Kan-on devoices
#    This is the single most reliable contrast.
VOICED_INITIALS = set('並定澄羣從邪崇俟常船匣')

# 2. 次濁 明母 (m-): Go-on → マ行, Kan-on → バ行
MIN_MOTHER = '明'

# 3. 次濁 泥(n-)/娘(nr-): Go-on → ナ行, Kan-on → ダ行
NI_MOTHERS = set('泥孃')

# 4. 次濁 日母 (ny-): Go-on → ニャ行, Kan-on → ザ行
RI_MOTHER = '日'

# 5. 疑母 (ng-): Go-on preserves ガ行, Kan-on often drops or stays
GI_MOTHER = '疑'

# 6. Rhyme category effects
#    梗攝: Go-on -ャウ, Kan-on -エイ
#    宕攝: Go-on -アウ(→オウ), Kan-on -ヤウ(→ヨウ) — complex, varies
KENG_RHYMES = set('庚耕清青')  # 梗攝
TOU_RHYMES = set('陽唐')       # 宕攝

# 7. 蟹攝 specific: Go-on -エ, Kan-on -アイ
KAI_RHYMES = set('齊祭泰佳皆夬灰咍廢')

# 8. 止攝 specific: Go-on -イ, Kan-on -イ (often same, sometimes differ)


def infer_reading_types(mc_readings):
    """For a list of MC readings, infer which produce Go-on vs Kan-on reflexes.

    Returns list of dicts with added 'reading_type' field:
      'go-on' / 'kan-on' / 'general' / 'uncertain'
    """
    results = []
    for r in mc_readings:
        rt = _infer_single(r)
        r_copy = dict(r)
        r_copy['inferred_reading_type'] = rt
        results.append(r_copy)
    return results


def _infer_single(mc):
    """Infer likely reading type for a single MC reading.

    Uses the most reliable contrasts. Many MC readings map to the same
    Japanese reflex in both Go-on and Kan-on — these are 'general'.
    """
    initial = mc.get('initial', '')
    voicing = mc.get('voicing', '')
    rhyme = mc.get('rhyme', '')
    rhyme_cat = mc.get('rhyme_category', '')
    tone = mc.get('tone', '')

    signals = {'go': 0, 'kan': 0}

    # ── Initial-based signals ──

    # 全濁 → Go-on voiced, Kan-on voiceless = DIFFERENT reflexes
    if initial in VOICED_INITIALS:
        # The reflex differs: Go-on uses voiced kana, Kan-on uses voiceless
        signals['go'] += 2
        signals['kan'] += 2  # Both layers valid, but different readings

    # 明母 → Go-on マ行, Kan-on バ行
    if initial == MIN_MOTHER:
        signals['go'] += 1
        signals['kan'] += 1

    # 泥/娘母 → Go-on ナ行, Kan-on ダ行
    if initial in NI_MOTHERS:
        signals['go'] += 1
        signals['kan'] += 1

    # 日母 → Go-on ニャ/ニ, Kan-on ザ/ジ
    if initial == RI_MOTHER:
        signals['go'] += 1
        signals['kan'] += 1

    # ── Rhyme-based signals ──

    # 梗攝: Go-on -ャウ, Kan-on -エイ (when not entering tone)
    if rhyme in KENG_RHYMES and tone != '入':
        signals['go'] += 1
        signals['kan'] += 1

    # ── Determine type ──

    if signals['go'] > 0 and signals['kan'] > 0:
        # Features where Go and Kan diverge → mark as having distinct layers
        return 'divergent'  # Both exist, but give different readings
    elif signals['go'] > signals['kan']:
        return 'go-on'
    elif signals['kan'] > signals['go']:
        return 'kan-on'
    else:
        return 'general'


# ── Predicted reflex patterns ──

def predict_reflex_pattern(mc):
    """Predict the Japanese reflex PATTERN (not exact kana) for Go-on and Kan-on.

    Returns dict with goon_pattern and kanon_pattern strings describing
    the expected phonological behavior.
    """
    initial = mc.get('initial', '')
    voicing = mc.get('voicing', '')
    rhyme = mc.get('rhyme', '')
    rhyme_cat = mc.get('rhyme_category', '')
    tone = mc.get('tone', '')

    patterns = {
        'goon_voicing': 'preserved',
        'kanon_voicing': 'preserved',
        'goon_initial_type': '',
        'kanon_initial_type': '',
        'notes': [],
    }

    # Voicing contrast
    if voicing == '全濁':
        patterns['goon_voicing'] = 'voiced'
        patterns['kanon_voicing'] = 'voiceless'
        patterns['notes'].append('全濁→呉音濁/漢音清')

    # 明母 special
    if initial == '明':
        patterns['goon_initial_type'] = 'マ行'
        patterns['kanon_initial_type'] = 'バ行'
        patterns['notes'].append('明母→呉マ行/漢バ行')

    # 日母 special
    if initial == '日':
        patterns['goon_initial_type'] = 'ナ/ニャ行'
        patterns['kanon_initial_type'] = 'ザ/ジャ行'
        patterns['notes'].append('日母→呉ナ行/漢ザ行')

    # 泥/娘母
    if initial in ('泥', '孃'):
        patterns['goon_initial_type'] = 'ナ行'
        patterns['kanon_initial_type'] = 'ダ行'
        patterns['notes'].append('泥母→呉ナ行/漢ダ行')

    # 梗攝 vowel contrast
    if rhyme in KENG_RHYMES and tone != '入':
        patterns['notes'].append('梗攝→呉ャウ/漢エイ')

    return patterns


def get_reading_type_label(mc):
    """Get a concise reading type label for display."""
    rt = _infer_single(mc)
    labels = {
        'go-on': '呉音',
        'kan-on': '漢音',
        'divergent': '呉/漢異',
        'general': '通用',
        'uncertain': '不明',
    }
    return labels.get(rt, rt)


# ── For rule generation: given MC features, does this rule apply to
#    Go-on specifically, Kan-on specifically, or both? ──

def classify_rule_reading_type(mc_features_dict):
    """Classify whether a rule (MC features → onyomi) is Go-on or Kan-on.

    Used in tiered_rules.py to annotate rules.
    Returns one of: 'go-on', 'kan-on', 'divergent', 'general'
    """
    return _infer_single(mc_features_dict)
