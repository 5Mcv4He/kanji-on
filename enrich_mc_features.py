"""Add Middle Chinese features from 廣韻 to JLPT datasets.

Stores ALL MC readings per kanji (not just primary), adds Go-on/Kan-on
inference, and keeps primary-reading columns for backward compatibility.
"""

import pandas as pd
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from predict_go_kan import infer_reading_types, get_reading_type_label, predict_reflex_pattern

# Load MC map (all readings per kanji)
with open(f"{HERE}/output/mc_reading_map.json") as f:
    mc_map = json.load(f)

# Load shinjitai -> kyujitai mapping
with open(f"{HERE}/output/shin_kyu_map.json") as f:
    SHIN_TO_KYU = json.load(f)

LEVELS = ["N5", "N4", "N3", "N2", "N1"]

for lv in LEVELS:
    fbase = lv.lower()
    ds_path = f"{HERE}/dataset/{fbase}_dataset.csv"
    df = pd.read_csv(ds_path)

    n_total = len(df)

    # ── New columns ──

    # Primary MC columns (single reading, for backward compat)
    mc_cols = {
        'mc_initial': '',
        'mc_initial_cat': '',
        'mc_voicing': '',
        'mc_rhyme': '',
        'mc_rhyme_cat': '',
        'mc_grade': '',
        'mc_openness': '',
        'mc_tone': '',
        'mc_is_entering': 0,
        'mc_entering_coda': '',
    }

    # Multi-reading and Go/Kan columns
    extra_cols = {
        'mc_readings_json': '',           # JSON array of ALL MC readings
        'mc_reading_count': 0,            # How many MC readings this kanji has
        'mc_reading_source': '',          # 'gy' (廣韻), 'wy' (王一), 'multi', or ''
        'mc_reading_types': '',           # JSON: inferred reading types per MC reading
        'mc_goon_primary': '',            # Primary MC reading tagged as go-on (if distinct)
        'mc_kanon_primary': '',           # Primary MC reading tagged as kan-on (if distinct)
        'mc_divergent': 0,                # 1 if kanji has divergent Go/Kan readings
    }

    all_new_cols = {**mc_cols, **extra_cols}

    for col, default in all_new_cols.items():
        if col not in df.columns:
            df[col] = default

    matched = 0
    multi = 0
    divergent = 0

    for idx, row in df.iterrows():
        kanji = str(row['kanji'])

        # Try direct lookup, then kyujitai form
        readings = mc_map.get(kanji)
        if not readings:
            kyu = SHIN_TO_KYU.get(kanji)
            if kyu:
                readings = mc_map.get(kyu)

        if not readings:
            continue

        # ── Infer reading types for all MC readings ──
        typed_readings = infer_reading_types(readings)

        # ── Store ALL readings as JSON ──
        df.at[idx, 'mc_readings_json'] = json.dumps(typed_readings, ensure_ascii=False)
        df.at[idx, 'mc_reading_count'] = len(readings)
        if len(readings) > 1:
            multi += 1

        # ── Reading source ──
        sources = set(r.get('source', '') for r in readings)
        has_gy = any('廣韻' in s for s in sources)
        has_wy = any('王一' in s for s in sources)
        if has_gy and has_wy:
            df.at[idx, 'mc_reading_source'] = 'multi'
        elif has_gy:
            df.at[idx, 'mc_reading_source'] = 'gy'
        elif has_wy:
            df.at[idx, 'mc_reading_source'] = 'wy'

        # ── Select primary reading (prefer 廣韻, then 平聲) ──
        primary = readings[0]
        if len(readings) > 1:
            gy = [r for r in readings if '廣韻' in r.get('source', '')]
            primary = gy[0] if gy else readings[0]

        # ── Populate primary MC columns (backward compat) ──
        df.at[idx, 'mc_initial'] = primary['initial']
        df.at[idx, 'mc_initial_cat'] = primary['initial_category']
        df.at[idx, 'mc_voicing'] = primary['voicing']
        df.at[idx, 'mc_rhyme'] = primary['rhyme']
        df.at[idx, 'mc_rhyme_cat'] = primary['rhyme_category']
        df.at[idx, 'mc_grade'] = primary['grade'] or ''
        df.at[idx, 'mc_openness'] = primary['openness'] or ''
        df.at[idx, 'mc_tone'] = primary['tone'] or ''
        df.at[idx, 'mc_is_entering'] = 1 if primary['is_entering'] else 0
        df.at[idx, 'mc_entering_coda'] = primary.get('entering_coda_type', '') or ''

        # ── Go-on / Kan-on annotation ──
        reading_types = [get_reading_type_label(r) for r in readings]
        df.at[idx, 'mc_reading_types'] = json.dumps(reading_types, ensure_ascii=False)

        # Find readings identified as go-on or kan-on leaning
        goon_readings = []
        kanon_readings = []
        for tr in typed_readings:
            rt = tr.get('inferred_reading_type', 'general')
            if rt in ('go-on', 'divergent'):
                goon_readings.append(tr)
            if rt in ('kan-on', 'divergent'):
                kanon_readings.append(tr)

        if goon_readings:
            # Store primary go-on MC feature summary
            g = goon_readings[0]
            df.at[idx, 'mc_goon_primary'] = json.dumps({
                'initial': g['initial'], 'rhyme': g['rhyme'],
                'tone': g['tone'], 'grade': g.get('grade', ''),
                'openness': g.get('openness', ''),
            }, ensure_ascii=False)

        if kanon_readings:
            k = kanon_readings[0]
            df.at[idx, 'mc_kanon_primary'] = json.dumps({
                'initial': k['initial'], 'rhyme': k['rhyme'],
                'tone': k['tone'], 'grade': k.get('grade', ''),
                'openness': k.get('openness', ''),
            }, ensure_ascii=False)

        if goon_readings and kanon_readings:
            df.at[idx, 'mc_divergent'] = 1
            divergent += 1

        matched += 1

    df.to_csv(ds_path, index=False)
    print(f"{lv}: {matched}/{n_total} ({matched/n_total*100:.1f}%) MC mapped, "
          f"{multi} multi-MC, {divergent} 呉/漢異")

print("\nDone. MC features (all readings + Go/Kan) added to all datasets.")
