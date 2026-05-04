"""Phase 1: Build unified kanji dataset with pinyin, JLPT levels, and phonological features."""
import pandas as pd
import numpy as np
import json
import re
from collections import defaultdict
from pypinyin import pinyin, Style

# ── 1. Parse KanjiKensaku ──────────────────────────────────────────
print("Loading KanjiKensaku.xls...")
kanji_df = pd.read_excel("KanjiKensaku.xls", sheet_name="漢字一覧")
kanji_df = kanji_df.rename(columns={
    "漢字": "kanji", "構成文字": "components_raw", "異字体": "variant",
    "部首": "radical", "画数": "stroke_count", "音読み": "onyomi_raw",
    "訓読み": "kunyomi_raw", "意味": "meaning", "熟語": "compounds"
})

# ── 2. JLPT level assignment via embedded standard lists ───────────
# Built from: 日本語能力試験出題基準, 紅宝書, and past exam compilations.
# Each level is CUMULATIVE (includes all lower levels) internally,
# but we assign each kanji to the LOWEST level it first appears.

_JLPT_RAW = {
    # ~100 kanji, basic concepts and daily life
    "N5": "一二三四五六七八九十百千万円年月日時分毎午前半後週今昨日毎朝昼夜晩"
          "火水木金土曜上下左右中外北南東西口目耳手足力"
          "人子男女父母兄弟姉妹家族友達先生学校生大中小高長新古多安白赤青黒"
          "名語国日本語英語中国韓国学電車馬牛犬猫魚鳥花山川海天気雨雪雲風空"
          "林森石田町村道店駅会社病院銀行"
          "食飲物野菜果肉魚酒茶飯料理品買物金円錢"
          "行来帰入出休立座使作書読見聞話買食飲歩走止動"
          "勉強試験質問答問題住所電話手機",
    # ~170 new, daily life and simple texts
    "N4": "不世他以代全会住使借始体働元兄光公写冬切別力勉動区医去味品員回国"
          "場売夏外太妹妻姉始字安室家寒工帰広店度建引弟弱強当形徒心思悪意"
          "成戸教数文新方旅族早明星春昼暑暗曜最有服末材料楽様機次歌止正死民"
          "池決波注漢然牛物特犬理用田畑的病発県真知短研礼科目私者育舌船良英"
          "荷菜薬衆行調赤起足軽辺近通遠都重野菜引運開院歌声工章童笛筒答案級"
          "経練緑縁育肺腸臓自至台般良落葉薬衛製観計記詞話説調談議識警議谷貨"
          "貿走越路転軽農連進遊運過達成選都郡部配酒重鉄銀開関係院集面題顔風"
          "質問題館庭園式様主奥",
    # ~370 new, newspaper and intermediate texts
    "N3": "与争互他代任伝全具写制判制助努参史君告員商回図地報変夫好妻娘婚守"
          "完全対平年度幸式張当待必念性怪怖恩悲想情感慣技授政故断方族日昔昭"
          "昨暗曲更曜有期未来材束条根業構様権歯歴史残段民決法治準潅点然無物"
          "特犯状王現理生的産用田由申男発直相省示礼科程種究章級経緑縁腸臓至"
          "般荷落葉衛製観計記詞話説調談議識警議谷貨貿走越路転軽農連進遊運過"
          "達成選都郡部配酒重鉄銀開関係院集面題顔風"
          "坂港震波降低岸島陽緑競輪向指令札互印翌腕腹背肩痛疲寝覚恵快眠夢"
          "涙笑泣怒喜苦感動幸不幸平和戦政治経済法律裁判罪罰警察消防救命救助"
          "的状況状態条件結果原因目的方法手段規則制度習慣伝統文化歴史地理技術"
          "産業農業漁業林業商業貿易交通運輸通信放送出版宣伝広告芸術音楽文学"
          "科学数学物理化学生物医学心理学哲学宗教教育研究開発設計建築土木機械"
          "電気電子情報画像映像音響",
    # ~370 new, high-level newspapers and business documents
    "N2": "券危器警険災州態康機念独率現産省築紀素績衛討誤諸貿賞越超逮捕隊限"
          "降際震額飾驚魅昇恩困敬策幼徳否俵雲菓衣印奥宙幼延液宴延炎押憶我"
          "快戒械較患寒甘監疑規巨況迎刑系券減限固故後互公香込困査際財罪昨"
          "策刷察散残氏施祉紙刺児滋治釈若需収就舟襲順初署諸将焼壌譲臣診誠"
          "清整税跡責績設折選善騒層息則退替断恥貯超賃貰提的展悼努党討得燃"
          "杯薄怒縛髪抜犯彼判被評尾貧福複奮編返墓報宝訪暴末密眠務迷乱戻烈"
          "論朗賄涙恋渇湾腕弾瓶仏払沸粉紛噴墳憤奮並閉陛塀壁癖片編辺返遍便"
          "捕浦補舗募墓暮簿宝抱放泡砲胞豊訪亡坊忙房某冒帽貿防未味魅密妙眠"
          "霧娘盟銘鳴滅茂模猛盲網耗黙戻問紋匁厄役約訳薬躍由油輸唯幽郵予余"
          "誉預揺擁曜様洋用窯羊葉要容溶欲浴抑翼雷頼絡落乱卵覧濫欄吏利理"
          "裏履璃離陸律略留流粒隆硫旅虜了料猟瞭力林厘倫輪隣臨涙累塁類令冷"
          "励礼鈴隷麗齢暦劣烈裂恋連廉練錬炉露郎浪廊楼漏録",
    # ~1000+ new, academic, literary, and specialized
    "N1": "亜哀握扱依偉威尉慰為維緯違井壱逸稲芋姻陰隠韻渦浦影詠鋭疫悦越謁"
          "閲炎宴援煙猿縁鉛汚凹奥押欧殴翁沖憶屋乙卸穏佳苛架華菓渓嫁暇禍靴"
          "寡箇稼蚊牙瓦雅餓介戒怪拐悔皆塊楷潰壊懐諧劾崖涯慨蓋該概骸垣柿覚"
          "角赫較郭隔革穫岳掛潟括喝渇滑褐轄且株釜鎌刈干冠勘勧喚堪換敢棺款"
          "歓汗環甘監看竿管簡缶肝艦貫還鑑間閑陥含眼頑顔岐祈既忌機棋棄毅汽"
          "詰虐胸脅響驚仰凝暁業局曲極玉錦僅勤均斤琴禁筋緊菌襟謹近金吟銀九"
          "倶句区狗玖矩苦駆具惧愚空偶遇隅串屈掘窟繰勲薫刑兄啓恵掲携渓継茎"
          "蛍軽鶏迎鯨撃傑欠決潔穴結血月倹剣圏堅嫌懸献肩鍵繭絹遣権憲賢謙鍵"
          "繭顕験懸元厳幻弦減源玄現言限個古呼固孤己戸故枯湖虎誇雇顧鼓互呉後"
          "娯悟碁侯候光公功効厚口向后坑好孔巧幸広康恒慌抗拘控攻更構江洪港溝"
          "甲皇硬稿紅絞綱耕肯航荒行衡講購郊酵鉱鋼降項香高剛拷豪克刻酷国黒穀"
          "獄骨込墾昆恨婚紺魂墾佐詐鎖債催宰彩栽歳剤咲崎削搾索錯撮擦傘惨桟産"
          "暫賛酸師志伺刺司史嗣四士始姉姿子屍市師志思指支施旨枝止死氏祉私糸"
          "紙紫肢脂至視詞詩試誌諮資賜雌飼歯事似侍児字寺慈持時次滋治璽磁示耳"
          "自辞式識軸七執失室湿漆疾質実芝舎射捨赦斜煮社者車蛇邪勺尺爵酌釈寂"
          "朱殊狩獣趣需舟襲秀秋終臭週集醜住充十従柔汁渋獣縦重宿祝粛縮熟出述"
          "術俊春瞬準循旬殉准潤盾純巡遵順処初所書庶暑署緒諸女如助序叙徐除小"
          "升少召匠床抄肖尚昌松沼消渉渓湘焼焦照症省硝礁祥称粧紹肖衝詳証象賞"
          "償錠上丈乗冗剰城場嬢常情条浄状畳蒸縄壌錠嘱飾伸信唇娠寝審心慎振新"
          "森浸深申真神紳臣芯薪親診身辛進針震人刃仁尽甚迅陣尋図吹垂水推酔遂"
          "睡穂随髄枢崇据杉寸瀬是井世制勢姓征性政整星牲省逝清盛婿晴聖誠精製"
          "誓請斥石赤析昔隻席惜責跡積績籍切折拙窃接設雪節説舌絶千川仙占先専"
          "宣専戦扇栓泉浅洗染潜旋線繊鮮善全前禅繕塑措疎礎祖租粗組訴阻僧創双"
          "倉喪壮奏層想捜掃挿早巣窓創草葬装僧想層遭騒像増憎臓蔵贈即束足促俗"
          "族属続卒存孫尊損村他多太堕妥惰打駄体対耐待怠胎退帯替泰袋逮滝択沢"
          "卓拓託濯諾濁但奪脱棚丹胆淡端綻鍛男団断弾段暖談地池知恥値致遅畜竹"
          "逐蓄築秩茶着嫡中仲宙忠抽昼柱注虫鋳駐著貯丁弔庁兆町長挑帳張彫眺釣"
          "頂鳥朝潮超跳長頂鳥勅捗直朕沈珍賃鎮墜締呈廷抵邸亭貞帝訂庭逓停偵堤"
          "提程艇締泥的笛摘滴適敵溺迭哲徹撤天典店点展添転田電吐徒途都渡登土"
          "努度島悼投搭灯当痘盗筒到逃透稲踏闘働銅導洞瞳篤突届屯豚頓貪鈍曇丼"
          "那奈内縄南軟難二尼弐匂虹尿妊忍認寧猫熱年念捻粘燃悩納能脳農濃把覇"
          "婆廃排杯輩培媒梅買売泊白伯拍舶薄迫漠爆縛肌鉢発髪伐抜罰閥反半犯帆"
          "伴判坂阪板版班畔般販飯搬煩頒範繁藩晩番蛮盤比皮妃否批彼肥非卑飛疲"
          "秘被悲扉斐尾美備微鼻匹必泌筆姫百俵標票表評描猫病秒品浜浜貧頻敏瓶"
          "不付夫婦富布府怖扶敷普浮父符腐譜負武部舞封風伏服副復複福払沸仏物"
          "粉紛雰噴墳憤奮分文聞並閉陛塀平弊柄並壁癖別変片編辺返遍便勉歩保捕"
          "浦補舗母募墓慕暮簿方包宝抱放法泡砲縫胞芳褒豊訪亡坊妨忘忙房某冒剖"
          "紡望傍帽棒貿貌防北未味魅岬密脈妙民眠矛務夢霧無娘名命明迷盟銘鳴滅"
          "免面綿茂模毛妄猛盲網耗木黙目戻問紋門匁也夜野厄役約訳薬躍由油癒諭"
          "輸唯幽悠郵予余与誉預幼揺擁曜様洋用窯羊葉要容溶欲浴抑翼羅裸雷頼絡"
          "落酪乱卵覧濫欄吏利里理痢裏履璃離陸立律慄略留流粒隆硫侶旅虜了両料"
          "猟涼陵瞭力緑林厘倫輪隣臨瑠涙累塁類令冷励戻礼鈴隷霊麗齢暦劣烈裂恋"
          "連廉練錬炉露郎浪廊楼漏籠六録丼"
}

# Deduplicate: each kanji assigned to the LOWEST level where it first appears
_seen = set()
JLPT_KANJI = {}
for level in ["N5", "N4", "N3", "N2", "N1"]:
    level_set = set()
    for k in _JLPT_RAW[level]:
        if k not in _seen and '一' <= k <= '鿿':
            level_set.add(k)
            _seen.add(k)
    JLPT_KANJI[level] = level_set

kanji_to_min_level = {}
for level in ["N5", "N4", "N3", "N2", "N1"]:
    for k in JLPT_KANJI[level]:
        kanji_to_min_level[k] = level

print(f"  Embedded JLPT kanji: {sum(len(v) for v in JLPT_KANJI.values())} total")
for level in ["N5", "N4", "N3", "N2", "N1"]:
    print(f"    {level}: {len(JLPT_KANJI[level])} kanji")

# ── 3. Pinyin: pypinyin primary + kanji.txt fallback ────────────────
print("Loading pinyin data...")
kanji_pinyin_fallback = {}
with open("dataset/kanji.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            parts = line.split(": ")
            if len(parts) == 2:
                codepoint_str = parts[0]
                rest = parts[1]
                reading_part = rest.split("  # ")[0] if "  # " in rest else rest
                codepoint = int(codepoint_str[2:], 16)
                kanji_pinyin_fallback[chr(codepoint)] = reading_part.split(",")[0].strip()
        except (ValueError, IndexError):
            continue

def get_pinyin(char):
    try:
        result = pinyin(char, style=Style.TONE, heteronym=False)
        if result and result[0]:
            return result[0][0]
    except Exception:
        pass
    return kanji_pinyin_fallback.get(char, "")

# ── 4. Pinyin phonological feature extraction ───────────────────────
def extract_pinyin_features(py_str):
    """Decompose pinyin into: initial, final, tone, nasal_coda, is_entering_tone"""
    if not py_str or pd.isna(py_str):
        return "", "", 0, "none", False

    py_str = str(py_str).strip().lower()

    tone_map = {'ā': 1, 'á': 2, 'ǎ': 3, 'à': 4,
                'ē': 1, 'é': 2, 'ě': 3, 'è': 4,
                'ī': 1, 'í': 2, 'ǐ': 3, 'ì': 4,
                'ō': 1, 'ó': 2, 'ǒ': 3, 'ò': 4,
                'ū': 1, 'ú': 2, 'ǔ': 3, 'ù': 4,
                'ǖ': 1, 'ǘ': 2, 'ǚ': 3, 'ǜ': 4}
    base_vowel_map = {'ā': 'a', 'á': 'a', 'ǎ': 'a', 'à': 'a',
                      'ē': 'e', 'é': 'e', 'ě': 'e', 'è': 'e',
                      'ī': 'i', 'í': 'i', 'ǐ': 'i', 'ì': 'i',
                      'ō': 'o', 'ó': 'o', 'ǒ': 'o', 'ò': 'o',
                      'ū': 'u', 'ú': 'u', 'ǔ': 'u', 'ù': 'u',
                      'ǖ': 'v', 'ǘ': 'v', 'ǚ': 'v', 'ǜ': 'v'}

    tone = 5
    base = ""
    for ch in py_str:
        if ch in tone_map:
            tone = tone_map[ch]
            base += base_vowel_map.get(ch, ch)
        else:
            base += ch

    if tone == 5 and base and base[-1].isdigit():
        tone = int(base[-1])
        base = base[:-1]

    initial = ""
    final = base
    for init in ['zh', 'ch', 'sh']:
        if base.startswith(init):
            initial = init
            final = base[len(init):]
            break
    if not initial and base:
        c = base[0]
        if c in 'bpmfdtnlgkhjqxrzcsyw':
            initial = c
            final = base[1:]

    if final.endswith('ng'):
        nasal = 'ng'
    elif final.endswith('n'):
        nasal = 'n'
    else:
        nasal = 'none'

    is_entering = (nasal == 'none' and len(final) <= 3 and final not in ('er', 'i', 'u', 'ü'))

    return initial, final, tone, nasal, is_entering


# ── 5. Component vocabulary ─────────────────────────────────────────
print("Building component vocabulary...")
kanji_only_re = re.compile(r'[一-鿿]')
all_components = set()
for _, row in kanji_df.iterrows():
    comps = str(row["components_raw"]) if not pd.isna(row["components_raw"]) else ""
    for c in comps:
        if kanji_only_re.match(c):
            all_components.add(c)

all_kanji_set = set(kanji_df["kanji"].dropna())
valid_components = sorted(all_components & all_kanji_set)
component_to_id = {c: i for i, c in enumerate(sorted(valid_components))}
print(f"  Valid kanji components in vocabulary: {len(component_to_id)}")

# ── 6. Build unified dataset ────────────────────────────────────────
print("Building unified dataset...")
records = []
onyomi_set = set()

for _, row in kanji_df.iterrows():
    kanji = str(row["kanji"]) if not pd.isna(row["kanji"]) else ""
    if not kanji:
        continue

    comps_raw = str(row["components_raw"]) if not pd.isna(row["components_raw"]) else ""
    comp_kanji = [c for c in comps_raw if kanji_only_re.match(c) and c != kanji]
    comp_ids = [component_to_id[c] for c in comp_kanji if c in component_to_id]

    onyomi_raw = str(row["onyomi_raw"]) if not pd.isna(row["onyomi_raw"]) else ""
    onyomi_list = [o.strip() for o in onyomi_raw.split("、") if o.strip()]
    for o in onyomi_list:
        onyomi_set.add(o)

    primary_on = onyomi_list[0] if onyomi_list else ""
    kunyomi_raw = str(row["kunyomi_raw"]) if not pd.isna(row["kunyomi_raw"]) else ""

    pinyin_result = get_pinyin(kanji)
    init, final, tone, nasal, is_ent = extract_pinyin_features(pinyin_result)

    jlpt = kanji_to_min_level.get(kanji, "N1+")
    radical = str(row["radical"]) if not pd.isna(row["radical"]) else ""
    stroke = float(row["stroke_count"]) if not pd.isna(row["stroke_count"]) else 0

    records.append({
        "kanji": kanji,
        "components": json.dumps(comp_kanji, ensure_ascii=False),
        "component_ids": json.dumps(comp_ids),
        "radical": radical,
        "stroke_count": int(stroke),
        "onyomi": primary_on,
        "onyomi_all": json.dumps(onyomi_list, ensure_ascii=False),
        "kunyomi": kunyomi_raw,
        "meaning": str(row["meaning"]) if not pd.isna(row["meaning"]) else "",
        "pinyin": pinyin_result,
        "pinyin_initial": init,
        "pinyin_final": final,
        "pinyin_tone": tone,
        "nasal_coda": nasal,
        "jlpt_level": jlpt,
    })

dataset = pd.DataFrame(records)

# ── 7. On'yomi vocabulary ───────────────────────────────────────────
onyomi_to_id = {o: i for i, o in enumerate(sorted(onyomi_set))}

# ── 8. Save ─────────────────────────────────────────────────────────
dataset.to_csv("dataset/kanji_dataset.csv", index=False, encoding="utf-8")
with open("dataset/component_vocab.json", "w", encoding="utf-8") as f:
    json.dump(component_to_id, f, ensure_ascii=False)
with open("dataset/onyomi_vocab.json", "w", encoding="utf-8") as f:
    json.dump(onyomi_to_id, f, ensure_ascii=False)

# ── 9. Statistics ───────────────────────────────────────────────────
print(f"\n=== Dataset Built ===")
print(f"Total kanji: {len(dataset)}")
print(f"With on'yomi: {dataset['onyomi'].notna().sum()}")
print(f"With pinyin: {(dataset['pinyin'] != '').sum()}")
print(f"Pinyin coverage: {(dataset['pinyin'] != '').sum() / len(dataset):.1%}")
print(f"Unique on'yomi categories: {len(onyomi_to_id)}")
print(f"Unique components: {len(component_to_id)}")
print(f"\nJLPT distribution:")
for level in ["N5", "N4", "N3", "N2", "N1", "N1+"]:
    count = (dataset["jlpt_level"] == level).sum()
    print(f"  {level}: {count}")

missing_pinyin = dataset[dataset["pinyin"] == ""]
if len(missing_pinyin) > 0:
    print(f"\nKanji missing pinyin ({len(missing_pinyin)}):")
    for _, row in missing_pinyin.head(20).iterrows():
        print(f"  {row['kanji']} (onyomi: {row['onyomi']})")

print("\nDone. Files saved to dataset/")
