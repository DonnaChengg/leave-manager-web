# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import List, Dict, Tuple, Optional
from io import BytesIO
from PIL import Image
import re

# ========================
# OCR backend: RapidOCR (ONNXRuntime)
# ========================
@st.cache_resource(show_spinner=True)
def load_ocr_reader():
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR()  # CPU、純 pip，雲端友善

def ocr_image(reader, file) -> List[str]:
    """回傳影像文字行（信心 >= 0.35），做常見正規化。"""
    img = Image.open(file).convert("RGB")
    arr = np.array(img)
    try:
        result, _ = reader(arr)   # result: list of [box, text, score]
    except Exception:
        result = []
    lines = []
    for item in (result or []):
        if not item or len(item) < 3:
            continue
        text = str(item[1]); score = float(item[2] or 0.0)
        if score >= 0.35:
            text = norm(text)
            if text:
                lines.append(text)
    return lines

# ========================
# Utils
# ========================
FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")
CHECK_MARKS = ["✓", "✔", "✅", "☑", "■", "□", "V", "v", "勾", "√"]

def norm(s: str) -> str:
    if s is None: return ""
    s = str(s).translate(FULLWIDTH_DIGITS)
    s = s.replace("\u3000", " ").replace("／", "/")
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s

def has_check(s: str) -> bool:
    return any(m in s for m in CHECK_MARKS)

def find_time(s: str) -> str:
    """
    擷取時間：09:05、9:5、9：05、9時05分 → 標準 HH:MM
    """
    x = s.replace("：", ":")
    m = re.search(r"(\d{1,2})\s*[時点點]\s*(\d{1,2})\s*分?", x)
    if m:
        hh = int(m.group(1)); mm = int(m.group(2))
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return f"{hh:02d}:{mm:02d}"
    m2 = re.search(r"\b(\d{1,2})\s*:\s*(\d{1,2})\b", x)
    if m2:
        hh = int(m2.group(1)); mm = int(m2.group(2))
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return f"{hh:02d}:{mm:02d}"
    return ""

def roc_to_date(roc_str: str, fallback_roc_year: Optional[int]=None) -> Optional[date]:
    """
    支援 114/10/31、10/31（補年）、114年10月31日 / 10月31日
    """
    s = norm(roc_str).replace("年", "/").replace("月", "/").replace("日", "")
    s = re.sub(r"[.\-]", "/", s)

    m = re.search(r"(\d{2,3})\s*/\s*(\d{1,2})\s*/\s*(\d{1,2})", s)
    if m:
        y, mo, da = map(int, m.groups())
        try:
            return date(y + 1911, mo, da)
        except Exception:
            return None

    m2 = re.search(r"(\d{1,2})\s*/\s*(\d{1,2})", s)
    if m2 and fallback_roc_year:
        mo, da = map(int, m2.groups())
        try:
            return date(fallback_roc_year + 1911, mo, da)
        except Exception:
            return None

    return None

def date_to_roc(d: date) -> str:
    return f"{d.year - 1911}/{d.month:02d}/{d.day:02d}"

def parse_mapping_txt(txt: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """name_id_clean.txt → (id2name, name2id)"""
    id2name, name2id = {}, {}
    for ln in txt.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"): continue
        parts = re.split(r"[\t,，\s]+", ln)
        if len(parts) < 2: continue
        a, b = parts[0], parts[1]
        if re.fullmatch(r"\d{7}", a) and not re.fullmatch(r"\d{7}", b):
            sid, name = a, b
        elif re.fullmatch(r"\d{7}", b) and not re.fullmatch(r"\d{7}", a):
            sid, name = b, a
        else:
            sid, name = (a, b) if re.search(r"\d", a) else (b, a)
        sid, name = norm(sid), norm(name)
        if re.fullmatch(r"\d{7}", sid) and name:
            id2name[sid] = name
            name2id[name] = sid
    return id2name, name2id

# ---- 假別（可擴充）----
# 你的預設：病、事、特別、公、論（可在 sidebar 再加自訂）
DEFAULT_TYPE_PATTERNS = {
    "病":   [r"病"],
    "事":   [r"事"],
    "特別": [r"特別", r"特休", r"特(?!警|殊)"],  # 避免誤抓非假別
    "公":   [r"公(假|出)?"],
    "論":   [r"論"],  # 論文/論出/論入
}

def detect_leave_type(text: str, extra_keywords: list[str]) -> str:
    """回傳偵測到的假別；優先你的類別，再試使用者自訂，最後抓通用「XX假」或常見詞。"""
    T = norm(text)

    for canon, regs in DEFAULT_TYPE_PATTERNS.items():
        for rgx in regs:
            if re.search(rgx, T):
                return canon

    for kw in extra_keywords:
        if kw and kw in T:
            return kw

    m = re.search(r"([\u4e00-\u9fa5]{1,4})\s*假", T)
    if m:
        return m.group(1)

    m2 = re.search(r"(喪|婚|產|補|慰|病|事|公|特|論|兵役)", T)
    if m2:
        return m2.group(1)

    return ""

def parse_sign_lines(
    lines: List[str],
    id2name: Dict[str,str],
    fallback_roc_year: int,
    source_label: str,
    extra_type_keywords: list[str]
) -> pd.DataFrame:
    """
    將「隊部/大門」簽出入照片 OCR 行轉為結構資料：
    欄位：sid, name, date(ROC), dir(出/入), time(HH:MM), type, source, raw
    """
    rows = []
    for ln in lines:
        L = norm(ln)
        if not L: continue

        sid = ""
        m_sid = re.search(r"(\d{7})", L)
        if m_sid: sid = m_sid.group(1)

        name = id2name.get(sid, "")
        if not name:
            mname = re.search(r"[\u4e00-\u9fa5]{2,4}", L)
            if mname: name = mname.group(0)

        d = roc_to_date(L, fallback_roc_year=fallback_roc_year)
        time_ = find_time(L)

        dir_ = ""
        if "出" in L: dir_ = "出"
        if "入" in L: dir_ = "入"
        if not dir_ and has_check(L):
            if re.search(r"出.{0,3}([✓✔Vv√勾])", L): dir_ = "出"
            elif re.search(r"入.{0,3}([✓✔Vv√勾])", L): dir_ = "入"

        kind = detect_leave_type(L, extra_type_keywords)

        if any([sid, name, d, dir_, time_, kind]) or has_check(L):
            rows.append({
                "sid": sid,
                "name": name,
                "date": date_to_roc(d) if d else "",
                "dir": dir_,
                "time": time_,
                "type": kind,
                "source": source_label,
                "raw": L
            })

    df = pd.DataFrame(rows, columns=["sid","name","date","dir","time","type","source","raw"]).drop_duplicates()
    if not df.empty:
        df["name"] = df.apply(lambda r: id2name.get(r["sid"], r["name"]), axis=1)
    return df

def parse_leave_from_lines(lines: List[str], id2name: Dict[str,str],
                           name2id: Dict[str,str], fallback_roc_year: int,
                           extra_type_keywords: list[str]) -> pd.DataFrame:
    """
    假單照片 → (sid, name, start, end, type)
    - 優先抓 7 碼學號；如未抓到，試姓名→學號
    - 支援單日或區間（114/10/27 ~ 114/10/29、或 10/27 ~ 10/29 用預設年補）
    """
    rows = []
    for ln in lines:
        L = norm(ln)
        if not L: continue

        sid = ""
        m = re.search(r"(\d{7})", L)
        if m: sid = m.group(1)

        name = id2name.get(sid, "")
        if not name:
            mname = re.search(r"[\u4e00-\u9fa5]{2,4}", L)
            if mname:
                name = mname.group(0)
                sid = sid or name2id.get(name, "")

        # 時段
        text = L.replace("年","/").replace("月","/").replace("日","")
        cand = re.findall(r"(\d{2,3}\s*/\s*\d{1,2}\s*/\s*\d{1,2}|\d{1,2}\s*/\s*\d{1,2})", text)
        if len(cand) >= 2:
            d1 = roc_to_date(cand[0], fallback_roc_year=fallback_roc_year)
            d2 = roc_to_date(cand[1], fallback_roc_year=fallback_roc_year)
        elif len(cand) == 1:
            d1 = roc_to_date(cand[0], fallback_roc_year=fallback_roc_year); d2 = d1
        else:
            d1 = d2 = None

        leave_type = detect_leave_type(L, extra_type_keywords)

        if sid and d1 and d2:
            rows.append({"sid": sid, "name": name, "start": d1, "end": d2, "type": leave_type, "raw": L})

    return pd.DataFrame(rows).drop_duplicates()

def expand_leave_days(df_leave: pd.DataFrame) -> Dict[str, List[date]]:
    mp: Dict[str, List[date]] = {}
    if df_leave.empty: return mp
    for _, r in df_leave.iterrows():
        if not r["sid"] or pd.isna(r["start"]) or pd.isna(r["end"]): continue
        cur = r["start"]
        while cur <= r["end"]:
            mp.setdefault(r["sid"], []).append(cur)
            cur += timedelta(days=1)
    return mp

def build_five_checks(df_guard: pd.DataFrame, df_squad: pd.DataFrame, df_leave: pd.DataFrame) -> pd.DataFrame:
    """
    以 (日期 × 學號 × 姓名) 為列，輸出五欄檢核：
      隊部簽出 / 隊部簽入 / 紙本假單 / 大門簽出 / 大門簽入
    """
    keys = set()
    for df in [df_guard, df_squad]:
        if df.empty: continue
        for _, r in df.iterrows():
            if r.get("date") and r.get("sid"):
                keys.add((r["date"], r["sid"], r.get("name","")))

    # 有假單也要列入五欄覆核
    if not df_leave.empty:
        for _, r in df_leave.iterrows():
            d = r["start"]
            while d <= r["end"]:
                keys.add((date_to_roc(d), r["sid"], r.get("name","")))
                d += timedelta(days=1)

    rows = [{
        "日期(ROC)": droc,
        "學號": sid,
        "姓名": name,
        "隊部簽出": "X",
        "隊部簽入": "X",
        "紙本假單": "X",
        "大門簽出": "X",
        "大門簽入": "X",
    } for (droc, sid, name) in sorted(keys)]
    table = pd.DataFrame(rows)

    # 補姓名
    def lookup_name(sid):
        for df in (df_guard, df_squad):
            t = df[df["sid"]==sid]["name"]
            if not t.empty: return t.iloc[0]
        return ""
    if not table.empty:
        table["姓名"] = table.apply(lambda r: r["姓名"] or lookup_name(r["學號"]), axis=1)

    # 打勾出入（隊部/大門）
    def mark(df, col_out, col_in):
        if df.empty: return
        for _, r in df.iterrows():
            m = (table["日期(ROC)"] == r["date"]) & (table["學號"] == r["sid"])
            if r.get("dir") == "出":
                table.loc[m, col_out] = "V"
            elif r.get("dir") == "入":
                table.loc[m, col_in"] = "V"

    # 修正：引號對稱
    def mark(df, col_out, col_in):
        if df.empty: return
        for _, r in df.iterrows():
            m = (table["日期(ROC)"] == r["date"]) & (table["學號"] == r["sid"])
            if r.get("dir") == "出":
                table.loc[m, col_out] = "V"
            elif r.get("dir") == "入":
                table.loc[m, col_in] = "V"

    mark(df_squad, "隊部簽出", "隊部簽入")
    mark(df_guard, "大門簽出", "大門簽入")

    # 紙本假單覆核（當天包含於區間即 V）
    if not df_leave.empty and not table.empty:
        daymap = expand_leave_days(df_leave)  # sid -> [date...]
        for i, r in table.iterrows():
            sid = r["學號"]; d_ = roc_to_date(r["日期(ROC)"])
            if sid in daymap and d_ and any(x == d_ for x in daymap[sid]):
                table.loc[i, "紙本假單"] = "V"

    return table.sort_values(["日期(ROC)", "學號"])

def build_download_excel(dfs: Dict[str, pd.DataFrame]) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as w:
        for name, df in dfs.items():
            if df is None or df.empty: continue
            df.to_excel(w, index=False, sheet_name=name[:31] or "Sheet1")
    bio.seek(0)
    return bio.read()

# ========================
# Streamlit UI
# ========================
st.set_page_config(page_title="差假管理員 Web v2.3.3 (Photo-only + 假別自訂)", layout="wide")
st.title("📋 差假管理員（Web v2.3.3）｜相片：簽出入 + 紙本假單 → Excel 報表（含假別自訂）")

with st.expander("ℹ️ 使用說明", expanded=True):
    st.markdown("""
**只上傳照片即可完成檢核與匯出：**
1. 上傳 `name_id_clean.txt`（學號 ↔ 姓名；順序不限；空白/逗號/Tab 都可）。
2. 上傳 **簽出入照片**：左「警衛隊（大門）」、右「研究生中隊（隊部）」。
   - 辨識 **出/入打勾**（✓/✔/V/勾）與 **時間**（09:05、9:5、9時05分）。
3. 上傳 **紙本假單照片**：自動解析 **學號/姓名** 與 **日期區間**（單日也可）。
   - 假單只寫姓名沒學號 → 會用名單自動補學號。
4. 側欄設定「**預設民國年**」與「**自訂假別**」。
5. 按「**OCR + 解析**」→「**比對並產生報表**」→ 下載 Excel。
""")

st.sidebar.header("日期與假別設定")
default_roc_year = st.sidebar.number_input("預設民國年（遇到只有 MM/DD 時補年）", min_value=110, max_value=200, value=114)
st.sidebar.subheader("假別關鍵字（可自訂）")
custom_type_str = st.sidebar.text_input("逗號分隔：如 喪,婚,慰,補休,產,兵役", value="")
EXTRA_TYPE_KEYWORDS = [norm(x) for x in re.split(r"[,，\s]+", custom_type_str) if norm(x)]

# 名單
st.header("👥 上傳學號姓名對照表（TXT）")
mapping_file = st.file_uploader("name_id_clean.txt", type=["txt"])
id2name, name2id = {}, {}
if mapping_file:
    content = mapping_file.read().decode("utf-8")
    id2name, name2id = parse_mapping_txt(content)
    st.success(f"已載入名單：{len(id2name)} 筆")

# 照片
st.header("🧾 上傳簽出入照片")
col1, col2 = st.columns(2)
with col1:
    guard_imgs = st.file_uploader("警衛隊（大門）照片（可多張）", type=["jpg","jpeg","png"], accept_multiple_files=True, key="guard")
with col2:
    squad_imgs = st.file_uploader("研究生中隊（隊部）照片（可多張）", type=["jpg","jpeg","png"], accept_multiple_files=True, key="squad")

st.header("📑 上傳紙本假單照片")
leave_imgs = st.file_uploader("假單照片（可多張）", type=["jpg","jpeg","png"], accept_multiple_files=True, key="leave_imgs")

btn_ocr = st.button("🖇️ OCR + 解析")
btn_compare = st.button("🔍 比對並產生報表")

# session 暫存
for key in ["df_guard","df_squad","df_leave","five_check"]:
    if key not in st.session_state:
        st.session_state[key] = pd.DataFrame()

# OCR + 解析
if btn_ocr:
    if not guard_imgs and not squad_imgs and not leave_imgs:
        st.warning("請至少上傳一張簽出入或假單照片。")
    else:
        reader = load_ocr_reader()

        # 大門
        rows_g = []
        for f in (guard_imgs or []):
            lines = ocr_image(reader, f)
            df_g = parse_sign_lines(lines, id2name, fallback_roc_year=default_roc_year,
                                    source_label="警衛隊", extra_type_keywords=EXTRA_TYPE_KEYWORDS)
            if not df_g.empty: rows_g.append(df_g)
        st.session_state.df_guard = pd.concat(rows_g, ignore_index=True) if rows_g else pd.DataFrame()

        # 隊部
        rows_s = []
        for f in (squad_imgs or []):
            lines = ocr_image(reader, f)
            df_s = parse_sign_lines(lines, id2name, fallback_roc_year=default_roc_year,
                                    source_label="中隊", extra_type_keywords=EXTRA_TYPE_KEYWORDS)
            if not df_s.empty: rows_s.append(df_s)
        st.session_state.df_squad = pd.concat(rows_s, ignore_index=True) if rows_s else pd.DataFrame()

        # 假單（照片）
        rows_l = []
        for f in (leave_imgs or []):
            lines = ocr_image(reader, f)
            dfL = parse_leave_from_lines(lines, id2name, name2id,
                                         fallback_roc_year=default_roc_year,
                                         extra_type_keywords=EXTRA_TYPE_KEYWORDS)
            if not dfL.empty: rows_l.append(dfL)
        st.session_state.df_leave = pd.concat(rows_l, ignore_index=True) if rows_l else pd.DataFrame()

        st.success(
            f"OCR 完成：警衛隊 {len(st.session_state.df_guard)} 筆，中隊 {len(st.session_state.df_squad)} 筆，假單 {len(st.session_state.df_leave)} 筆。"
        )

# 顯示 OCR 結果
if not st.session_state.df_guard.empty or not st.session_state.df_squad.empty:
    st.subheader("🔎 OCR 結果（簽出入，含時間與假別）")
    df_view = pd.concat([st.session_state.df_guard, st.session_state.df_squad], ignore_index=True)
    st.dataframe(df_view, use_container_width=True)

if not st.session_state.df_leave.empty:
    st.subheader("📄 假單（照片）解析結果（含假別）")
    st.dataframe(st.session_state.df_leave, use_container_width=True)

# 比對 & 匯出
if btn_compare:
    if st.session_state.df_guard.empty and st.session_state.df_squad.empty and st.session_state.df_leave.empty:
        st.warning("請先執行 OCR。")
    else:
        five = build_five_checks(st.session_state.df_guard, st.session_state.df_squad, st.session_state.df_leave)
        st.session_state.five_check = five

        st.subheader("✅ 五欄檢核表（V=有記錄 / X=缺）")
        if not five.empty:
            st.dataframe(five, use_container_width=True)
        else:
            st.info("尚無可產出的五欄檢核資料。")

        # 分開統計
        if not five.empty:
            n1_out = int((five["隊部簽出"]=="X").sum())
            n1_in  = int((five["隊部簽入"]=="X").sum())
            n2_out = int((five["大門簽出"]=="X").sum())
            n2_in  = int((five["大門簽入"]=="X").sum())
            n3     = int((five["紙本假單"]=="X").sum())
        else:
            n1_out = n1_in = n2_out = n2_in = n3 = 0

        st.subheader("📊 分項統計")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("未簽出（隊部）", n1_out)
        c2.metric("未簽入（隊部）", n1_in)
        c3.metric("未簽出（大門）", n2_out)
        c4.metric("未簽入（大門）", n2_in)
        c5.metric("未交假單", n3)

        out_bytes = build_download_excel({
            "五欄檢核": five,
            "警衛隊_OCR": st.session_state.df_guard,
            "中隊_OCR": st.session_state.df_squad,
            "假單清單": st.session_state.df_leave
        })
        st.download_button(
            label="📥 下載 Excel 報表（含五欄檢核）",
            data=out_bytes,
            file_name=f"差假管理員_五欄檢核_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
