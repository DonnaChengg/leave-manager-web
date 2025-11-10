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
    # use_cuda=False (CPU), text detection+recognition，預設中文可
    return RapidOCR()

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
        text = str(item[1])
        score = float(item[2]) if item[2] is not None else 0.0
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

DIR_WORDS = {"出":"出", "入":"入"}
TYPE_WORDS = ["病","事","特","公","假","講","書"]

def parse_sign_lines(lines: List[str], id2name: Dict[str,str], fallback_roc_year: int,
                     source_label: str) -> pd.DataFrame:
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
        for k in DIR_WORDS:
            if k in L:
                dir_ = DIR_WORDS[k]; break
        if not dir_ and has_check(L):
            if re.search(r"出.{0,3}([✓✔Vv√勾])", L): dir_ = "出"
            elif re.search(r"入.{0,3}([✓✔Vv√勾])", L): dir_ = "入"
        kind = ""
        for k in TYPE_WORDS:
            if k in L:
                kind = k; break
        if any([sid, name, d, dir_, time_]) or has_check(L):
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

def load_leave_excel(file, fallback_roc_year: int) -> pd.DataFrame:
    fn = file.name.lower()
    if fn.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    cols = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in cols: return cols[n]
        return None
    c_sid = pick("sid", "學號", "id")
    c_name = pick("name", "姓名")
    c_start = pick("start", "開始", "開始日", "從", "from")
    c_end = pick("end", "結束", "結束日", "至", "to")
    if not (c_sid and c_start and c_end):
        raise ValueError("假單需包含欄位：sid/學號、start/開始、end/結束。")
    def parse_d(x):
        if pd.isna(x): return None
        if isinstance(x, (datetime, date)):
            return x.date() if isinstance(x, datetime) else x
        s = str(x)
        d = roc_to_date(s, fallback_roc_year=fallback_roc_year)
        if d: return d
        try:
            return pd.to_datetime(s).date()
        except Exception:
            return None
    out = pd.DataFrame({
        "sid": df[c_sid].astype(str).str.extract(r"(\d{7})", expand=False),
        "name": df[c_name] if c_name else "",
        "start": df[c_start].apply(parse_d),
        "end": df[c_end].apply(parse_d),
    }).dropna(subset=["sid","start","end"])
    return out

def parse_leave_from_lines(lines: List[str], fallback_roc_year: int) -> pd.DataFrame:
    rows = []
    for ln in lines:
        L = norm(ln)
        if not L: continue
        sid = ""
        m = re.search(r"(\d{7})", L)
        if m: sid = m.group(1)
        text = L.replace("年","/").replace("月","/").replace("日","")
        cand = re.findall(r"(\d{2,3}\s*/\s*\d{1,2}\s*/\s*\d{1,2}|\d{1,2}\s*/\s*\d{1,2})", text)
        if len(cand) >= 2:
            d1 = roc_to_date(cand[0], fallback_roc_year=fallback_roc_year)
            d2 = roc_to_date(cand[1], fallback_roc_year=fallback_roc_year)
            if sid and d1 and d2:
                rows.append({"sid": sid, "name": "", "start": d1, "end": d2})
        elif len(cand) == 1:
            d1 = roc_to_date(cand[0], fallback_roc_year=fallback_roc_year)
            if sid and d1:
                rows.append({"sid": sid, "name": "", "start": d1, "end": d1})
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
    keys = set()
    for df in [df_guard, df_squad]:
        if df.empty: continue
        for _, r in df.iterrows():
            if r.get("date") and r.get("sid"):
                keys.add((r["date"], r["sid"], r.get("name","")))
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
    def lookup_name(sid):
        for df in (df_guard, df_squad):
            t = df[df["sid"]==sid]["name"]
            if not t.empty: return t.iloc[0]
        return ""
    if not table.empty:
        table["姓名"] = table.apply(lambda r: r["姓名"] or lookup_name(r["學號"]), axis=1)
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
    if not df_leave.empty and not table.empty:
        daymap = expand_leave_days(df_leave)
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
st.set_page_config(page_title="差假管理員 Web v2.3.1 (RapidOCR)", layout="wide")
st.title("📋 差假管理員（Web v2.3.1）｜RapidOCR + 名單/假單覆核 + 隊部/大門出入 + 分項統計 + 匯出")

with st.expander("ℹ️ 使用說明", expanded=True):
    st.markdown("""
**流程：**
1. 上傳 `name_id_clean.txt`（學號 ↔ 姓名，空白/逗號/Tab 皆可）。
2. 上傳 **簽出入照片**：左「警衛隊（大門）」、右「研究生中隊（隊部）」。
   - 支援照片上的**出/入打勾**與**時間**（09:05、9:5、9時05分）。
3. 上傳 **紙本假單**（擇一）：Excel/CSV（有 sid/start/end）或假單照片（會解析區間）。
4. 側欄設定「**預設民國年**」（當 OCR 只有 MM/DD 時用此補年）。
5. 點「**OCR + 解析**」→「**比對並產生報表**」→ 下載 Excel。
""")

st.sidebar.header("日期設定")
default_roc_year = st.sidebar.number_input("預設民國年（表單只有 MM/DD 時使用）", min_value=110, max_value=200, value=114)

# 名單
st.header("👥 上傳學號姓名對照表（TXT）")
mapping_file = st.file_uploader("name_id_clean.txt", type=["txt"])
id2name, name2id = {}, {}
if mapping_file:
    content = mapping_file.read().decode("utf-8")
    id2name, name2id = parse_mapping_txt(content)
    st.success(f"已載入名單：{len(id2name)} 筆")

# 照片（大門/隊部）
st.header("🧾 上傳簽出入照片（辨識出/入打勾與時間）")
col1, col2 = st.columns(2)
with col1:
    guard_imgs = st.file_uploader("警衛隊（學生總隊/大門）照片（可多張）", type=["jpg","jpeg","png"], accept_multiple_files=True, key="guard")
with col2:
    squad_imgs = st.file_uploader("研究生中隊（隊部）照片（可多張）", type=["jpg","jpeg","png"], accept_multiple_files=True, key="squad")

# 假單（Excel/CSV 或 照片）
st.header("📑 上傳紙本假單（Excel/CSV 或 假單照片）")
leave_file = st.file_uploader("Excel/CSV（需含：sid/學號、start/開始、end/結束）", type=["xlsx","xls","csv"])
leave_imgs = st.file_uploader("假單照片（可多張）", type=["jpg","jpeg","png"], accept_multiple_files=True, key="leave_imgs")

# 按鈕
btn_ocr = st.button("🖇️ OCR + 解析")
btn_compare = st.button("🔍 比對並產生報表")

# session
for key in ["df_guard","df_squad","df_leave_from_imgs","df_leave","five_check"]:
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
            df_g = parse_sign_lines(lines, id2name, fallback_roc_year=default_roc_year, source_label="警衛隊")
            if not df_g.empty: rows_g.append(df_g)
        st.session_state.df_guard = pd.concat(rows_g, ignore_index=True) if rows_g else pd.DataFrame()
        # 隊部
        rows_s = []
        for f in (squad_imgs or []):
            lines = ocr_image(reader, f)
            df_s = parse_sign_lines(lines, id2name, fallback_roc_year=default_roc_year, source_label="中隊")
            if not df_s.empty: rows_s.append(df_s)
        st.session_state.df_squad = pd.concat(rows_s, ignore_index=True) if rows_s else pd.DataFrame()
        # 假單照片 → 區間
        rows_l = []
        for f in (leave_imgs or []):
            lines = ocr_image(reader, f)
            dfL = parse_leave_from_lines(lines, fallback_roc_year=default_roc_year)
            if not dfL.empty: rows_l.append(dfL)
        st.session_state.df_leave_from_imgs = pd.concat(rows_l, ignore_index=True) if rows_l else pd.DataFrame()
        st.success(
            f"OCR 完成：警衛隊 {len(st.session_state.df_guard)} 筆，中隊 {len(st.session_state.df_squad)} 筆，假單(照片) {len(st.session_state.df_leave_from_imgs)} 筆。"
        )

# 顯示 OCR 結果
if not st.session_state.df_guard.empty or not st.session_state.df_squad.empty:
    st.subheader("🔎 OCR 結果（簽出入，含時間）")
    df_view = pd.concat([st.session_state.df_guard, st.session_state.df_squad], ignore_index=True)
    st.dataframe(df_view, use_container_width=True)

# 讀假單 Excel/CSV
if leave_file:
    try:
        st.session_state.df_leave = load_leave_excel(leave_file, fallback_roc_year=default_roc_year)
        st.success(f"已載入假單（Excel/CSV）：{len(st.session_state.df_leave)} 筆")
    except Exception as e:
        st.error(f"假單讀取失敗：{e}")

# 比對 & 匯出
if btn_compare:
    if st.session_state.df_guard.empty and st.session_state.df_squad.empty and st.session_state.df_leave.empty and st.session_state.df_leave_from_imgs.empty:
        st.warning("請先執行 OCR 或上傳假單 Excel。")
    else:
        df_leave_final = st.session_state.df_leave.copy() if not st.session_state.df_leave.empty \
            else st.session_state.df_leave_from_imgs.copy()
        five = build_five_checks(st.session_state.df_guard, st.session_state.df_squad, df_leave_final)
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
            "假單清單": df_leave_final
        })
        st.download_button(
            label="📥 下載 Excel 報表（含五欄檢核）",
            data=out_bytes,
            file_name=f"差假管理員_五欄檢核_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
