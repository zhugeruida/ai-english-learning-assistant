# app.py
from urllib.parse import quote  # 用于 Content-Disposition 的 UTF-8 文件名
from fastapi import FastAPI, Request, UploadFile, File, Query
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import io
import os
import re
import unicodedata
import pandas as pd
from collections import Counter
import csv    # 轻量读取 ecdict.csv

# 可选：pdf/docx 解析
try:
    from pdfminer_high_level import extract_text as pdf_extract_text  # 某些环境包名不同
except Exception:
    try:
        from pdfminer.high_level import extract_text as pdf_extract_text
    except Exception:
        pdf_extract_text = None

try:
    import docx
except Exception:
    docx = None

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 进程内状态
STATE = {
    "filename": None,
    "df_freq": None,
    "df_pos": None,
    "page_size": 500,
}

# 读取 ECDICT（轻量：只取需要列并存为 dict，降低内存占用）
DICT_PATH = "data/ecdict.csv"
if not os.path.exists(DICT_PATH):
    raise RuntimeError("缺少 data/ecdict.csv，请放到项目 data/ 目录下。")

_ec_dict = {}
with open(DICT_PATH, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    word_key = None
    zh_key = None
    for h in reader.fieldnames or []:
        hl = h.strip().lower()
        if hl == "word":
            word_key = h
        elif hl in ("translation", "zh", "trans"):
            zh_key = h
    if word_key is None or zh_key is None:
        raise RuntimeError("ecdict.csv 缺少必须字段：word / translation")

    for row in reader:
        w = (row.get(word_key) or "").strip().lower()
        zh = (row.get(zh_key) or "").strip()
        if w:
            _ec_dict[w] = zh

# 给 possessive "'s" 固定翻译
_ec_dict["'s"] = "…的"

# ---------- 分词 ----------
_WORD_RE = re.compile(
    r"(?:[A-Za-z]+(?:['’][A-Za-z]+)?)"
    r"|(?:\d+(?:[A-Za-z]+|[A-Za-z]*[\/\-][A-Za-z]+))"
)

def _read_text_from_upload(fname: str, data: bytes) -> str:
    name = (fname or "").lower()
    if name.endswith(".pdf") and pdf_extract_text is not None:
        with io.BytesIO(data) as f:
            return pdf_extract_text(f)
    if name.endswith(".docx") and docx is not None:
        with io.BytesIO(data) as bio:
            tmp = ".__tmp__.docx"
            with open(tmp, "wb") as w:
                w.write(bio.read())
            d = docx.Document(tmp)
            os.remove(tmp)
        return "\n".join(p.text for p in d.paragraphs)
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""

def _tokenize(text: str):
    text = unicodedata.normalize("NFKC", text)
    raw = [m.group(0).lower() for m in _WORD_RE.finditer(text)]
    toks = []
    for w in raw:
        if (w.endswith("'s") or w.endswith("’s")) and len(w) > 2:
            base = w[:-2]
            if base:
                toks.append(base)
            toks.append("'s")
        elif (w.endswith("s’") or w.endswith("s'")) and len(w) > 2:
            base = w[:-2]
            if base:
                toks.append(base)
            toks.append("'s")
        elif w in {"’s", "'s"}:
            toks.append("'s")
        else:
            toks.append(w)
    # ✅ 仅保留 a / i / v / x，其他单字母丢弃（满足“移除孤立垃圾字母”且保留罗马数字）
    allow_single = {"a", "i", "v", "x"}
    toks = [w for w in toks if (len(w) > 1) or (w in allow_single)]
    return toks

# ---------- 显示与释义清洗 ----------
_POS_TAGS = ["n", "vi", "vt", "v", "adj", "a", "adv", "art", "pron", "prep",
             "conj", "int", "num", "aux", "abbr", "pref", "suff"]

def _normalize_pos_display(line: str) -> str:
    return re.sub(r"(^|\n)\s*a\.", r"\1adj.", line)

def _format_zh(zh: str) -> str:
    if not zh:
        return ""
    s = zh.replace("\\n", "\n").replace("\r", "\n").replace("\\r", "")
    s = re.sub(r"\n+", "\n", s)
    t = re.sub(r"\s*((?:" + "|".join(_POS_TAGS) + r")\.)", r"\n\1", s)
    lines = [ln.strip() for ln in t.split("\n")]
    out, seen = [], set()
    for ln in lines:
        if not ln:
            continue
        ln = _normalize_pos_display(ln)
        if ln in seen:
            continue
        seen.add(ln)
        out.append(ln)
    return "\n".join(out)

# 手动缩略词词典
# 手动缩略词词典（直引号/弯引号都覆盖）
MANUAL_CONTRACTIONS = {
    "don't": "aux. 表示否定（= do not）", "don’t": "aux. 表示否定（= do not）",
    "didn't": "aux. 不（= did not）", "didn’t": "aux. 不（= did not）",
    "isn't": "不是（= is not）", "isn’t": "不是（= is not）",
    "hasn't": "没有（= has not）", "hasn’t": "没有（= has not）",
    "haven't": "没有（= have not）", "haven’t": "没有（= have not）",
    "hadn't": "没有（= had not）", "hadn’t": "没有（= had not）",
    "won't": "不会（= will not）", "won’t": "不会（= will not）",
    "wouldn't": "不会；不愿（= would not）", "wouldn’t": "不会；不愿（= would not）",
    "can't": "不能（= cannot）", "can’t": "不能（= cannot）",
    "couldn't": "不能（= could not）", "couldn’t": "不能（= could not）",
    "shouldn't": "不应该（= should not）", "shouldn’t": "不应该（= should not）",
    "doesn't": "aux. 表示否定（= does not）", "doesn’t": "aux. 表示否定（= does not）",

    "i'm": "abbr. 我是（= I am）", "i’m": "abbr. 我是（= I am）",
    "i've": "contr. 我已/我有（= I have）", "i’ve": "contr. 我已/我有（= I have）",
    "i'd": "contr. 我愿意/我已经/我应该（= I would / I had / I should）", "i’d": "contr. 我愿意/我已经/我应该（= I would / I had / I should）",
    "i'll": "contr. 我将/我会（= I will / I shall）", "i’ll": "contr. 我将/我会（= I will / I shall）",

    "you're": "contr. 你是（= you are）", "you’re": "contr. 你是（= you are）",
    "you've": "contr. 你已/你有（= you have）", "you’ve": "contr. 你已/你有（= you have）",
    "you'll": "contr. 你将/你会（= you will / you shall）", "you’ll": "contr. 你将/你会（= you will / you shall）",

    "we're": "contr. 我们是（= we are）", "we’re": "contr. 我们是（= we are）",
    "we've": "contr. 我们已/我们有（= we have）", "we’ve": "contr. 我们已/我们有（= we have）",
    "we'll": "contr. 我们将/我们会（= we will / we shall）", "we’ll": "contr. 我们将/我们会（= we will / we shall）",

    "they're": "contr. 他们是（= they are）", "they’re": "contr. 他们是（= they are）",
    "they've": "contr. 他们已/他们有（= they have）", "they’ve": "contr. 他们已/他们有（= they have）",
    "they'll": "contr. 他们将/他们会（= they will / they shall）", "they’ll": "contr. 他们将/他们会（= they will / they shall）",

    "he's": "contr. 他是/他有（= he is / he has）", "he’s": "contr. 他是/他有（= he is / he has）",
    "he'd": "contr. 他愿意/他已经/他应该（= he would / he had / he should）", "he’d": "contr. 他愿意/他已经/他应该（= he would / he had / he should）",
    "he'll": "contr. 他将/他会（= he will / he shall）", "he’ll": "contr. 他将/他会（= he will / he shall）",

    "she's": "contr. 她是/她有（= she is / she has）", "she’s": "contr. 她是/她有（= she is / she has）",
    "she'd": "contr. 她愿意/她已经/她应该（= she would / she had / she should）", "she’d": "contr. 她愿意/她已经/她应该（= she would / she had / she should）",
    "she'll": "contr. 她将/她会（= she will / she shall）", "she’ll": "contr. 她将/她会（= she will / she shall）",

    "it's": "contr. 它是/它有（= it is / it has）", "it’s": "contr. 它是/它有（= it is / it has）",
    "it'll": "contr. 它将/它会（= it will）", "it’ll": "contr. 它将/它会（= it will）",
    "it'd": "contr. 它将会/它本会（= it would / it had）", "it’d": "contr. 它将会/它本会（= it would / it had）",

    "that's": "那是；正是（= that is / that has）", "that’s": "那是；正是（= that is / that has）",
    "there's": "那里有；有（= there is / there has）", "there’s": "那里有；有（= there is / there has）",
    "here's": "这儿是；给你（= here is）", "here’s": "这儿是；给你（= here is）",
    "what's": "什么是（= what is / what has）", "what’s": "什么是（= what is / what has）",
    "who's": "谁是（= who is / who has）", "who’s": "谁是（= who is / who has）",
    "where's": "哪里是；哪里有（= where is / where has）", "where’s": "哪里是；哪里有（= where is / where has）",
    "how's": "情况如何（= how is / how has）", "how’s": "情况如何（= how is / how has）",

    "let's": "让我们…（= let us）", "let’s": "让我们…（= let us）",
    "o'clock": "abbr. ……点钟（of the clock）", "o’clock": "abbr. ……点钟（of the clock）",
}

MONTHS = {"january","february","march","april","may","june","july","august","september","october","november","december"}
WEEKDAYS = {"monday","tuesday","wednesday","thursday","friday","saturday","sunday"}
ALWAYS_CAP = {"washington","australia","gutenberg","tom","daisy","gatsby"}

ROMAN_MAP = {
    "i":1,"ii":2,"iii":3,"iv":4,"v":5,"vi":6,"vii":7,"viii":8,"ix":9,"x":10,
    "xi":11,"xii":12,"xiii":13,"xiv":14,"xv":15,"xvi":16,"xvii":17,"xviii":18,"xix":19,"xx":20
}

def _display_casing(word: str) -> str:
    if word == "i":
        return "I"
    if word.startswith("i'") or word.startswith("i’"):
        return "I" + word[1:]
    if word in MONTHS or word in WEEKDAYS or word in ALWAYS_CAP:
        return word[:1].upper() + word[1:]
    if word in ROMAN_MAP:
        return word.upper()
    return word

def _plural_fallback(word: str, base_zh_lookup) -> str:
    if not word or len(word) < 2:
        return ""
    cand = []
    if word.endswith("ies") and len(word) > 3:
        cand.append(word[:-3] + "y")
    if word.endswith("es") and len(word) > 2:
        cand.append(word[:-2])
    if word.endswith("s") and len(word) > 1:
        cand.append(word[:-1])
    for b in cand:
        zh = base_zh_lookup.get(b, "")
        if zh:
            return f"{_format_zh(zh)}（复数）"
    return ""

def _build_dataframe(tokens):
    ctr = Counter(tokens)
    first_pos = {}
    for idx, w in enumerate(tokens):
        if w not in first_pos:
            first_pos[w] = idx

    df = pd.DataFrame({
        "word": list(ctr.keys()),
        "count": [ctr[w] for w in ctr.keys()],
        "pos":   [first_pos[w] for w in ctr.keys()],
    })

    # ① 用内存字典 _ec_dict 取释义
    df["zh"] = df["word"].str.lower().map(_ec_dict).fillna("")

    # ② 罗马数字兜底：译为“罗马数字N”
    is_roman = df["word"].map(lambda w: w in ROMAN_MAP)
    df.loc[is_roman, "zh"] = df.loc[is_roman, "word"].map(lambda w: f"罗马数字{ROMAN_MAP[w]}")

    # ③ 缩略词兜底
    df["zh"] = df.apply(lambda r: MANUAL_CONTRACTIONS.get(r["word"], r["zh"]), axis=1).astype(str)

    # ④ special: i / I 多义显示——分行
    mask_i = df["word"].eq("i")
    df.loc[mask_i, "zh"] = "pron. 我\nn. 碘元素；字母I；罗马数字1"

    # ⑤ 复数兜底（ebooks -> 电子书（复数））
    need_plural = df["zh"].eq("") & df["word"].str.endswith("s")
    df.loc[need_plural, "zh"] = df.loc[need_plural, "word"].map(lambda w: _plural_fallback(w, _ec_dict))

    # ⑥ 清洗/分行
    df["zh"] = df["zh"].map(_format_zh)

    # ⑦ 显示层大小写
    df["word"] = df["word"].map(_display_casing)

    df_freq = df.sort_values(["count", "word"], ascending=[False, True]).reset_index(drop=True)
    df_pos  = df.sort_values(["pos"]).reset_index(drop=True)
    return df_freq, df_pos

# ---------- 页面 ----------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload(request: Request, file: UploadFile = File(...)):
    try:
        data = await file.read()
        text = _read_text_from_upload(file.filename, data)
        tokens = _tokenize(text)
        if not tokens:
            raise ValueError("未解析到有效英文单词")

        df_freq, df_pos = _build_dataframe(tokens)
        STATE["filename"] = file.filename
        STATE["df_freq"] = df_freq
        STATE["df_pos"] = df_pos
        return RedirectResponse(url="/result?sort=freq&page=1", status_code=303)
    except Exception as e:
        return PlainTextResponse(
            "处理文件时出现错误，请确认文件无损坏或换一份测试。\n\nDETAILS: " + str(e),
            status_code=500
        )

def _slice_page(df: pd.DataFrame, page: int, page_size: int):
    total = len(df)
    pages = max(1, (total + page_size - 1) // page_size)
    page = max(1, min(page, pages))
    s = (page - 1) * page_size
    e = min(s + page_size, total)

    sub = df.iloc[s:e].copy()
    sub.insert(0, "序号", range(s + 1, e + 1))
    sub = sub.rename(columns={"count": "出现次数", "word": "单词", "zh": "翻译"})
    sub = sub[["序号", "出现次数", "单词", "翻译"]]
    return sub, page, pages, total

@app.get("/result", response_class=HTMLResponse)
def result(request: Request, sort: str = Query("freq", pattern="^(freq|pos)$"), page: int = 1):
    if STATE["df_freq"] is None:
        return RedirectResponse("/", status_code=303)

    df = STATE["df_freq"] if sort == "freq" else STATE["df_pos"]
    page_size = STATE["page_size"]
    sub, cur, pages, total = _slice_page(df, page, page_size)
    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "filename": STATE["filename"],
            "rows": sub.to_dict(orient="records"),
            "page": cur,
            "pages": pages,
            "total": total,
            "page_size": page_size,
            "sort": sort,
        }
    )

# ---------- 导出（跟随当前排序） ----------
@app.get("/export")
def export(sort: str = Query("freq", pattern="^(freq|pos)$")):
    if STATE["df_freq"] is None:
        return RedirectResponse("/", status_code=303)

    df = STATE["df_freq"] if sort == "freq" else STATE["df_pos"]
    out = df.rename(columns={"count": "出现次数", "word": "单词", "zh": "翻译"}).copy()
    if "序号" not in out.columns:
        out.insert(0, "序号", range(1, len(out) + 1))
    out = out[["序号", "出现次数", "单词", "翻译"]]

    try:
        import xlsxwriter  # noqa
    except Exception:
        return PlainTextResponse("缺少依赖：xlsxwriter。请在虚拟环境中执行：pip install xlsxwriter", status_code=500)

    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        out.to_excel(writer, index=False, sheet_name="words")
        wb  = writer.book
        ws  = writer.sheets["words"]

        fmt_header = wb.add_format({"bold": True, "align": "center", "valign": "vcenter", "font_size": 12})
        fmt_center = wb.add_format({"align": "center", "valign": "vcenter", "text_wrap": True})
        fmt_left_wrap = wb.add_format({"align": "left", "valign": "vcenter", "text_wrap": True})

        ws.set_row(0, 22, fmt_header)
        ws.set_column("A:A", 6,  fmt_center)
        ws.set_column("B:B", 10, fmt_center)
        ws.set_column("C:C", 20, fmt_left_wrap)
        ws.set_column("D:D", 80, fmt_left_wrap)

    bio.seek(0)
    mode_cn  = "词频降序排列" if sort == "freq" else "出现位置排列"
    docname  = (STATE["filename"] or "文档").rsplit(".", 1)[0]
    final_xlsx_name = f"单词（{mode_cn}）—（{docname}）.xlsx"
    headers = {
        "Content-Disposition": f"attachment; filename=\"words.xlsx\"; filename*=UTF-8''{quote(final_xlsx_name)}"
    }
    return StreamingResponse(
        bio,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers=headers
    )

@app.get("/healthz")
def healthz():
    return {"status": "ok"}         # ← GET 保留返回 JSON

@app.head("/healthz")
def healthz_head():
    # UptimeRobot 免费版用 HEAD 探测，返回 200 即可
    return PlainTextResponse("", status_code=200)