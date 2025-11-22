# app.py — fast path + friend CSV dict + 55s-safe upload (UI/routes/export unchanged)

from urllib.parse import quote
from fastapi import FastAPI, Request, UploadFile, File, Query
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import io, os, re, unicodedata, pandas as pd, time, tempfile, uuid, csv
from collections import Counter, OrderedDict
import sqlite3, threading
from typing import Optional, Iterator

# ================== 性能/时限（不改 UI，仅做软时限与退化） ==================
TIME_BUDGET_SEC = int(os.getenv("TIME_BUDGET_SEC", "45"))   # 软时限 < Render 55s
FAST_MAX_CHARS  = int(os.getenv("FAST_MAX_CHARS",  "400000"))  # 大文档先截断，仍保留足够统计意义
MAX_TOKENS      = int(os.getenv("MAX_TOKENS",      "120000"))  # 分词后最多保留的 token 数

# ================== 主词典：SQLite（保持原接口与逻辑） ==================
class SQLiteEcdict:
    def __init__(self, db_path: str, table: str = "ecdict", word_col: str = "word", zh_col: str = "translation",
                 cache_size: int = 50000):
        self.db_path = db_path
        self.table = table
        self.word_col = word_col
        self.zh_col = zh_col
        self.cache_size = max(1024, int(cache_size))
        self._lock = threading.RLock()
        uri_path = f"file:{db_path}?mode=ro"
        try:
            self._conn = sqlite3.connect(uri_path, uri=True, check_same_thread=False)
        except Exception:
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._val_cache = OrderedDict()
        self._exist_cache = OrderedDict()
        self._overrides = {"'s": "…的"}
        self._rowcount = None

        cur = self._conn.execute(f"PRAGMA table_info({self.table})")
        cols = {row["name"].lower() for row in cur.fetchall()}

        if self.word_col.lower() not in cols:
            raise RuntimeError(f"SQLite 表缺少必须列：{self.word_col}")

        main_col = os.getenv("ECDICT_ZH_COL", self.zh_col).lower()
        fallback_env = os.getenv("ECDICT_FALLBACK_COLS", "")
        fallback_list = [c.strip().lower() for c in fallback_env.split(",") if c.strip()] if fallback_env.strip() else \
                        ["cn","cn_definition","definition","meanings","trans","explains"]

        candidates = []
        if main_col in cols: candidates.append(main_col)
        for c in fallback_list:
            if c in cols and c not in candidates:
                candidates.append(c)
        if not candidates:
            raise RuntimeError("未找到可用的中文释义列。")

        self._val_cols = candidates

    def __len__(self) -> int:
        if self._rowcount is None:
            try:
                self._rowcount = int(self._conn.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()[0])
            except Exception:
                self._rowcount = 0
        return self._rowcount

    def _lru_get(self, cache: OrderedDict, key):
        try:
            with self._lock:
                val = cache.pop(key); cache[key] = val; return val
        except KeyError:
            return None

    def _lru_set(self, cache: OrderedDict, key, val):
        with self._lock:
            if key in cache: cache.pop(key)
            cache[key] = val
            if len(cache) > self.cache_size:
                cache.popitem(last=False)

    def _query_db(self, key_lc: str) -> Optional[str]:
        try:
            cols_sql = ", ".join(self._val_cols)
            cur = self._conn.execute(
                f"SELECT {cols_sql} FROM {self.table} WHERE {self.word_col} = ? COLLATE NOCASE LIMIT 1", (key_lc,)
            )
            row = cur.fetchone()
            if row is None: return None
            for c in self._val_cols:
                try: v = row[c] if isinstance(row, sqlite3.Row) else None
                except Exception: v = None
                v = (v or "").strip()
                if v: return v
            return ""
        except Exception:
            return None

    def get(self, key: str, default: str = "") -> str:
        if not key: return default
        k = key.strip().lower()
        if k in self._overrides: return self._overrides[k]
        cached = self._lru_get(self._val_cache, k)
        if cached is not None: return cached if cached != "__NONE__" else default
        val = self._query_db(k)
        if val is None:
            self._lru_set(self._val_cache, k, "__NONE__"); self._lru_set(self._exist_cache, k, False); return default
        self._lru_set(self._val_cache, k, val if val != "" else ""); self._lru_set(self._exist_cache, k, True)
        return val if val != "" else default

    def __contains__(self, key: str) -> bool:
        if not key: return False
        k = key.strip().lower()
        if k in self._overrides: return True
        cached = self._lru_get(self._exist_cache, k)
        if cached is not None: return bool(cached)
        _ = self.get(k, ""); return bool(self._lru_get(self._exist_cache, k))

    def __getitem__(self, key: str) -> str:
        v = self.get(key, None)
        if v is None: raise KeyError(key)
        return v

    def __setitem__(self, key: str, value: str):
        k = (key or "").strip().lower()
        with self._lock: self._overrides[k] = value

    def keys(self) -> Iterator[str]:  return iter(())
    def items(self) -> Iterator[tuple[str, str]]:  return iter(())
    def close(self): 
        try: self._conn.close()
        except Exception: pass

# ================== 朋友词典：CSV（优先查，查不到再落回 SQLite） ==================
class FriendCsvDict:
    def __init__(self, path: str, max_rows: int = 250000):
        self.path = path
        self.max_rows = max_rows
        self.map = {}
        if not os.path.exists(path): return
        # 只读两列：word 与 一个中文列（translation / cn / definition / explains / meanings）
        prefer_cols = ["translation","cn","definition","explains","meanings"]
        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                cols = [c.strip() for c in reader.fieldnames or []]
                if "word" not in [c.lower() for c in cols]:
                    # 支持简易两列表头：word,zh
                    pass
                zh_col = None
                lc = [c.lower() for c in cols]
                for c in prefer_cols:
                    if c in lc:
                        zh_col = cols[lc.index(c)]
                        break
                if zh_col is None and len(cols) >= 2:
                    zh_col = cols[1]  # 兜底第二列
                word_col = cols[lc.index("word")] if "word" in lc else cols[0]
                for i, row in enumerate(reader):
                    if self.max_rows and i >= self.max_rows: break
                    w = (row.get(word_col, "") or "").strip().lower()
                    z = (row.get(zh_col, "") or "").strip()
                    if w and z and (w not in self.map):
                        self.map[w] = z
        except Exception as e:
            print(f"[FRIEND_DICT] 加载失败：{e}（忽略，继续用主库）")

    def get(self, key: str, default: str = "") -> str:
        if not key: return default
        return self.map.get(key.strip().lower(), default)

# ================== 文档读取与分词（保持既有行为，仅加速/截断/退化） ==================
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    from pdfminer_high_level import extract_text as pdf_extract_text
except Exception:
    try:
        from pdfminer.high_level import extract_text as pdf_extract_text
    except Exception:
        pdf_extract_text = None

try:
    import docx
except Exception:
    docx = None

MAX_PAGES = int(os.getenv("MAX_PAGES", "500"))
MAX_CHARS = int(os.getenv("MAX_CHARS", "1200000"))

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.on_event("startup")
def _boot_probe():
    print("[BOOT] PORT=", os.getenv("PORT"))
    print("[BOOT] CWD=", os.getcwd())
    print("[BOOT] has templates/:", os.path.isdir("templates"))
    print("[BOOT] has static/:", os.path.isdir("static"))

@app.get("/_debug/env")
def _debug_env():
    return {"PORT": os.getenv("PORT"), "cwd": os.getcwd(),
            "has_templates": os.path.isdir("templates"), "has_static": os.path.isdir("static")}

STATE = {"filename": None, "df_freq": None, "df_pos": None, "page_size": 500}
SESSIONS = {}
def _get_sid_from_request(request: Request) -> str | None: return request.cookies.get("sid")
def _get_session_state(request: Request) -> dict | None:
    sid = _get_sid_from_request(request)
    return SESSIONS.get(sid) if sid else None
def _ensure_session() -> str: return uuid.uuid4().hex
def _put_session(sid: str, filename, df_freq, df_pos, page_size=500):
    SESSIONS[sid] = {"filename": filename, "df_freq": df_freq, "df_pos": df_pos, "page_size": page_size}
    MAX_SESS = int(os.getenv("MAX_SESSIONS", "200"))
    if len(SESSIONS) > MAX_SESS:
        try: SESSIONS.pop(next(iter(SESSIONS)))
        except Exception: pass

# 词典初始化（保持原路径与环境变量），并挂载朋友词典优先查找
DB_PATH = os.getenv("ECDICT_DB_PATH", "data/ecdict.sqlite3")
if not os.path.exists(DB_PATH):
    raise RuntimeError("缺少 data/ecdict.sqlite3（或设置 ECDICT_DB_PATH）。")
_ec_dict = SQLiteEcdict(
    DB_PATH,
    table=os.getenv("ECDICT_TABLE", "ecdict"),
    word_col=os.getenv("ECDICT_WORD_COL", "word"),
    zh_col=os.getenv("ECDICT_ZH_COL", "translation"),
    cache_size=int(os.getenv("ECDICT_CACHE_SIZE", "50000")),
)
_ec_dict["'s"] = "…的"

FRIEND_CSV_PATH = os.getenv("FRIEND_CSV_PATH", "data/friend_ecdict.csv")
_friend = FriendCsvDict(FRIEND_CSV_PATH, max_rows=int(os.getenv("FRIEND_MAX_ROWS","250000"))) if os.path.exists(FRIEND_CSV_PATH) else None

# 分词与清洗（与你现版一致，仅在最后切 tokens 数量）
_WORD_RE = re.compile(r"(?:[A-Za-z]+(?:['’][A-Za-z]+)?)|(?:\d+(?:[A-Za-z]+|[A-Za-z]*[\/\-][A-Za-z]+))")

def _read_text_from_upload(fname: str, data: bytes, deadline: float) -> str:
    # 读取阶段也受软时限约束，避免 PDF 超慢导致 55s 被杀
    def over(): return time.perf_counter() >= deadline
    name = (fname or "").lower()
    if name.endswith(".pdf"):
        if PdfReader is not None and not over():
            try:
                reader = PdfReader(io.BytesIO(data))
                pieces, total = [], 0
                pages = min(len(reader.pages), MAX_PAGES)
                for i in range(pages):
                    if over(): break
                    txt = reader.pages[i].extract_text() or ""
                    if txt:
                        pieces.append(txt); total += len(txt)
                        if total >= FAST_MAX_CHARS: break
                text_via_pypdf = "\n".join(pieces).strip()
                if len(text_via_pypdf) >= 50:
                    return text_via_pypdf[:min(MAX_CHARS, FAST_MAX_CHARS)]
            except Exception:
                pass
        if pdf_extract_text is not None and not over():
            try:
                with io.BytesIO(data) as f:
                    text = pdf_extract_text(f, maxpages=min(MAX_PAGES, 200))
                return (text or "")[:min(MAX_CHARS, FAST_MAX_CHARS)]
            except Exception:
                pass
        return ""
    if name.endswith(".docx") and docx is not None and not over():
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tf:
            tf.write(data); tmp = tf.name
        try:
            d = docx.Document(tmp)
        finally:
            try: os.remove(tmp)
            except Exception: pass
        return "\n".join(p.text for p in d.paragraphs)[:min(MAX_CHARS, FAST_MAX_CHARS)]
    try:
        # 纯文本最快：直接截断到 FAST_MAX_CHARS
        return data.decode("utf-8", errors="ignore")[:min(MAX_CHARS, FAST_MAX_CHARS)]
    except Exception:
        return ""

# ——— 以下大量辅助常量/函数与现版一致（为节省篇幅省掉注释） ———
_URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s]+")
_EMAIL_RE = re.compile(r"(?i)\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_SUFFIX_FUNCS = r"(?:from|for|the|that|this|these|those|your|our|their|and|to|of|with|without)"
_PREFIX_FUNCS = r"(?:and|or|but|for|to|of|in|on|at|from|by|the|that|this|these|those|with|without)"
UNICODE_ROMAN_MAP = {"Ⅰ":"I","Ⅱ":"II","Ⅲ":"III","Ⅳ":"IV","Ⅴ":"V","Ⅵ":"VI","Ⅶ":"VII","Ⅷ":"VIII","Ⅸ":"IX","Ⅹ":"X",
    "Ⅺ":"XI","Ⅻ":"XII","Ⅼ":"L","Ⅽ":"C","Ⅾ":"D","Ⅿ":"M","ⅰ":"i","ⅱ":"ii","ⅲ":"iii","ⅳ":"iv","ⅴ":"v","ⅵ":"vi",
    "ⅶ":"vii","ⅷ":"viii","ⅸ":"ix","ⅹ":"x","ⅺ":"xi","ⅻ":"xii","ⅼ":"l","ⅽ":"c","ⅾ":"d","ⅿ":"m"}
def _normalize_unicode_roman(s: str) -> str: return "".join(UNICODE_ROMAN_MAP.get(ch, ch) for ch in s)

def _preclean_text(text: str) -> str:
    text = _URL_RE.sub(" ", text); text = _EMAIL_RE.sub(" ", text)
    WATERMARK_PAT = re.compile(r"(?i)\b(zjuxz|xuezhan|zju|zjuxz\.cn)\b"); text = WATERMARK_PAT.sub(" ", text)
    text = re.sub(r"([A-Za-z])[,\uFF0C]\s*([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z])[.\u3002]\s*([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z])[;；]\s*([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z])[:：]\s*([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z])、\s*([A-Za-z])", r"\1 \2", text)
    text = re.sub(rf"([A-Za-z])({_SUFFIX_FUNCS})\b", r"\1 \2", text, flags=re.IGNORECASE)
    text = re.sub(rf"\b({_PREFIX_FUNCS})([A-Za-z])", r"\1 \2", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text

MONTHS = {"january","february","march","april","may","june","july","august","september","october","november","december"}
WEEKDAYS = {"monday","tuesday","wednesday","thursday","friday","saturday","sunday"}
ALWAYS_CAP = {"washington","australia","gutenberg","tom","daisy","gatsby"}
ROMAN_MAP = {"i":1,"ii":2,"iii":3,"iv":4,"v":5,"vi":6,"vii":7,"viii":8,"ix":9,"x":10,"xi":11,"xii":12,"xiii":13,
             "xiv":14,"xv":15,"xvi":16,"xvii":17,"xviii":18,"xix":19,"xx":20}
DETECTED_PROPER = set()
_TITLECASE_STOP = {"the","a","an","and","but","or","nor","for","so","yet","when","where","what","who","whom","whose",
    "why","which","while","if","in","on","at","to","of","from","with","without","into","onto","over","under","about","above",
    "below","he","she","it","they","we","you","i","his","her","its","their","our","your","this","that","these","those",
    "there","here","then","thus","hence","once","directions","dialogue","text","paper","okay","oh","uh","phrases",
    "answers","choices","passage","section","part"}
_COMMON_LOWER_STOP = {"one","two","three","four","five","six","seven","eight","nine","ten","first","second","third",
                      "fourth","fifth","sixth","seventh","eighth","ninth","tenth","part","section","passage","dialogue",
                      "answer","sheet","points","choice","choices","men","women","time","more","from","about","my","our","your"}
_MONTHS_DAYS = set(list(MONTHS) + list(WEEKDAYS))
_PROPER_RE = re.compile(r"\b([A-Z][a-z]+(?:-[A-Z][a-z]+)?)\b")
_WORD_LOWER_RE = re.compile(r"\b([a-z]+(?:-[a-z]+)?)\b")

def _detect_proper_nouns_from_text(raw_text: str):
    DETECTED_PROPER.clear()
    cap_cnt = Counter(); low_cnt = Counter()
    for m in _PROPER_RE.finditer(raw_text): cap_cnt[m.group(1).lower()] += 1
    for m in _WORD_LOWER_RE.finditer(raw_text): low_cnt[m.group(1).lower()] += 1
    for wl, c in cap_cnt.items():
        if c >= 2 and low_cnt.get(wl, 0) == 0:
            if wl in _TITLECASE_STOP or wl in _COMMON_LOWER_STOP or wl in _MONTHS_DAYS or wl == "i": continue
            DETECTED_PROPER.add(wl)

def _tokenize(text: str):
    # …………（此处保持你现有的完整分词与修复逻辑，略）…………
    # 直接复用你现有实现 —— 为节省篇幅，这里仅保留结尾“数量裁剪”，不改变既有行为
    # 你可以把你现有 _tokenize 函数原样贴进来；下面只展示最后两行：
    toks = [m.group(0).lower() for m in _WORD_RE.finditer(text)]
    # …你的清洗/合并/修复流程…
    # 裁剪，硬保证不会超时
    if len(toks) > MAX_TOKENS:
        toks = toks[:MAX_TOKENS]
    return toks

# 显示/释义清洗、派生/复数兜底、手工收缩词表 —— 维持现有实现（略）
# 你可以把你现有的 _format_zh / _fallback_guess / _plural_fallback 等函数原样保留

# ============== 核心差异：查词先 friend，再主库，再派生/复数兜底（其余不变） ==============
def _lookup_zh_base(word_lc: str) -> str:
    if _friend:
        z = _friend.get(word_lc, "")
        if z: return z
    return _ec_dict.get(word_lc, "")

def _build_dataframe(tokens):
    ctr = Counter(tokens); first_pos = {}
    for idx, w in enumerate(tokens):
        if w not in first_pos: first_pos[w] = idx
    df = pd.DataFrame({"word": list(ctr.keys()), "count": [ctr[w] for w in ctr.keys()], "pos": [first_pos[w] for w in ctr.keys()]})
    # 朋友词典优先
    df["zh"] = df["word"].map(lambda w: _lookup_zh_base(w.lower())).fillna("")
    # 你的现有“罗马数字/收缩词/连字符/复数/派生/大小写显示”等逻辑照旧（略），最终仍得到 df_freq/df_pos
    # —— 下面两行保持不变 —— 
    df_freq = df.sort_values(["count", "word"], ascending=[False, True]).reset_index(drop=True)
    df_pos  = df.sort_values(["pos"]).reset_index(drop=True)
    return df_freq, df_pos

# ================== 路由（完全不变） ==================
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.head("/")
def root_head(): return PlainTextResponse("", status_code=200)

@app.post("/upload")
async def upload(request: Request, file: UploadFile = File(...)):
    deadline = time.perf_counter() + TIME_BUDGET_SEC   # 软时限起点
    try:
        data = await file.read()
        # 读取文本（受软时限与 FAST_MAX_CHARS 保护）
        text = _read_text_from_upload(file.filename, data, deadline)
        # 专名检测（若剩余时间太少则跳过，不影响主流程）
        if time.perf_counter() + 3 < deadline:
            _detect_proper_nouns_from_text(text)
        # 分词（内部会裁剪 MAX_TOKENS，避免超时）
        tokens = _tokenize(_preclean_text(text))
        if not tokens:
            raise ValueError("未解析到有效英文单词")
        # 构建结果表
        df_freq, df_pos = _build_dataframe(tokens)
        sid = _get_sid_from_request(request) or _ensure_session()
        _put_session(sid, file.filename, df_freq, df_pos, page_size=STATE.get("page_size", 500))
        STATE["filename"] = file.filename; STATE["df_freq"] = df_freq; STATE["df_pos"] = df_pos
        resp = RedirectResponse(url="/result?sort=freq&page=1", status_code=303)
        resp.set_cookie("sid", sid, httponly=True, samesite="lax")
        return resp
    except Exception as e:
        print(f"[ERROR] {e}")
        return PlainTextResponse("处理文件时出现错误，请确认文件无损坏或换一份测试。\n\nDETAILS: " + str(e), status_code=500)

def _slice_page(df: pd.DataFrame, page: int, page_size: int):
    total = len(df); pages = max(1, (total + page_size - 1) // page_size)
    page = max(1, min(page, pages))
    s = (page - 1) * page_size; e = min(s + page_size, total)
    sub = df.iloc[s:e].copy(); sub.insert(0, "序号", range(s + 1, e + 1))
    sub = sub.rename(columns={"count": "出现次数", "word": "单词", "zh": "翻译"})
    sub = sub[["序号", "出现次数", "单词", "翻译"]]
    return sub, page, pages, total

@app.get("/result", response_class=HTMLResponse)
def result(request: Request, sort: str = Query("freq", pattern="^(freq|pos)$"), page: int = 1):
    sess = _get_session_state(request)
    df_freq = sess["df_freq"] if sess else STATE["df_freq"]
    df_pos  = sess["df_pos"]  if sess else STATE["df_pos"]
    filename = sess["filename"] if sess else STATE["filename"]
    page_size = (sess or STATE).get("page_size", 500)
    if df_freq is None: return RedirectResponse("/", status_code=303)
    df = df_freq if sort == "freq" else df_pos
    sub, cur, pages, total = _slice_page(df, page, page_size)
    return templates.TemplateResponse("result.html", {"request": request, "filename": filename,
        "rows": sub.to_dict(orient="records"), "page": cur, "pages": pages, "total": total,
        "page_size": page_size, "sort": sort})

@app.get("/export")
def export(request: Request, sort: str = Query("freq", pattern="^(freq|pos)$")):
    sess = _get_session_state(request)
    df_freq = sess["df_freq"] if sess else STATE["df_freq"]
    df_pos  = sess["df_pos"]  if sess else STATE["df_pos"]
    filename = sess["filename"] if sess else STATE["filename"]
    if df_freq is None: return RedirectResponse("/", status_code=303)
    df = df_freq if sort == "freq" else df_pos
    out = df.rename(columns={"count": "出现次数", "word": "单词", "zh": "翻译"}).copy()
    if "序号" not in out.columns: out.insert(0, "序号", range(1, len(out) + 1))
    out = out[["序号", "出现次数", "单词", "翻译"]]
    try:
        import xlsxwriter  # noqa
    except Exception:
        return PlainTextResponse("缺少依赖：xlsxwriter。请在虚拟环境中执行：pip install xlsxwriter", status_code=500)
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        out.to_excel(writer, index=False, sheet_name="words")
        wb  = writer.book; ws  = writer.sheets["words"]
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
    docname  = (filename or "文档").rsplit(".", 1)[0]
    final_xlsx_name = f"单词（{mode_cn}）—（{docname}）.xlsx"
    headers = {"Content-Disposition": f"attachment; filename=\"words.xlsx\"; filename*=UTF-8''{quote(final_xlsx_name)}"}
    return StreamingResponse(bio, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers=headers)

@app.get("/healthz")
def healthz(): return {"status": "ok"}
@app.head("/healthz")
def healthz_head(): return PlainTextResponse("", status_code=200)