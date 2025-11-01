# app.py — fix white screen after upload & reduce "保留原词" further
# 注意：UI/模板/导出/路由签名不变；仅做内部稳健性和轻量性能修补

from urllib.parse import quote
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
import time
import tempfile
import uuid

# ========= SQLite + LRU 词典（保持接口不变）=========
import sqlite3
import threading
from collections import OrderedDict
from typing import Optional, Iterator

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

        try:
            cur = self._conn.execute(f"PRAGMA table_info({self.table})")
            cols = {row["name"].lower() for row in cur.fetchall()}
        except Exception as e:
            raise RuntimeError(f"无法读取 SQLite 元数据：{e}")

        if self.word_col.lower() not in cols:
            raise RuntimeError(f"SQLite 表缺少必须列：{self.word_col}")

        main_col = os.getenv("ECDICT_ZH_COL", self.zh_col).lower()
        fallback_env = os.getenv("ECDICT_FALLBACK_COLS", "")
        if fallback_env.strip():
            fallback_list = [c.strip().lower() for c in fallback_env.split(",") if c.strip()]
        else:
            fallback_list = ["cn", "cn_definition", "definition", "meanings", "trans", "explains"]

        candidates = []
        if main_col in cols:
            candidates.append(main_col)
        for c in fallback_list:
            if c in cols and c not in candidates:
                candidates.append(c)

        if not candidates:
            raise RuntimeError(
                "未找到可用的中文释义列。请提供包含 translation/cn/definition 等列的 ecdict 数据库，"
                "或通过 ECDICT_ZH_COL / ECDICT_FALLBACK_COLS 指定。"
            )
        self._val_cols = candidates

    def __len__(self) -> int:
        if self._rowcount is None:
            try:
                cur = self._conn.execute(f"SELECT COUNT(*) FROM {self.table}")
                self._rowcount = int(cur.fetchone()[0])
            except Exception:
                self._rowcount = 0
        return self._rowcount

    def _lru_get(self, cache: OrderedDict, key):
        try:
            with self._lock:
                val = cache.pop(key)
                cache[key] = val
                return val
        except KeyError:
            return None

    def _lru_set(self, cache: OrderedDict, key, val):
        with self._lock:
            if key in cache:
                cache.pop(key)
            cache[key] = val
            if len(cache) > self.cache_size:
                cache.popitem(last=False)

    def _query_db(self, key_lc: str) -> Optional[str]:
        try:
            cols_sql = ", ".join(self._val_cols)
            sql = (
                f"SELECT {cols_sql} "
                f"FROM {self.table} "
                f"WHERE {self.word_col} = ? COLLATE NOCASE "
                f"LIMIT 1"
            )
            cur = self._conn.execute(sql, (key_lc,))
            row = cur.fetchone()
            if row is None:
                return None
            for c in self._val_cols:
                try:
                    v = row[c] if isinstance(row, sqlite3.Row) else None
                except Exception:
                    v = None
                v = (v or "").strip()
                if v:
                    return v
            return ""
        except Exception:
            return None

    def get(self, key: str, default: str = "") -> str:
        if not key:
            return default
        k = key.strip().lower()
        if k in self._overrides:
            return self._overrides[k]
        cached = self._lru_get(self._val_cache, k)
        if cached is not None:
            return cached if cached != "__NONE__" else default
        val = self._query_db(k)
        if val is None:
            self._lru_set(self._val_cache, k, "__NONE__")
            self._lru_set(self._exist_cache, k, False)
            return default
        self._lru_set(self._val_cache, k, val if val != "" else "")
        self._lru_set(self._exist_cache, k, True)
        return val if val != "" else default

    def __contains__(self, key: str) -> bool:
        if not key:
            return False
        k = key.strip().lower()
        if k in self._overrides:
            return True
        cached = self._lru_get(self._exist_cache, k)
        if cached is not None:
            return bool(cached)
        _ = self.get(k, "")
        cached2 = self._lru_get(self._exist_cache, k)
        return bool(cached2)

    def __getitem__(self, key: str) -> str:
        v = self.get(key, None)
        if v is None:
            raise KeyError(key)
        return v

    def __setitem__(self, key: str, value: str):
        k = (key or "").strip().lower()
        with self._lock:
            self._overrides[k] = value

    def keys(self) -> Iterator[str]:
        return iter(())

    def items(self) -> Iterator[tuple[str, str]]:
        return iter(())

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass


# ========= 优先尝试 PyPDF（保持）=========
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

# 调低默认字符上限，避免免费实例冷启动叠加超时；可用环境变量 MAX_CHARS 调整，不改 UI
MAX_PAGES = int(os.getenv("MAX_PAGES", "500"))
MAX_CHARS = int(os.getenv("MAX_CHARS", "350000"))

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
    return {
        "PORT": os.getenv("PORT"),
        "cwd": os.getcwd(),
        "has_templates": os.path.isdir("templates"),
        "has_static": os.path.isdir("static"),
    }

STATE = {"filename": None, "df_freq": None, "df_pos": None, "page_size": 500}
SESSIONS = {}

def _get_sid_from_request(request: Request) -> str | None:
    return request.cookies.get("sid")

def _get_session_state(request: Request) -> dict | None:
    sid = _get_sid_from_request(request)
    if sid and sid in SESSIONS:
        return SESSIONS[sid]
    return None

def _ensure_session() -> str:
    return uuid.uuid4().hex

def _put_session(sid: str, filename, df_freq, df_pos, page_size=500):
    SESSIONS[sid] = {"filename": filename, "df_freq": df_freq, "df_pos": df_pos, "page_size": page_size}
    MAX_SESS = int(os.getenv("MAX_SESSIONS", "200"))
    if len(SESSIONS) > MAX_SESS:
        try:
            SESSIONS.pop(next(iter(SESSIONS)))
        except Exception:
            pass

# ================== 词典配置（不变） ==================
DB_PATH = os.getenv("ECDICT_DB_PATH", "data/ecdict.sqlite3")
if not os.path.exists(DB_PATH):
    raise RuntimeError("缺少 data/ecdict.sqlite3（或设置 ECDICT_DB_PATH）。请提供包含 word/translation 列的 SQLite 数据库。")

_ECDICT_CACHE_SIZE = int(os.getenv("ECDICT_CACHE_SIZE", "50000"))
_ec_dict = SQLiteEcdict(
    DB_PATH,
    table=os.getenv("ECDICT_TABLE", "ecdict"),
    word_col=os.getenv("ECDICT_WORD_COL", "word"),
    zh_col=os.getenv("ECDICT_ZH_COL", "translation"),
    cache_size=_ECDICT_CACHE_SIZE
)
_ec_dict["'s"] = "…的"

# ---------- 分词 ----------
_WORD_RE = re.compile(r"(?:[A-Za-z]+(?:['’][A-Za-z]+)?)|(?:\d+(?:[A-Za-z]+|[A-Za-z]*[\/\-][A-Za-z]+))")

def _read_text_from_upload(fname: str, data: bytes) -> str:
    name = (fname or "").lower()
    if name.endswith(".pdf"):
        if PdfReader is not None:
            try:
                reader = PdfReader(io.BytesIO(data))
                pieces, total = [], 0
                pages = min(len(reader.pages), MAX_PAGES)
                for i in range(pages):
                    txt = reader.pages[i].extract_text() or ""
                    if txt:
                        pieces.append(txt)
                        total += len(txt)
                        if total >= MAX_CHARS:
                            break
                text_via_pypdf = "\n".join(pieces).strip()
                if len(text_via_pypdf) > 50:
                    return text_via_pypdf[:MAX_CHARS]
            except Exception:
                pass
        if pdf_extract_text is not None:
            try:
                with io.BytesIO(data) as f:
                    text = pdf_extract_text(f, maxpages=MAX_PAGES)
                return (text or "")[:MAX_CHARS]
            except Exception:
                pass
        return ""
    if name.endswith(".docx") and docx is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tf:
            tf.write(data); tmp = tf.name
        try:
            d = docx.Document(tmp)
        finally:
            try: os.remove(tmp)
            except Exception: pass
        return "\n".join(p.text for p in d.paragraphs)[:MAX_CHARS]
    try:
        return data.decode("utf-8", errors="ignore")[:MAX_CHARS]
    except Exception:
        return ""

_URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s]+")
_EMAIL_RE = re.compile(r"(?i)\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_SUFFIX_FUNCS = r"(?:from|for|the|that|this|these|those|your|our|their|and|to|of|with|without)"
_PREFIX_FUNCS = r"(?:and|or|but|for|to|of|in|on|at|from|by|the|that|this|these|those|with|without)"

UNICODE_ROMAN_MAP = {"Ⅰ":"I","Ⅱ":"II","Ⅲ":"III","Ⅳ":"IV","Ⅴ":"V","Ⅵ":"VI","Ⅶ":"VII","Ⅷ":"VIII","Ⅸ":"IX","Ⅹ":"X",
    "Ⅺ":"XI","Ⅻ":"XII","Ⅼ":"L","Ⅽ":"C","Ⅾ":"D","Ⅿ":"M","ⅰ":"i","ⅱ":"ii","ⅲ":"iii","ⅳ":"iv","ⅴ":"v","ⅵ":"vi",
    "ⅶ":"vii","ⅷ":"viii","ⅸ":"ix","ⅹ":"x","ⅺ":"xi","ⅻ":"xii","ⅼ":"l","ⅽ":"c","ⅾ":"d","ⅿ":"m"}

def _normalize_unicode_roman(s: str) -> str:
    return "".join(UNICODE_ROMAN_MAP.get(ch, ch) for ch in s)

def _preclean_text(text: str) -> str:
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    WATERMARK_PAT = re.compile(r"(?i)\b(zjuxz|xuezhan|zju|zjuxz\.cn)\b")
    text = WATERMARK_PAT.sub(" ", text)
    text = re.sub(r"([A-Za-z])[,\uFF0C]\s*([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z])[.\u3002]\s*([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z])[;；]\s*([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z])[:：]\s*([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z])、\s*([A-Za-z])", r"\1 \2", text)
    text = re.sub(rf"([A-Za-z])({_SUFFIX_FUNCS})\b", r"\1 \2", text, flags=re.IGNORECASE)
    text = re.sub(rf"\b({_PREFIX_FUNCS})([A-Za-z])", r"\1 \2", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text

# 词性/专名/罗马数字等辅件（保持不变）
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
    # 性能安全：仅用前 MAX_CHARS 的 60% 做专名探测，避免超时
    sample = (raw_text or "")[: int(MAX_CHARS * 0.6)]
    DETECTED_PROPER.clear()
    cap_cnt = Counter(); low_cnt = Counter()
    for m in _PROPER_RE.finditer(sample):
        wl = m.group(1).lower(); cap_cnt[wl] += 1
    for m in _WORD_LOWER_RE.finditer(sample):
        wl = m.group(1).lower(); low_cnt[wl] += 1
    for wl, c in cap_cnt.items():
        if c >= 2 and low_cnt.get(wl, 0) == 0:
            if wl in _TITLECASE_STOP or wl in _COMMON_LOWER_STOP or wl in _MONTHS_DAYS or wl == "i":
                continue
            DETECTED_PROPER.add(wl)

# 预编译功能词切分的正则集合（原函数每次动态编译，成本高）
_FUNC_WORDS = {'to','of','on','in','for','from','with','without','and','or','but','that','this','these','those',
    'your','our','their','his','her','my','by','as','at','than','into','onto','over','under','about','above','below',
    'because','before','after','between','within','during','until','since','against','among','per','via','are','is',
    'be','been','being','was','were','not','no','do','does','did','will','would','can','could','should','may','might',
    'must','if','then','so','such','other','another','each','every','some','any','more','most','many','much','few',
    'several','both','either','neither','own','same','too','very','just','even','also','less','least','again',
    'further','up','down','out','off','here','there','where','when','why','how','one','two','three','four','five',
    'six','seven','eight','nine','ten'}
_SPLIT_PATS = tuple(re.compile(rf'([a-z]{{2,}})({re.escape(fw)})([a-z]{{2,}})', re.IGNORECASE) for fw in _FUNC_WORDS)

def _tokenize(text: str):
    text = unicodedata.normalize("NFKC", text)
    text = _normalize_unicode_roman(text)
    text = re.sub(r"([A-Za-z])-\s*\n\s*([A-Za-z])", r"\1\2\n", text)
    text = re.sub(r"([A-Za-z])\s*\n\s*([A-Za-z])",   r"\1 \2", text)
    text = (text.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("ﬃ", "ffi").replace("ﬄ", "ffl"))
    text = re.sub(r"_{2,}|[_]{1,}\d+|\b\d+_{1,}\b", " ", text)
    text = re.sub(r"(?<=\w)[\\/](?=\w)", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    _CONNECTORS = ('the','that','your','our','their','his','her','my','answer','answers','phrases','demand')
    for w in _CONNECTORS:
        pat = re.compile(rf'([A-Za-z])({w})([A-Za-z])', re.IGNORECASE)
        text = pat.sub(r'\1 \2 \3', text)
    text = re.sub(r"\b([IA])([a-z])", r"\1 \2", text)
    text = re.sub(r"\b([A-Za-z]{2,})([IVXLCDM]{1,4})\b", r"\1 \2", text)
    text = _preclean_text(text)

    raw = [m.group(0).lower() for m in _WORD_RE.finditer(text)]

    toks = []
    for w in raw:
        if (w.endswith("'s") or w.endswith("’s")) and len(w) > 2:
            base = w[:-2]; 
            if base: toks.append(base)
            toks.append("'s")
        elif (w.endswith("s’") or w.endswith("s'")) and len(w) > 2:
            base = w[:-2]; 
            if base: toks.append(base)
            toks.append("'s")
        elif w in {"’s", "'s"}:
            toks.append("'s")
        else:
            toks.append(w)

    fixed = []
    for w in toks:
        if (w.endswith("’ll") or w.endswith("'ll")) and len(w) > 3:
            base = w[:-3]
            if base in ALWAYS_CAP:
                fixed.append(base); continue
        fixed.append(w)
    toks = fixed

    stitched = []
    i = 0
    while i < len(toks):
        w1 = toks[i]
        w2 = toks[i+1] if i+1 < len(toks) else None
        w3 = toks[i+2] if i+2 < len(toks) else None

        merged = False
        if w2:
            cand2 = (w1 + w2)
            if 2 <= len(w1) <= 10 and 1 <= len(w2) <= 10 and cand2 in _ec_dict:
                stitched.append(cand2); i += 2; merged = True
            elif w3:
                cand3 = (w1 + w2 + w3)
                if 2 <= len(w1) <= 10 and 1 <= len(w2) <= 10 and 1 <= len(w3) <= 10 and cand3 in _ec_dict:
                    stitched.append(cand3); i += 3; merged = True
        if merged: 
            continue

        if len(w1) == 1 and w1.isalpha() and w2:
            cand = (w1 + w2)
            if cand in _ec_dict and len(cand) >= 4:
                stitched.append(cand); i += 2; continue
            if w1 not in {"a", "i"} and len(w2) >= 2 and w2[0].isalpha() and cand in _ec_dict:
                stitched.append(cand); i += 2; continue

        stitched.append(w1); i += 1

    SMALL_FWS = {'to','the','from','your','our','their','his','her','my','and','or','of','in','on','for'}

    def _pair_fix(seq):
        out = []; i = 0
        while i < len(seq):
            w = seq[i]; nxt = seq[i+1] if i+1 < len(seq) else None
            if nxt:
                if w == "the" and nxt == "re":
                    out.append("there"); i += 2; continue
                if w == "int" and nxt == "his":
                    out.extend(["in", "this"]); i += 2; continue
                if w == "on" and nxt == "eof":
                    out.extend(["one", "of"]); i += 2; continue
            _FW, _ART = ("or","for","and","in","on","to"), ("a","an")
            fixed = False
            for fw in _FW:
                for art in _ART:
                    if w == (fw + art):
                        out.extend([fw, art]); i += 1; fixed = True; break
                if fixed: break
            if fixed: continue
            out.append(w); i += 1
        return out

    stitched = _pair_fix(stitched)

    _CHAP_WORDS = ("part","section","chapter","figure","table","question")
    _ROMAN_TAIL = r"(i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii|xiii|xiv|xv|xvi|xvii|xviii|xix|xx)"
    chap_pat = re.compile(rf"^({'|'.join(_CHAP_WORDS)}){_ROMAN_TAIL}$", re.IGNORECASE)
    def _chapter_roman_fix(seq):
        out = []
        for w in seq:
            m = chap_pat.match(w)
            if m: out.extend([m.group(1).lower(), m.group(2).lower()])
            else: out.append(w)
        return out
    stitched = _chapter_roman_fix(stitched)

    # === 轻量救援：缺首/缺尾一字；并包含少量高频 OCR 伪词直映射 ===
    HOT_PREFIX = ('a','u','t','s','c','e','h','o','r','p','m','b','d','g','l','n','f','i')
    HEAD_MISS_MAP = {
        "nyone":"anyone","fter":"after","ntil":"until","mong":"among","ecause":"because",
        "etween":"between","hese":"these","hose":"those","here":"there","hich":"which",
        "ver":"over","eople":"people","nstead":"instead","gain":"again","nother":"another",
        "niversity":"university","ction":"section","uestion":"question","rogram":"program",
        "eview":"review","nternet":"internet","telge":"college"  # 针对常见 OCR 错词
    }
    TAIL_TRY = ("e","y","d","t","n","r","l","g")

    def _rescue_head_tail(seq):
        out = []
        for idx, w in enumerate(seq):
            wl = w.lower()
            if wl.isalpha() and 4 <= len(wl) <= 10 and wl not in _ec_dict:
                direct = HEAD_MISS_MAP.get(wl)
                if direct and direct in _ec_dict:
                    out.append(direct); continue
                fixed = None
                for ch in HOT_PREFIX:
                    cand = ch + wl
                    if cand in _ec_dict:
                        fixed = cand; break
                if fixed:
                    out.append(fixed); continue
                for ch in TAIL_TRY:
                    cand2 = wl + ch
                    if cand2 in _ec_dict:
                        out.append(cand2); break
                else:
                    out.append(wl)
            else:
                out.append(wl)
        return out
    stitched = _rescue_head_tail(stitched)
    # === 轻量救援结束 ===

    rescued = []
    i = 0
    while i < len(stitched):
        t = stitched[i].lower(); n = len(stitched)
        handled = False
        for fw in SMALL_FWS:
            if t.startswith(fw) and 1 <= len(t) - len(fw) <= 3:
                tail = t[len(fw):]
                if i + 1 < n and (tail + str(stitched[i+1]).lower()) in _ec_dict:
                    rescued.append(fw); rescued.append(tail + str(stitched[i+1]).lower()); i += 2
                else:
                    rescued.append(fw); i += 1
                handled = True; break
        if handled: continue
        for fw in SMALL_FWS:
            if t.endswith(fw) and 1 <= len(t) - len(fw) <= 4:
                pre = t[:-len(fw)]
                if rescued and (rescued[-1] + pre) in _ec_dict:
                    rescued[-1] = (rescued[-1] + pre)
                elif i + 1 < n and (pre + str(stitched[i+1]).lower()) in _ec_dict:
                    rescued.append(pre + str(stitched[i+1]).lower()); i += 1
                rescued.append(fw); i += 1
                handled = True; break
        if handled: continue
        if t == "or" and i + 1 < n and str(stitched[i+1]).lower() in {"example", "instance"}:
            rescued.append("for"); i += 1; continue
        rescued.append(t); i += 1
    stitched = rescued

    def _split_by_func_words(s: str) -> str:
        # 改为使用全局预编译的 _SPLIT_PATS，避免在循环里反复 compile
        changed = True
        while changed:
            changed = False
            for pat in _SPLIT_PATS:
                new_s, n = pat.subn(r'\1 \2 \3', s)
                if n > 0:
                    s = new_s; changed = True
        return s

    from functools import lru_cache
    @lru_cache(maxsize=2048)
    def _greedy_dict_cut(s: str):
        sl = s.lower()
        if sl in _ec_dict or sl in _FUNC_WORDS:
            return [sl]
        if len(sl) < 6:
            return None
        for i in range(min(len(sl), 20), 1, -1):
            left = sl[:i]
            if (left in _ec_dict) or (left in _FUNC_WORDS):
                rest = sl[i:]
                if not rest:
                    return [left]
                cut_rest = _greedy_dict_cut(rest)
                if cut_rest:
                    return [left] + cut_rest
        return None

    repaired = []
    for w in stitched:
        wl = w.lower()
        if wl in _ec_dict or wl in {"'s"} or len(wl) <= 1:
            repaired.append(wl); continue
        s = _split_by_func_words(wl)
        parts = s.split()
        if len(parts) > 1:
            tmp = []
            for p in parts:
                if (p in _ec_dict) or (p in _FUNC_WORDS) or (len(p) <= 1):
                    tmp.append(p)
                else:
                    cut = _greedy_dict_cut(p); tmp.extend(cut if cut else [p])
            repaired.extend(tmp); continue
        cut = _greedy_dict_cut(wl)
        repaired.extend(cut if cut else [wl])

    allow_single = {"a", "i", "v", "x"}
    ALLOW_2 = {'am','an','as','at','be','by','do','go','he','if','in','is','it','me','my','no','of','on','or','so','to','up','us','we'}
    NOISE_2 = {'ou','kf','rw','th','ei','ab','ac','ad','ba','bc','bd','ca','cb','cd','da','db','dc'}
    NOISE_3 = {'mor','ses','phr','eof'}
    NOISE_4 = {'toge','rang','gfrom'}

    def _is_consonant_only(s: str) -> bool:
        return re.fullmatch(r"[bcdfghjklmnpqrstvwxyz]+", s) is not None

    filtered = []
    for w in repaired:
        if len(w) == 1:
            if w in allow_single:
                filtered.append(w)
            continue
        if len(w) == 2:
            if w in NOISE_2: 
                continue
            if (w not in _ec_dict) and (w not in ALLOW_2) and (w not in _FUNC_WORDS):
                continue
            filtered.append(w); continue
        if len(w) == 3:
            if w in NOISE_3:
                continue
            if (w not in _ec_dict) and (w not in _FUNC_WORDS):
                vowels = len(re.findall(r"[aeiou]", w)); uniq = len(set(w))
                if _is_consonant_only(w) or vowels <= 1 or uniq <= 2:
                    continue
            filtered.append(w); continue
        if len(w) == 4 and w in NOISE_4:
            continue
        if 4 <= len(w) <= 5 and (w not in _ec_dict) and (w not in _FUNC_WORDS):
            vowels = len(re.findall(r"[aeiou]", w)); uniq = len(set(w))
            if _is_consonant_only(w) or vowels <= 1 or uniq <= 2:
                continue
        filtered.append(w)
    return filtered

# ---------- 显示与释义清洗（保持不变） ----------
_POS_TAGS = ["n","vi","vt","v","adj","a","adv","art","pron","prep","conj","int","num","aux","abbr","pref","suff"]

def _normalize_pos_display(line: str) -> str:
    return re.sub(r"(^|\n)\s*a\.", r"\1adj.", line)

def _format_zh(zh: str) -> str:
    if not zh: return ""
    s = zh.replace("\\n", "\n").replace("\r", "\n").replace("\\r", "")
    s = re.sub(r"\n+", "\n", s)
    t = re.sub(r"\s*((?:" + "|".join(_POS_TAGS) + r")\.)", r"\n\1", s)
    lines = [ln.strip() for ln in t.split("\n")]
    out, seen = [], set()
    for ln in lines:
        if not ln: continue
        ln = _normalize_pos_display(ln)
        if ln in seen: continue
        seen.add(ln); out.append(ln)
    return "\n".join(out)

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

def _display_casing(word: str) -> str:
    if word == "i": return "I"
    if word in ROMAN_MAP: return word.upper()
    if word in DETECTED_PROPER or word in ALWAYS_CAP: return word[:1].upper() + word[1:]
    if word in MONTHS or word in WEEKDAYS: return word[:1].upper() + word[1:]
    return word

def _plural_fallback(word: str, base_zh_lookup) -> str:
    if not word or len(word) < 2: return ""
    cand = []
    if word.endswith("ies") and len(word) > 3: cand.append(word[:-3] + "y")
    if word.endswith("es")  and len(word) > 2: cand.append(word[:-2])
    if word.endswith("s")   and len(word) > 1: cand.append(word[:-1])
    for b in cand:
        zh = base_zh_lookup.get(b, "")
        if zh:
            return f"{_format_zh(zh)}（复数）"
    return ""

_SUFFIXES = [("ing",""),("ed",""),("er",""),("est",""),("ly",""),("tion","te"),("sion","de"),
             ("ment",""),("ness",""),("able",""),("ible",""),("al",""),("ous",""),("ive",""),("ize",""),("ise","")]
_ABBR_RE = re.compile(r"^[A-Z]{2,}$")
_ORD_RE = re.compile(r"^\d+(st|nd|rd|th)$", re.IGNORECASE)
_MODEL1_RE = re.compile(r"^\d+[a-z]+$", re.IGNORECASE)
_MODEL2_RE = re.compile(r"^[a-z]+\d+$", re.IGNORECASE)

def _fallback_guess(word: str, cur_zh: str) -> str:
    if cur_zh: return cur_zh
    wshow = _display_casing(word); wl = word.lower()
    if (wl in DETECTED_PROPER) or (word[:1].isupper() and wl not in _TITLECASE_STOP):
        return "专有名词（人名/地名/机构名）"
    if _ABBR_RE.match(word): return "abbr. 缩写，含义视上下文"
    if _ORD_RE.match(word): return "序数词"
    if _MODEL1_RE.match(word) or _MODEL2_RE.match(word): return "型号/代码"
    if word.isdigit(): return "数值/编号"
    if "-" in word:
        parts = [p for p in word.split("-") if p]
        zh_parts = []
        for p in parts:
            base = _ec_dict.get(p.lower(), "")
            if not base:
                if p[:1].isupper(): zh_parts.append("专有名词")
                elif _ABBR_RE.match(p): zh_parts.append("缩写")
                elif _MODEL1_RE.match(p) or _MODEL2_RE.match(p) or p.isdigit(): zh_parts.append("型号/编号")
                else:
                    done = False
                    for suf, rep in _SUFFIXES:
                        if p.lower().endswith(suf) and len(p) > len(suf)+1:
                            stem = (p[:-len(suf)] + rep).lower()
                            base2 = _ec_dict.get(stem, "")
                            if base2:
                                zh_parts.append(_format_zh(base2) + "（派生）"); done = True; break
                    if not done: zh_parts.append(p)
            else:
                zh_parts.append(_format_zh(base))
        return "-".join(zh_parts)
    for suf, rep in _SUFFIXES:
        if wl.endswith(suf) and len(wl) > len(suf)+1:
            stem = wl[:-len(suf)] + rep
            base = _ec_dict.get(stem, "")
            if base: return _format_zh(base) + "（派生）"
    plural_try = _plural_fallback(wl, _ec_dict)
    if plural_try: return plural_try
    return f"保留原词：{wshow}（暂缺词典释义）"

def _build_dataframe(tokens):
    ctr = Counter(tokens); first_pos = {}
    for idx, w in enumerate(tokens):
        if w not in first_pos: first_pos[w] = idx
    df = pd.DataFrame({"word": list(ctr.keys()), "count": [ctr[w] for w in ctr.keys()], "pos": [first_pos[w] for w in ctr.keys()]})
    df["zh"] = df["word"].str.lower().map(_ec_dict).fillna("")
    is_roman = df["word"].map(lambda w: w in ROMAN_MAP)
    df.loc[is_roman, "zh"] = df.loc[is_roman, "word"].map(lambda w: f"罗马数字{ROMAN_MAP[w]}")
    df["zh"] = df.apply(lambda r: MANUAL_CONTRACTIONS.get(r["word"], r["zh"]), axis=1).astype(str)
    mask_3d = df["word"].str.lower().eq("3d"); mask_2d = df["word"].str.lower().eq("2d")
    df.loc[mask_3d & df["zh"].eq(""), "zh"] = "abbr. 三维的（three dimensional）；三维（three dimensions）"
    df.loc[mask_2d & df["zh"].eq(""), "zh"] = "abbr. 2维（2 dimensional）"
    mask_nd = df["word"].str.match(r"^\d+d$", na=False)
    df.loc[mask_nd & df["zh"].eq(""), "zh"] = df.loc[mask_nd & df["zh"].eq(""), "word"].map(lambda w: f"abbr. {w[:-1]}维（{w[:-1]} dimensional）")
    mask_i = df["word"].eq("i"); df.loc[mask_i, "zh"] = "pron. 我\nn. 碘元素；字母I；罗马数字1"
    def _hyphen_zh(word: str, cur: str) -> str:
        if cur: return cur
        if "-" in word:
            parts = [p for p in word.split("-") if len(p) > 1]
            seg_zh = []
            for p in parts:
                z = _ec_dict.get(p.lower(), "")
                if not z and p.lower().endswith("s"):
                    if p.lower().endswith("ies") and len(p) > 3: z = _ec_dict.get(p[:-3].lower() + "y", "")
                    if not z and p.lower().endswith("es") and len(p) > 2: z = _ec_dict.get(p[:-2].lower(), "")
                    if not z and p.lower().endswith("s") and len(p) > 1: z = _ec_dict.get(p[:-1].lower(), "")
                if z: seg_zh.append(_format_zh(z))
            if seg_zh: return "-".join(seg_zh)
            return ""
        return cur
    df["zh"] = df.apply(lambda r: _hyphen_zh(r["word"], r["zh"]), axis=1)
    need_plural = df["zh"].eq("") & df["word"].str.endswith("s")
    df.loc[need_plural, "zh"] = df.loc[need_plural, "word"].map(lambda w: _plural_fallback(w, _ec_dict))
    df["zh"] = df["zh"].map(_format_zh)
    df["zh"] = df.apply(lambda r: _fallback_guess(r["word"], r["zh"]), axis=1)
    df["word"] = df["word"].map(_display_casing)
    df_freq = df.sort_values(["count", "word"], ascending=[False, True]).reset_index(drop=True)
    df_pos  = df.sort_values(["pos"]).reset_index(drop=True)
    return df_freq, df_pos

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 关键修复：有人直接 GET /upload（或刷新），统一带回首页，避免空白页
@app.get("/upload")
def upload_get_redirect():
    return RedirectResponse("/", status_code=303)

@app.head("/")
def root_head():
    return PlainTextResponse("", status_code=200)

@app.post("/upload")
async def upload(request: Request, file: UploadFile = File(...)):
    try:
        t0 = time.perf_counter()
        data = await file.read()
        t1 = time.perf_counter()
        text = _read_text_from_upload(file.filename, data)
        _detect_proper_nouns_from_text(text)
        t2 = time.perf_counter()
        tokens = _tokenize(text)
        t3 = time.perf_counter()
        if not tokens:
            raise ValueError("未解析到有效英文单词")
        df_freq, df_pos = _build_dataframe(tokens)
        t4 = time.perf_counter()
        sid = _get_sid_from_request(request) or _ensure_session()
        _put_session(sid, file.filename, df_freq, df_pos, page_size=STATE.get("page_size", 500))
        STATE["filename"] = file.filename; STATE["df_freq"] = df_freq; STATE["df_pos"] = df_pos
        resp = RedirectResponse(url="/result?sort=freq&page=1", status_code=303)
        resp.set_cookie("sid", sid, httponly=True, samesite="lax")
        print(f"[TIMING] read={t1-t0:.2f}s extract={t2-t1:.2f}s tokenize={t3-t2:.2f}s df={t4-t3:.2f}s")
        return resp
    except Exception as e:
        print(f"[ERROR] {e}")
        return PlainTextResponse("处理文件时出现错误，请确认文件无损坏或换一份测试。\n\nDETAILS: " + str(e), status_code=500)

def _slice_page(df: pd.DataFrame, page: int, page_size: int):
    total = len(df)
    pages = max(1, (total + page_size - 1) // page_size)
    page = max(1, min(page, pages))
    s = (page - 1) * page_size; e = min(s + page_size, total)
    sub = df.iloc[s:e].copy()
    sub.insert(0, "序号", range(s + 1, e + 1))
    sub = sub.rename(columns={"count": "出现次数", "word": "单词", "zh": "翻译"})
    sub = sub[["序号", "出现次数", "单词", "翻译"]]
    return sub, page, pages, total

@app.get("/result", response_class=HTMLResponse)
def result(request: Request, sort: str = Query("freq", pattern="^(freq|pos)$"), page: int = 1):
    sess = _get_session_state(request)
    df_freq = sess["df_freq"] if sess else STATE["df_freq"]
    df_pos  = sess["df_pos"] if sess else STATE["df_pos"]
    filename = sess["filename"] if sess else STATE["filename"]
    page_size = (sess or STATE).get("page_size", 500)
    if df_freq is None:
        return RedirectResponse("/", status_code=303)
    df = df_freq if sort == "freq" else df_pos
    sub, cur, pages, total = _slice_page(df, page, page_size)
    return templates.TemplateResponse("result.html", {"request": request, "filename": filename,
        "rows": sub.to_dict(orient="records"), "page": cur, "pages": pages, "total": total,
        "page_size": page_size, "sort": sort})

@app.get("/export")
def export(request: Request, sort: str = Query("freq", pattern="^(freq|pos)$")):
    sess = _get_session_state(request)
    df_freq = sess["df_freq"] if sess else STATE["df_freq"]
    df_pos  = sess["df_pos"] if sess else STATE["df_pos"]
    filename = sess["filename"] if sess else STATE["filename"]
    if df_freq is None:
        return RedirectResponse("/", status_code=303)
    df = df_freq if sort == "freq" else df_pos
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
def healthz():
    return {"status": "ok"}

@app.head("/healthz")
def healthz_head():
    return PlainTextResponse("", status_code=200)