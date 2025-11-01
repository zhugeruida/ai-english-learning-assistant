# -*- coding: utf-8 -*-
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

# ===================== 词典层：仅此区域增强，其它不动 =====================
import sqlite3
import threading
from collections import OrderedDict
from typing import Optional, Iterator, Dict, List

# —— 高频功能词/代词/助动词：内置覆盖，永不为空 —— #
FUNCTION_WORDS_ZH: Dict[str, str] = {
    # 冠/连/介
    "the":"art. （定冠词）这；那；该",
    "a":"art. （不定冠词）一；每一",
    "an":"art. （不定冠词，元音前）一",
    "and":"conj. 和；并且",
    "or":"conj. 或者；否则",
    "but":"conj. 但是；然而",
    "so":"conj./adv. 所以；如此",
    "for":"prep. 为了；对于；达（时间）",
    "of":"prep. ……的；属于；含有",
    "in":"prep. 在……里面；在……期间",
    "on":"prep. 在……上；关于",
    "to":"prep. 向；到；（不定式符号）",
    "from":"prep. 从；来自；由于",
    "with":"prep. 和；用；具有",
    "without":"prep. 没有；不",
    "by":"prep. 通过；被；在……旁",
    "as":"conj./prep. 作为；因为；如同",
    "at":"prep. 在；以；朝向",
    "than":"conj. 比；而不是",
    "into":"prep. 进入；变成",
    "onto":"prep. 到……之上",
    "over":"prep. 在……之上；超过",
    "under":"prep. 在……之下；小于",
    "about":"prep. 关于；大约",
    "above":"prep. 在……上方；超过",
    "below":"prep. 在……下方；低于",
    "before":"prep./conj. 在……之前",
    "after":"prep./conj. 在……之后",
    "between":"prep. 在……之间",
    "within":"prep. 在……之内",
    "during":"prep. 在……期间",
    "until":"prep./conj. 直到",
    "since":"prep./conj. 自从；因为",
    "against":"prep. 反对；靠；对着",
    "among":"prep. 在……之中",
    "per":"prep. 每；按照",
    "via":"prep. 通过；经由",
    "because":"conj. 因为",

    # 系/助/情态/完成
    "be":"v. 是；存在；成为",
    "am":"v. be 的第一人称单数现在时",
    "is":"v. be 的第三人称单数现在时",
    "are":"v. be 的复数现在时",
    "was":"v. be 的单数过去式",
    "were":"v. be 的复数过去式",
    "been":"v. be 的过去分词",
    "being":"n./v. 存在；正在……",
    "do":"v. 做；助动词",
    "does":"v. do 的第三人称单数现在时；助动词",
    "did":"v. do 的过去式；助动词",
    "done":"v. do 的过去分词",
    "have":"v. 有；完成助动词",
    "has":"v. have 的第三人称单数",
    "had":"v. have 的过去式/分词",
    "will":"modal. 将要；愿意",
    "would":"modal. 将会；愿",
    "can":"modal. 能；可以",
    "could":"modal. 能；可能",
    "should":"modal. 应该",
    "may":"modal. 可能；可以",
    "might":"modal. 可能",
    "must":"modal. 必须",

    # 指示/疑问/代词 等
    "i":"pron. 我", "you":"pron. 你；你们", "he":"pron. 他", "she":"pron. 她", "it":"pron. 它；这件事",
    "we":"pron. 我们", "they":"pron. 他们/它们", "me":"pron. 我（宾）", "him":"pron. 他（宾）", "her":"pron. 她（宾/形）",
    "us":"pron. 我们（宾）", "them":"pron. 他们/它们（宾）",
    "my":"det. 我的", "your":"det. 你的；你们的", "his":"det./pron. 他的", "its":"det./pron. 它的",
    "our":"det. 我们的", "their":"det. 他们/它们的",
    "this":"det./pron. 这；这个", "that":"det./pron. 那；那个",
    "these":"det./pron. 这些", "those":"det./pron. 那些",
    "who":"pron. 谁", "whom":"pron. 谁（宾）", "whose":"pron./det. 谁的",
    "which":"pron./det. 哪个；哪些", "what":"pron./det. 什么",
    "where":"adv. 哪里", "when":"adv. 何时", "why":"adv. 为什么", "how":"adv. 如何",
    "here":"adv. 这里", "there":"adv. 那里；存在句",
    "one":"num./pron. 一；一个", "other":"adj./pron. 其他的", "another":"det./pron. 另一个；再一个",
    "own":"adj. 自己的", "same":"adj. 相同的", "such":"det./pron. 这样的"
}

class SQLiteEcdict:
    """只读 SQLite 词典：COALESCE(translation, definition, collins, oxford)，小写匹配 + LRU。"""
    def __init__(self, db_path: str, table: str = "ecdict", word_col: str = "word",
                 zh_col: str = "translation", cache_size: int = 50000):
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

        self._val_cache = OrderedDict()   # word -> str 或 "__NONE__"
        self._exist_cache = OrderedDict() # word -> True/False
        self._rowcount = None

        cur = self._conn.execute(f"PRAGMA table_info({self.table})")
        cols = {row["name"].lower() for row in cur.fetchall()}
        if self.word_col.lower() not in cols:
            raise RuntimeError(f"SQLite 表缺少列：{self.word_col}")
        if self.zh_col.lower() not in cols:
            raise RuntimeError(f"SQLite 表缺少列：{self.zh_col}")

        # 允许的备用列
        self._fallback_cols: List[str] = [c for c in ("definition", "collins", "oxford") if c in cols]
        self._val_expr = "COALESCE(" + ", ".join(
            [f"NULLIF(TRIM({self.zh_col}), '')"] +
            [f"NULLIF(TRIM({c}), '')" for c in self._fallback_cols] +
            ["''"]
        ) + ")"

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
            cur = self._conn.execute(
                f"SELECT {self._val_expr} AS v FROM {self.table} WHERE LOWER({self.word_col})=? LIMIT 1",
                (key_lc,)
            )
            row = cur.fetchone()
            if row is None:
                return None
            return (row[0] or "").strip()
        except Exception:
            return None

    def get(self, key: str, default: str = "") -> str:
        if not key:
            return default
        k = key.strip().lower()
        cached = self._lru_get(self._val_cache, k)
        if cached is not None:
            return "" if cached == "__NONE__" else cached
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
        cached = self._lru_get(self._exist_cache, k)
        if cached is not None:
            return bool(cached)
        _ = self.get(k, "")
        cached2 = self._lru_get(self._exist_cache, k)
        return bool(cached2)


class CsvEcdict:
    """只读 CSV 词典（始终启用作为回退）。自动识别列，自动尝试 UTF-8/UTF-8-SIG/GBK。"""
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._data: Dict[str, str] = {}
        self._loaded = False

    def _try_read(self):
        errs = []
        for enc in ("utf-8", "utf-8-sig", "gbk"):
            for sep in (",", "\t"):
                try:
                    return pd.read_csv(self.csv_path, encoding=enc, sep=sep)
                except Exception as e:
                    errs.append((enc, sep, str(e)))
        # 最后一次不指定 sep 让 pandas 自行推断
        try:
            return pd.read_csv(self.csv_path, engine="python")
        except Exception:
            raise

    def _ensure_load(self):
        if self._loaded:
            return
        if not os.path.exists(self.csv_path):
            self._loaded = True
            return
        try:
            df = self._try_read()
        except Exception:
            self._loaded = True
            return
        cols = {c.lower(): c for c in df.columns}
        if "word" not in cols:
            self._loaded = True
            return
        # 允许的释义列（尽量覆盖常见命名）
        cand = [c for c in (
            "translation", "definition", "collins", "oxford",
            "zh", "cn", "mean", "meaning", "释义", "解释"
        ) if c in cols]
        if not cand:
            self._loaded = True
            return
        wcol = cols["word"]
        zhcol = cols[cand[0]]
        for w, z in zip(df[wcol].astype(str), df[zhcol].astype(str)):
            wl = (w or "").strip().lower()
            zv = (z or "").strip()
            if wl and zv:
                self._data[wl] = zv
        self._loaded = True

    def get(self, key: str, default: str = "") -> str:
        self._ensure_load()
        return self._data.get((key or "").strip().lower(), default)

    def __contains__(self, key: str) -> bool:
        self._ensure_load()
        return (key or "").strip().lower() in self._data

# —— 统一查词函数（先 SQLite，再 CSV，再功能词覆盖） —— #
DB_PATH = os.getenv("ECDICT_DB_PATH", "data/ecdict.sqlite3")
if not os.path.exists(DB_PATH):
    raise RuntimeError("缺少 data/ecdict.sqlite3（或设置 ECDICT_DB_PATH）。")

_sql = SQLiteEcdict(
    DB_PATH,
    table=os.getenv("ECDICT_TABLE", "ecdict"),
    word_col=os.getenv("ECDICT_WORD_COL", "word"),
    zh_col=os.getenv("ECDICT_ZH_COL", "translation"),
    cache_size=int(os.getenv("ECDICT_CACHE_SIZE", "50000")),
)

_csv_path = os.getenv("ECDICT_CSV_PATH", "data/ecdict.csv")
_csv = CsvEcdict(_csv_path) if os.path.exists(_csv_path) else None

def dict_lookup_raw(w: str) -> str:
    """返回原始中文释义；未命中返回空串。"""
    k = (w or "").strip().lower()
    if not k:
        return ""
    # 1) SQLite
    try:
        val = _sql.get(k, "")
        if val:
            return val
    except Exception:
        pass
    # 2) CSV
    if _csv is not None:
        try:
            val2 = _csv.get(k, "")
            if val2:
                return val2
        except Exception:
            pass
    # 3) 高频功能词覆盖
    if k in FUNCTION_WORDS_ZH:
        return FUNCTION_WORDS_ZH[k]
    return ""

# =================== 文件解析与分词（你已有的逻辑保留） ===================
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
    print("[BOOT] dict sources:", f"sqlite:{os.path.exists(DB_PATH)}",
          f"csv:{os.path.exists(_csv_path)}")

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

# ---------- 分词（保留你的一系列补丁） ----------
_WORD_RE = re.compile(
    r"(?:[A-Za-z]+(?:['’][A-Za-z]+)?)"
    r"|(?:\d+(?:[A-Za-z]+|[A-Za-z]*[\/\-][A-Za-z]+))"
)

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
            tf.write(data)
            tmp = tf.name
        try:
            d = docx.Document(tmp)
        finally:
            try:
                os.remove(tmp)
            except Exception:
                pass
        return "\n".join(p.text for p in d.paragraphs)[:MAX_CHARS]
    try:
        return data.decode("utf-8", errors="ignore")[:MAX_CHARS]
    except Exception:
        return ""

_URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s]+")
_EMAIL_RE = re.compile(r"(?i)\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_SUFFIX_FUNCS = r"(?:from|for|the|that|this|these|those|your|our|their|and|to|of|with|without)"
_PREFIX_FUNCS = r"(?:and|or|but|for|to|of|in|on|at|from|by|the|that|this|these|those|with|without)"

UNICODE_ROMAN_MAP = {
    "Ⅰ":"I","Ⅱ":"II","Ⅲ":"III","Ⅳ":"IV","Ⅴ":"V","Ⅵ":"VI","Ⅶ":"VII","Ⅷ":"VIII","Ⅸ":"IX","Ⅹ":"X",
    "Ⅺ":"XI","Ⅻ":"XII","Ⅼ":"L","Ⅽ":"C","Ⅾ":"D","Ⅿ":"M",
    "ⅰ":"i","ⅱ":"ii","ⅲ":"iii","ⅳ":"iv","ⅴ":"v","ⅵ":"vi","ⅶ":"vii","ⅷ":"viii","ⅸ":"ix","ⅹ":"x",
    "ⅺ":"xi","ⅻ":"xii","ⅼ":"l","ⅽ":"c","ⅾ":"d","ⅿ":"m"
}
def _normalize_unicode_roman(s: str) -> str:
    return "".join(UNICODE_ROMAN_MAP.get(ch, ch) for ch in s)

def _preclean_text(text: str) -> str:
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    WATERMARK_PAT = re.compile(r"(?i)\b(zjuxz|xuezhan|zju|zjuxz\.cn)\b")
    text = WATERMARK_PAT.sub(" ", text)
    text = re.sub(r"([A-Za-z])[,\uFF0C]\s*([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z])[.\u3002]\s*([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z])[;；]\s*([A-Za-z])",      r"\1 \2", text)
    text = re.sub(r"([A-Za-z])[:：]\s*([A-Za-z])",      r"\1 \2", text)
    text = re.sub(r"([A-Za-z])、\s*([A-Za-z])",         r"\1 \2", text)
    text = re.sub(rf"([A-Za-z])({_SUFFIX_FUNCS})\b", r"\1 \2", text, flags=re.IGNORECASE)
    text = re.sub(rf"\b({_PREFIX_FUNCS})([A-Za-z])", r"\1 \2", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text

def _tokenize(text: str):
    text = unicodedata.normalize("NFKC", text)
    text = _normalize_unicode_roman(text)
    text = re.sub(r"([A-Za-z])-\s*\n\s*([A-Za-z])", r"\1\2\n", text)
    text = re.sub(r"([A-Za-z])\s*\n\s*([A-Za-z])",   r"\1 \2", text)
    text = (text.replace("ﬁ","fi").replace("ﬂ","fl").replace("ﬃ","ffi").replace("ﬄ","ffl"))
    text = re.sub(r"_{2,}|[_]{1,}\d+|\b\d+_{1,}\b", " ", text)
    text = re.sub(r"(?<=\w)[\\/](?=\w)", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    _CONNECTORS = ('the','that','your','our','their','his','her','my','answer','answers','phrases','demand')
    for w in _CONNECTORS:
        text = re.sub(rf'([A-Za-z])({w})([A-Za-z])', r'\1 \2 \3', text, flags=re.IGNORECASE)
    text = re.sub(r"\b([IA])([a-z])", r"\1 \2", text)
    text = re.sub(r"\b([A-Za-z]{2,})([IVXLCDM]{1,4})\b", r"\1 \2", text)
    text = _preclean_text(text)

    raw = [m.group(0).lower() for m in _WORD_RE.finditer(text)]

    toks = []
    for w in raw:
        if (w.endswith("'s") or w.endswith("’s")) and len(w)>2:
            base = w[:-2]
            if base: toks.append(base)
            toks.append("'s")
        elif (w.endswith("s’") or w.endswith("s'")) and len(w)>2:
            base = w[:-2]
            if base: toks.append(base)
            toks.append("'s")
        elif w in {"’s","'s"}:
            toks.append("'s")
        else:
            toks.append(w)

    ALWAYS_CAP = {"washington","australia","gutenberg","tom","daisy","gatsby"}
    fixed = []
    for w in toks:
        if (w.endswith("’ll") or w.endswith("'ll")) and len(w)>3:
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
            if 2<=len(w1)<=10 and 1<=len(w2)<=10 and dict_lookup_raw(cand2):
                stitched.append(cand2); i+=2; merged=True
            elif w3:
                cand3 = (w1 + w2 + w3)
                if 2<=len(w1)<=10 and 1<=len(w2)<=10 and 1<=len(w3)<=10 and dict_lookup_raw(cand3):
                    stitched.append(cand3); i+=3; merged=True
        if merged: continue
        if len(w1)==1 and w1.isalpha() and w2 and w1 not in {"a","i"}:
            cand = (w1 + w2)
            if len(w2)>=2 and w2[0].isalpha() and dict_lookup_raw(cand):
                stitched.append(cand); i+=2; continue
        stitched.append(w1); i+=1

    SMALL_FWS = {'to','the','from','your','our','their','his','her','my','and','or','of','in','on','for'}
    def _pair_fix(seq):
        out=[]; i=0
        while i<len(seq):
            w = seq[i]; nxt = seq[i+1] if i+1<len(seq) else None
            if nxt:
                if w=="the" and nxt=="re":
                    out.append("there"); i+=2; continue
                if w=="int" and nxt=="his":
                    out.extend(["in","this"]); i+=2; continue
                if w=="on" and nxt=="eof":
                    out.extend(["one","of"]); i+=2; continue
            _FW, _ART = ("or","for","and","in","on","to"), ("a","an")
            fixed=False
            for fw in _FW:
                for art in _ART:
                    if w==(fw+art):
                        out.extend([fw,art]); i+=1; fixed=True; break
                if fixed: break
            if fixed: continue
            out.append(w); i+=1
        return out
    stitched = _pair_fix(stitched)

    _CHAP_WORDS = ("part","section","chapter","figure","table","question")
    _ROMAN_TAIL = r"(i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii|xiii|xiv|xv|xvi|xvii|xviii|xix|xx)"
    chap_pat = re.compile(rf"^({'|'.join(_CHAP_WORDS)}){_ROMAN_TAIL}$", re.IGNORECASE)
    def _chapter_roman_fix(seq):
        out=[]
        for w in seq:
            m = chap_pat.match(w)
            if m: out.extend([m.group(1).lower(), m.group(2).lower()])
            else: out.append(w)
        return out
    stitched = _chapter_roman_fix(stitched)

    rescued=[]; i=0
    while i<len(stitched):
        t = stitched[i].lower(); n = len(stitched); handled=False
        for fw in SMALL_FWS:
            if t.startswith(fw) and 1 <= len(t)-len(fw) <= 3:
                tail = t[len(fw):]
                if i+1<n and dict_lookup_raw(tail+stitched[i+1].lower()):
                    rescued.append(fw); rescued.append(tail+stitched[i+1].lower()); i+=2
                else:
                    rescued.append(fw); i+=1
                handled=True; break
        if handled: continue
        for fw in SMALL_FWS:
            if t.endswith(fw) and 1 <= len(t)-len(fw) <= 4:
                pre = t[:-len(fw)]
                if rescued and dict_lookup_raw(rescued[-1]+pre):
                    rescued[-1] = (rescued[-1]+pre)
                elif i+1<n and dict_lookup_raw(pre+stitched[i+1].lower()):
                    rescued.append(pre+stitched[i+1].lower()); i+=1
                rescued.append(fw); i+=1; handled=True; break
        if handled: continue
        if t=="or" and i+1<n and stitched[i+1].lower() in {"example","instance"}:
            rescued.append("for"); i+=1; continue
        rescued.append(t); i+=1
    stitched = rescued

    _FUNC_WORDS = {
        'to','of','on','in','for','from','with','without','and','or','but','that','this','these','those',
        'your','our','their','his','her','my','by','as','at','than','into','onto','over','under','about',
        'above','below','because','before','after','between','within','during','until','since','against',
        'among','per','via','are','is','be','been','being','was','were','not','no','do','does','did',
        'will','would','can','could','should','may','might','must','if','then','so','such','other',
        'another','each','every','some','any','more','most','many','much','few','several','both','either',
        'neither','own','same','too','very','just','even','also','less','least','again','further','up',
        'down','out','off','here','there','where','when','why','how','one','two','three','four','five',
        'six','seven','eight','nine','ten'
    }

    def _split_by_func_words(s: str) -> str:
        changed=True
        while changed:
            changed=False
            pat_list=[re.compile(rf'([a-z]{{2,}})({fw})([a-z]{{2,}})', re.IGNORECASE) for fw in _FUNC_WORDS]
            for pat in pat_list:
                new_s, n = pat.subn(r'\1 \2 \3', s)
                if n>0: s=new_s; changed=True
        return s

    from functools import lru_cache
    @lru_cache(maxsize=2048)
    def _greedy_dict_cut(s: str):
        sl=s.lower()
        if dict_lookup_raw(sl) or (sl in _FUNC_WORDS): return [sl]
        if len(sl)<6: return None
        for i in range(min(len(sl),20),1,-1):
            left=sl[:i]
            if dict_lookup_raw(left) or (left in _FUNC_WORDS):
                rest=sl[i:]
                if not rest: return [left]
                cut_rest=_greedy_dict_cut(rest)
                if cut_rest: return [left]+cut_rest
        return None

    repaired=[]
    for w in stitched:
        wl=w.lower()
        if dict_lookup_raw(wl) or wl in {"'s"} or len(wl)<=1:
            repaired.append(wl); continue
        s=_split_by_func_words(wl); parts=s.split()
        if len(parts)>1:
            tmp=[]
            for p in parts:
                if dict_lookup_raw(p) or (p in _FUNC_WORDS) or (len(p)<=1):
                    tmp.append(p)
                else:
                    cut=_greedy_dict_cut(p); tmp.extend(cut if cut else [p])
            repaired.extend(tmp); continue
        cut=_greedy_dict_cut(wl); repaired.extend(cut if cut else [wl])

    allow_single={"a","i","v","x"}
    ALLOW_2={'am','an','as','at','be','by','do','go','he','if','in','is','it','me','my','no','of','on','or','so','to','up','us','we'}
    NOISE_2={'ou','kf','rw','th','ei','ab','ac','ad','ba','bc','bd','ca','cb','cd','da','db','dc'}
    NOISE_3={'mor','ses','phr','eof'}
    NOISE_4={'toge','rang','gfrom'}
    def _is_consonant_only(s: str)->bool:
        return re.fullmatch(r"[bcdfghjklmnpqrstvwxyz]+", s) is not None

    filtered=[]
    for w in repaired:
        if len(w)==1:
            if w in allow_single: filtered.append(w)
            continue
        if len(w)==2:
            if w in NOISE_2: continue
            if (not dict_lookup_raw(w)) and (w not in ALLOW_2) and (w not in _FUNC_WORDS): continue
            filtered.append(w); continue
        if len(w)==3:
            if w in NOISE_3: continue
            if (not dict_lookup_raw(w)) and (w not in _FUNC_WORDS):
                vowels=len(re.findall(r"[aeiou]", w)); uniq=len(set(w))
                if _is_consonant_only(w) or vowels<=1 or uniq<=2: continue
            filtered.append(w); continue
        if len(w)==4 and w in NOISE_4: continue
        if 4<=len(w)<=5 and (not dict_lookup_raw(w)) and (w not in _FUNC_WORDS):
            vowels=len(re.findall(r"[aeiou]", w)); uniq=len(set(w))
            if _is_consonant_only(w) or vowels<=1 or uniq<=2: continue
        filtered.append(w)
    return filtered

# ---------- 显示与释义清洗 ----------
_POS_TAGS = ["n", "vi", "vt", "v", "adj", "a", "adv", "art", "pron", "prep", "conj", "int", "num", "aux", "abbr", "pref", "suff"]
def _normalize_pos_display(line: str) -> str:
    return re.sub(r"(^|\n)\s*a\.", r"\1adj.", line)

def _format_zh(zh: str) -> str:
    if not zh: return ""
    s = zh.replace("\\n","\n").replace("\r","\n").replace("\\r","")
    s = re.sub(r"\n+","\n", s)
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
    "don't":"aux. 表示否定（= do not）","don’t":"aux. 表示否定（= do not）",
    "didn't":"aux. 不（= did not）","didn’t":"aux. 不（= did not）",
    "isn't":"不是（= is not）","isn’t":"不是（= is not）",
    "hasn't":"没有（= has not）","hasn’t":"没有（= has not）",
    "haven't":"没有（= have not）","haven’t":"没有（= have not）",
    "hadn't":"没有（= had not）","hadn’t":"没有（= had not）",
    "won't":"不会（= will not）","won’t":"不会（= will not）",
    "wouldn't":"不会；不愿（= would not）","wouldn’t":"不会；不愿（= would not）",
    "can't":"不能（= cannot）","can’t":"不能（= cannot）",
    "couldn't":"不能（= could not）","couldn’t":"不能（= could not）",
    "shouldn't":"不应该（= should not）","shouldn’t":"不应该（= should not）",
    "doesn't":"aux. 表示否定（= does not）","doesn’t":"aux. 表示否定（= does not）",

    "i'm":"abbr. 我是（= I am）","i’m":"abbr. 我是（= I am）",
    "i've":"contr. 我已/我有（= I have）","i’ve":"contr. 我已/我有（= I have）",
    "i'd":"contr. 我愿意/我已经/我应该（= I would / I had / I should）","i’d":"contr. 我愿意/我已经/我应该（= I would / I had / I should）",
    "i'll":"contr. 我将/我会（= I will / I shall）","i’ll":"contr. 我将/我会（= I will / I shall）",

    "you're":"contr. 你是（= you are）","you’re":"contr. 你是（= you are）",
    "you've":"contr. 你已/你有（= you have）","you’ve":"contr. 你已/你有（= you have）",
    "you'll":"contr. 你将/你会（= you will / you shall）","you’ll":"contr. 你将/你会（= you will / you shall）",

    "we're":"contr. 我们是（= we are）","we’re":"contr. 我们是（= we are）",
    "we've":"contr. 我们已/我们有（= we have）","we’ve":"contr. 我们已/我们有（= we have）",
    "we'll":"contr. 我们将/我们会（= we will / we shall）","we’ll":"contr. 我们将/我们会（= we will / we shall）",

    "they're":"contr. 他们是（= they are）","they’re":"contr. 他们是（= they are）",
    "they've":"contr. 他们已/他们有（= they have）","they’ve":"contr. 他们已/他们有（= they have）",
    "they'll":"contr. 他们将/他们会（= they will / they shall）","they’ll":"contr. 他们将/他们会（= they will / they shall）",

    "he's":"contr. 他是/他有（= he is / he has）","he’s":"contr. 他是/他有（= he is / he has）",
    "he'd":"contr. 他愿意/他已经/他应该（= he would / he had / he should）","he’d":"contr. 他愿意/他已经/他应该（= he would / he had / he should）",
    "he'll":"contr. 他将/他会（= he will / he shall）","he’ll":"contr. 他将/他会（= he will / he shall）",

    "she's":"contr. 她是/她有（= she is / she has）","she’s":"contr. 她是/她有（= she is / she has）",
    "she'd":"contr. 她愿意/她已经/她应该（= she would / she had / she should）","she’d":"contr. 她愿意/她已经/她应该（= she would / she had / she should）",
    "she'll":"contr. 她将/她会（= she will / she shall）","she’ll":"contr. 她将/她会（= she will / she shall）",

    "it's":"contr. 它是/它有（= it is / it has）","it’s":"contr. 它是/它有（= it is / it has）",
    "it'll":"contr. 它将/它会（= it will）","it’ll":"contr. 它将/它会（= it will）",
    "it'd":"contr. 它将会/它本会（= it would / it had）","it’d":"contr. 它将会/它本会（= it would / it had）",

    "that's":"那是；正是（= that is / that has）","that’s":"那是；正是（= that is / that has）",
    "there's":"那里有；有（= there is / there has）","there’s":"那里有；有（= there is / there has）",
    "here's":"这儿是；给你（= here is）","here’s":"这儿是；给你（= here is）",
    "what's":"什么是（= what is / what has）","what’s":"什么是（= what is / what has）",
    "who's":"谁是（= who is / who has）","who’s":"谁是（= who is / who has）",
    "where's":"哪里是/哪里有（= where is / where has）","where’s":"哪里是/哪里有（= where is / where has）",
    "how's":"情况如何（= how is / how has）","how’s":"情况如何（= how is / how has）",

    "let's":"让我们…（= let us）","let’s":"让我们…（= let us）",
    "o'clock":"abbr. ……点钟","o’clock":"abbr. ……点钟",
}

MONTHS = {"january","february","march","april","may","june","july","august","september","october","november","december"}
WEEKDAYS = {"monday","tuesday","wednesday","thursday","friday","saturday","sunday"}
ALWAYS_CAP = {"washington","australia","gutenberg","tom","daisy","gatsby"}

ROMAN_MAP = {"i":1,"ii":2,"iii":3,"iv":4,"v":5,"vi":6,"vii":7,"viii":8,"ix":9,"x":10,"xi":11,"xii":12,"xiii":13,"xiv":14,"xv":15,"xvi":16,"xvii":17,"xviii":18,"xix":19,"xx":20}

DETECTED_PROPER = set()
_TITLECASE_STOP = {
    "the","a","an","and","but","or","nor","for","so","yet",
    "when","where","what","who","whom","whose","why","which","while",
    "if","in","on","at","to","of","from","with","without","into","onto","over","under","about","above","below",
    "he","she","it","they","we","you","i","his","her","its","their","our","your",
    "this","that","these","those","there","here","then","thus","hence","once",
    "directions","dialogue","text","paper","okay","oh","uh",
    "phrases","answers","choices","passage","section","part"
}
_COMMON_LOWER_STOP = {
    "one","two","three","four","five","six","seven","eight","nine","ten",
    "first","second","third","fourth","fifth","sixth","seventh","eighth","ninth","tenth",
    "part","section","passage","dialogue","answer","sheet","points","choice","choices",
    "men","women","time","more","from","about","my","our","your",
}

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
            if wl in _TITLECASE_STOP or wl in _COMMON_LOWER_STOP or wl in _MONTHS_DAYS or wl == "i":
                continue
            DETECTED_PROPER.add(wl)

def _display_casing(word: str) -> str:
    if word == "i": return "I"
    if word in ROMAN_MAP: return word.upper()
    if word in DETECTED_PROPER or word in ALWAYS_CAP: return word[:1].upper() + word[1:]
    if word in MONTHS or word in WEEKDAYS: return word[:1].upper() + word[1:]
    return word

# —— 形态派生：在查词前“真回退”一次 —— #
_SUFFIXES = [
    ("ies", "y"), ("es",""), ("s",""),
    ("ing",""), ("ed",""), ("er",""), ("est",""), ("ly",""),
    ("tion","te"), ("sion","de"), ("ment",""), ("ness",""),
    ("able",""), ("ible",""), ("al",""), ("ous",""), ("ive",""),
    ("ize",""), ("ise",""),
]

def _morph_lookup(word: str) -> str:
    """先原词查词典，若空，再做词形还原尝试（逐条查词典，不是写文案）。"""
    wl = (word or "").strip().lower()
    if not wl:
        return ""
    val = dict_lookup_raw(wl)
    if val:
        return val
    for suf, rep in _SUFFIXES:
        if wl.endswith(suf) and len(wl) > len(suf) + 1:
            stem = wl[:-len(suf)] + rep
            val2 = dict_lookup_raw(stem)
            if val2:
                return val2  # 真命中
    return ""

def _plural_fallback(word: str) -> str:
    wl = (word or "").strip().lower()
    if not wl or len(wl) < 2:
        return ""
    cand=[]
    if wl.endswith("ies") and len(wl)>3: cand.append(wl[:-3]+"y")
    if wl.endswith("es")  and len(wl)>2: cand.append(wl[:-2])
    if wl.endswith("s")   and len(wl)>1: cand.append(wl[:-1])
    for b in cand:
        val = dict_lookup_raw(b)
        if val:
            return _format_zh(val) + "（复数）"
    return ""

_ABBR_RE   = re.compile(r"^[A-Z]{2,}$")
_ORD_RE    = re.compile(r"^\d+(st|nd|rd|th)$", re.IGNORECASE)
_MODEL1_RE = re.compile(r"^\d+[a-z]+$", re.IGNORECASE)
_MODEL2_RE = re.compile(r"^[a-z]+\d+$", re.IGNORECASE)

def _fallback_guess(word: str, cur_zh: str) -> str:
    if cur_zh:
        return cur_zh
    wshow = _display_casing(word)
    wl = word.lower()
    if (wl in DETECTED_PROPER) or (word[:1].isupper() and wl not in _TITLECASE_STOP):
        return "专有名词（人名/地名/机构名）"
    if _ABBR_RE.match(word):
        return "abbr. 缩写，含义视上下文"
    if _ORD_RE.match(word):
        return "序数词"
    if _MODEL1_RE.match(word) or _MODEL2_RE.match(word):
        return "型号/代码"
    if word.isdigit():
        return "数值/编号"
    if "-" in word:
        parts=[p for p in word.split("-") if p]; seg=[]
        for p in parts:
            val = _morph_lookup(p)
            if val:
                seg.append(_format_zh(val))
        if seg:
            return "-".join(seg)
    # 最终兜底
    return f"保留原词：{wshow}（暂缺词典释义）"

def _lookup_word_for_map(w: str) -> str:
    """提供给 pandas.map 的函数式查词：SQLite→CSV→功能词→形态→复数。"""
    if w == "'s":
        return "…的"
    # 原词/形态派生
    zh = _morph_lookup(w)
    if not zh and w.endswith("s"):
        zh = _plural_fallback(w)
    if not zh and w in FUNCTION_WORDS_ZH:
        zh = FUNCTION_WORDS_ZH[w]
    return _format_zh(zh)

# ================== 构建 DataFrame（其余流程保持不动） ==================
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

    # ① 用函数式 map（避免 pandas 对自定义 dict-like 的兼容差异）
    df["zh"] = df["word"].str.lower().map(_lookup_word_for_map).fillna("")

    # ② 罗马数字
    is_roman = df["word"].map(lambda w: w in ROMAN_MAP)
    df.loc[is_roman, "zh"] = df.loc[is_roman, "word"].map(lambda w: f"罗马数字{ROMAN_MAP[w]}")

    # ③ 缩略词兜底
    df["zh"] = df.apply(lambda r: MANUAL_CONTRACTIONS.get(r["word"], r["zh"]), axis=1).astype(str)

    # ③.1 数字+d
    mask_3d = df["word"].str.lower().eq("3d")
    mask_2d = df["word"].str.lower().eq("2d")
    df.loc[mask_3d & df["zh"].eq(""), "zh"] = "abbr. 三维的（three dimensional）；三维（three dimensions）"
    df.loc[mask_2d & df["zh"].eq(""), "zh"] = "abbr. 2维（2 dimensional）"
    mask_nd = df["word"].str.match(r"^\d+d$", na=False)
    df.loc[mask_nd & df["zh"].eq(""), "zh"] = df.loc[mask_nd & df["zh"].eq(""), "word"].map(
        lambda w: f"abbr. {w[:-1]}维（{w[:-1]} dimensional）"
    )

    # ④ special: i / I
    mask_i = df["word"].eq("i")
    df.loc[mask_i, "zh"] = "pron. 我\nn. 碘元素；字母I；罗马数字1"

    # ⑤ 连字符词再尝试片段拼译
    def _hyphen_zh(word: str, cur: str) -> str:
        if cur: return cur
        if "-" in word:
            parts = [p for p in word.split("-") if len(p) > 1]
            seg_zh = []
            for p in parts:
                v = _lookup_word_for_map(p)
                if v:
                    seg_zh.append(v)
            if seg_zh:
                return "-".join(seg_zh)
            return ""
        return cur
    df["zh"] = df.apply(lambda r: _hyphen_zh(r["word"], r["zh"]), axis=1)

    # ⑥ 仍为空的，用最终兜底（尽量少见）
    df["zh"] = df.apply(lambda r: _fallback_guess(r["word"], r["zh"]), axis=1)

    # ⑦ 显示层大小写
    df["word"] = df["word"].map(_display_casing)

    df_freq = df.sort_values(["count", "word"], ascending=[False, True]).reset_index(drop=True)
    df_pos  = df.sort_values(["pos"]).reset_index(drop=True)
    return df_freq, df_pos

# ------------------ 页面与 API（保持不变） ------------------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.head("/")
def root_head():
    return PlainTextResponse("", status_code=200)

@app.post("/upload")
async def upload(request: Request, file: UploadFile = File(...)):
    try:
        t0 = time.perf_counter()
        print(f"[TIMER] start upload {file.filename}")

        data = await file.read()
        t1 = time.perf_counter()
        print(f"[TIMER] read bytes: {t1 - t0:.3f}s")

        text = _read_text_from_upload(file.filename, data)
        _detect_proper_nouns_from_text(text)
        t2 = time.perf_counter()
        print(f"[TIMER] extract text: {t2 - t1:.3f}s")

        tokens = _tokenize(text)
        t3 = time.perf_counter()
        print(f"[TIMER] tokenize: {t3 - t2:.3f}s, tokens={len(tokens)}")

        if not tokens:
            raise ValueError("未解析到有效英文单词")

        df_freq, df_pos = _build_dataframe(tokens)
        t4 = time.perf_counter()
        print(f"[TIMER] build dataframe: {t4 - t3:.3f}s")

        sid = _get_sid_from_request(request) or _ensure_session()
        _put_session(sid, file.filename, df_freq, df_pos, page_size=STATE.get("page_size", 500))

        STATE["filename"] = file.filename
        STATE["df_freq"] = df_freq
        STATE["df_pos"] = df_pos

        print(f"[TIMER] total: {time.perf_counter() - t0:.3f}s")

        resp = RedirectResponse(url="/result?sort=freq&page=1", status_code=303)
        resp.set_cookie("sid", sid, httponly=True, samesite="lax")
        return resp

    except Exception as e:
        print(f"[ERROR] {e}")
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
    sess = _get_session_state(request)
    df_freq = sess["df_freq"] if sess else STATE["df_freq"]
    df_pos  = sess["df_pos"] if sess else STATE["df_pos"]
    filename = sess["filename"] if sess else STATE["filename"]
    page_size = (sess or STATE).get("page_size", 500)

    if df_freq is None:
        return RedirectResponse("/", status_code=303)

    df = df_freq if sort == "freq" else df_pos
    sub, cur, pages, total = _slice_page(df, page, page_size)
    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "filename": filename,
            "rows": sub.to_dict(orient="records"),
            "page": cur,
            "pages": pages,
            "total": total,
            "page_size": page_size,
            "sort": sort,
        }
    )

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
    docname  = (filename or "文档").rsplit(".", 1)[0]
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
    return {"status": "ok"}

@app.head("/healthz")
def healthz_head():
    return PlainTextResponse("", status_code=200)