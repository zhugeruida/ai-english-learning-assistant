# nlp.py —— 统一分词器：支持数字+单位、保留 don't、过滤噪声单字
from collections import Counter
import io
import re
from typing import Counter as CounterType

# PDF
from pdfminer.high_level import extract_text as pdf_extract_text
# DOCX
from docx import Document

# -------- 分词器 --------
# 1) 英文单词，可带一次撇号（don't, it's, I'm）
# 2) 数字 + 单位（10kg, 6t, 60%, 3rd）；单位允许字母/百分号
# 3) 其它只字母串
TOKEN_PATTERN = re.compile(
    r"(?:[A-Za-z]+(?:'[A-Za-z]+)?)"         # don't / it's / I'm / books
    r"|(?:\d+(?:[A-Za-z%]+))"               # 10kg / 6t / 60%
)

# 需要保留的单字
KEEP_SINGLE = {"a", "i"}  # 注意：下游统一小写

def tokenize(text: str):
    # 抽取全部 token，统一小写
    toks = [m.group(0) for m in TOKEN_PATTERN.finditer(text)]
    toks = [t.lower() for t in toks]

    clean = []
    for t in toks:
        # 过滤非常见单字噪声（比如 s, t 等）
        if len(t) == 1 and t not in KEEP_SINGLE:
            continue

        # 过滤 possessive 残片（如 "'s" 被错误分出），我们的正则已避免，但加一层保险
        if t in {"'s", "’s"}:
            continue

        clean.append(t)
    return clean

# -------- 统计：返回 Counter --------
def _counts_from_text(text: str) -> CounterType[str]:
    tokens = tokenize(text)
    return Counter(tokens)

def pdf_token_counts(raw: bytes) -> CounterType[str]:
    # 直接从 bytes 解析文本
    text = pdf_extract_text(io.BytesIO(raw))
    return _counts_from_text(text or "")

def docx_token_counts(raw: bytes) -> CounterType[str]:
    bio = io.BytesIO(raw)
    doc = Document(bio)
    text = "\n".join(p.text for p in doc.paragraphs)
    return _counts_from_text(text or "")

# -------- Counter -> DataFrame（降序） --------
import pandas as pd

def counts_to_df(counts: CounterType[str]) -> pd.DataFrame:
    if not counts:
        return pd.DataFrame(columns=["word", "count"])
    items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    df = pd.DataFrame(items, columns=["word", "count"])
    return df
