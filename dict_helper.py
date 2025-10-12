# dict_helper.py  —— 纯本地 ECDICT 查询（零网络）+ 懒加载缓存
from __future__ import annotations
from typing import Dict, List
import os, re
import pandas as pd

_ECDICT_PATH = os.path.join("data", "ecdict.csv")
_DICT_DF: pd.DataFrame | None = None
_MAP: Dict[str, Dict[str, str]] | None = None

# 去掉“zk gk”等考试标签、奇怪标记
_TAG_JUNK = re.compile(r"\b(?:zk|gk|cet4|cet6|ky|gre|toefl|ielts|tem4|tem8)\b", re.I)

def _load_ecdict() -> None:
    global _DICT_DF, _MAP
    if _MAP is not None:
        return
    usecols = ["word", "translation", "tag"]
    _DICT_DF = pd.read_csv(
        _ECDICT_PATH, usecols=usecols, dtype=str,
        keep_default_na=False, low_memory=False, encoding="utf-8"
    )
    _DICT_DF["word"] = _DICT_DF["word"].str.lower()
    # 仅保留我们用到的列，建立 map（更快）
    _MAP = {}
    for w, zh, tag in _DICT_DF[["word", "translation", "tag"]].itertuples(index=False):
        if w not in _MAP:
            _MAP[w] = {
                "translation": zh or "",
                "tag": tag or "",
            }

def clean_translation(s: str) -> str:
    if not s:
        return ""
    s = _TAG_JUNK.sub("", s)              # 去掉考试标签
    s = s.replace("\r", "").replace("\t", " ")
    # ECDICT 里常见的“; ”换行、逗号等，合并为更易读的分号
    s = s.replace("\n", "；")
    return re.sub(r"[ ；]{2,}", "；", s).strip("； ").strip()

def split_pos_lines(tag: str) -> List[str]:
    """
    把 ECDICT 的 tag 拆成多行词性；去掉杂质，只保留像 n. / v. / adj. / adv. / pron. / prep. / conj. / art.
    """
    if not tag:
        return []
    tag = _TAG_JUNK.sub("", tag)
    # 一些字典会用空格或逗号分隔
    parts = re.split(r"[，,;/\s]+\s*", tag)
    keep = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # 只保留典型词性缩写
        if re.match(r"^(n|v|adj|adv|pron|prep|conj|art|num|int|aux|vi|vt)\.?$", p, re.I):
            keep.append(p.lower() + ".")
    # 去重但保留顺序
    seen = set()
    out = []
    for k in keep:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out

def lookup_word(word: str) -> Dict[str, str | List[str]]:
    _load_ecdict()
    key = (word or "").lower()
    rec = _MAP.get(key)
    if not rec:
        return {"zh": "", "pos_lines": []}
    zh = clean_translation(rec.get("translation", ""))
    pos_lines = split_pos_lines(rec.get("tag", ""))
    return {"zh": zh, "pos_lines": pos_lines}

def batch_lookup(words: List[str]) -> Dict[str, Dict[str, str | List[str]]]:
    _load_ecdict()
    out: Dict[str, Dict[str, str | List[str]]] = {}
    for w in words:
        out[w] = lookup_word(w)
    return out
