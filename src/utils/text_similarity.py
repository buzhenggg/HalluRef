"""文本相似度计算工具"""

from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz


def normalize_text(text: str) -> str:
    """文本归一化: 小写、去多余空格、Unicode NFKD"""
    text = unicodedata.normalize("NFKD", text)
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    # 去除标点
    text = re.sub(r'[^\w\s]', '', text)
    return text


def title_similarity(a: str, b: str) -> float:
    """计算两个标题的相似度 (0-1)

    使用 token_sort_ratio 对词序不敏感。
    """
    na, nb = normalize_text(a), normalize_text(b)
    if not na or not nb:
        return 0.0
    return fuzz.token_sort_ratio(na, nb) / 100.0


def venue_similarity(a: str, b: str) -> float:
    """期刊/会议名相似度, 处理常见缩写"""
    _ABBREV_MAP = {
        "neurips": "neural information processing systems",
        "nips": "neural information processing systems",
        "icml": "international conference on machine learning",
        "iclr": "international conference on learning representations",
        "acl": "association for computational linguistics",
        "emnlp": "empirical methods in natural language processing",
        "naacl": "north american chapter of the association for computational linguistics",
        "cvpr": "computer vision and pattern recognition",
        "iccv": "international conference on computer vision",
        "eccv": "european conference on computer vision",
        "aaai": "association for the advancement of artificial intelligence",
        "ijcai": "international joint conference on artificial intelligence",
        "sigir": "special interest group on information retrieval",
        "www": "world wide web conference",
        "kdd": "knowledge discovery and data mining",
        "eacl": "european chapter of the association for computational linguistics",
    }

    na = normalize_text(a)
    nb = normalize_text(b)

    # 展开缩写再比
    na_expanded = _ABBREV_MAP.get(na, na)
    nb_expanded = _ABBREV_MAP.get(nb, nb)

    # 取最高分
    scores = [
        fuzz.token_sort_ratio(na, nb) / 100.0,
        fuzz.token_sort_ratio(na_expanded, nb_expanded) / 100.0,
    ]
    return max(scores)
