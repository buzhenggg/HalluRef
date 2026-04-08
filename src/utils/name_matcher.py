"""作者姓名模糊匹配工具"""

from __future__ import annotations

import re

from rapidfuzz import fuzz


def normalize_author_name(name: str) -> str:
    """归一化作者名: 去标点、统一空格、小写"""
    name = name.strip().lower()
    # 去掉 "Jr.", "III" 等后缀
    name = re.sub(r'\b(jr|sr|ii|iii|iv)\b\.?', '', name)
    # 去标点
    name = re.sub(r'[^\w\s]', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def _extract_surname(name: str) -> str:
    """提取姓氏 (假设最后一个词是姓, 或逗号前是姓)"""
    name = name.strip()
    if ',' in name:
        return name.split(',')[0].strip().lower()
    parts = name.split()
    return parts[-1].lower() if parts else name.lower()


def author_name_similarity(a: str, b: str) -> float:
    """计算两个作者名的相似度 (0-1)

    策略:
    1. 全名归一化后模糊匹配
    2. 姓氏精确匹配 + 名字首字母匹配 → 高分
    """
    na, nb = normalize_author_name(a), normalize_author_name(b)
    if not na or not nb:
        return 0.0

    # 全名模糊匹配
    full_score = fuzz.token_sort_ratio(na, nb) / 100.0

    # 姓氏匹配
    surname_a = _extract_surname(a)
    surname_b = _extract_surname(b)
    surname_score = fuzz.ratio(surname_a, surname_b) / 100.0

    # 如果姓完全一致, 给较高基础分
    if surname_score > 0.9:
        return max(full_score, 0.75)

    return full_score


def _is_et_al(authors: list[str]) -> bool:
    """判断作者列表是否为 'X et al.' 缩写形式"""
    if not authors:
        return False
    joined = " ".join(authors).lower()
    return "et al" in joined or len(authors) == 1


def _first_author_matches(claimed: list[str], actual: list[str], threshold: float = 0.75) -> bool:
    """检查第一作者姓氏是否匹配"""
    if not claimed or not actual:
        return False
    surname_c = _extract_surname(claimed[0])
    # 在 actual 作者列表中找第一作者姓氏匹配
    for a in actual:
        surname_a = _extract_surname(a)
        if fuzz.ratio(surname_c, surname_a) / 100.0 >= threshold:
            return True
    return False


def author_list_similarity(
    claimed: list[str],
    actual: list[str],
    threshold: float = 0.80,
) -> tuple[float, list[tuple[str, str | None]]]:
    """比较两个作者列表的相似度

    特殊处理:
    - 'X et al.' 缩写形式: 只要第一作者匹配即给 0.8 分 (PARTIAL)
    - 单作者 vs 多作者: 同上

    Returns:
        (score, matches) 其中 matches 是 [(claimed_name, matched_actual_name or None), ...]
    """
    if not claimed and not actual:
        return 1.0, []
    if not claimed or not actual:
        return 0.0, [(c, None) for c in claimed]

    # 特殊处理 "et al." 缩写 — LLM 常提取为 ["Vaswani et al."] 或 ["Vaswani"]
    if _is_et_al(claimed) and len(actual) > 1:
        if _first_author_matches(claimed, actual):
            return 0.8, [(claimed[0], actual[0])]
        else:
            return 0.2, [(claimed[0], None)]

    matches: list[tuple[str, str | None]] = []
    remaining = list(actual)

    for c_name in claimed:
        best_score = 0.0
        best_idx = -1
        for i, a_name in enumerate(remaining):
            score = author_name_similarity(c_name, a_name)
            if score > best_score:
                best_score = score
                best_idx = i

        if best_score >= threshold and best_idx >= 0:
            matches.append((c_name, remaining[best_idx]))
            remaining.pop(best_idx)
        else:
            matches.append((c_name, None))

    matched_count = sum(1 for _, m in matches if m is not None)
    total = max(len(claimed), len(actual))
    score = matched_count / total if total > 0 else 0.0

    return score, matches
