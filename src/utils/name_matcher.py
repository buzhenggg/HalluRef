"""Author-name fuzzy matching utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz

_SUFFIX_RE = re.compile(r"\b(jr|sr|ii|iii|iv)\b\.?", re.IGNORECASE)
_ET_AL_RE = re.compile(r"\bet\.?\s+al\.?\b", re.IGNORECASE)
_PARTICLE_WORDS = {"da", "de", "del", "der", "di", "du", "la", "le", "van", "von"}


@dataclass(frozen=True)
class ParsedName:
    original: str
    normalized: str
    surname: str
    given_tokens: tuple[str, ...]
    initials: tuple[str, ...]


def normalize_author_name(name: str) -> str:
    """Normalize an author name for fuzzy comparison."""
    name = name.strip().lower()
    name = _ET_AL_RE.sub("", name)
    name = _SUFFIX_RE.sub("", name)
    name = re.sub(r"[^\w\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _tokens(text: str) -> list[str]:
    normalized = normalize_author_name(text)
    return re.findall(r"[a-z0-9]+", normalized)


def _extract_surname(name: str) -> str:
    """Extract a surname from common academic citation name formats."""
    name = name.strip()
    if "," in name:
        return " ".join(_tokens(name.split(",", 1)[0]))

    parts = _tokens(name)
    if len(parts) >= 2 and parts[-2] in _PARTICLE_WORDS:
        return " ".join(parts[-2:])
    return parts[-1] if parts else normalize_author_name(name)


def _parse_author_name(name: str) -> ParsedName:
    normalized = normalize_author_name(name)
    if "," in name:
        surname_part, given_part = name.split(",", 1)
        surname_tokens = _tokens(surname_part)
        given_tokens = _tokens(given_part)
    else:
        parts = _tokens(name)
        if len(parts) >= 2 and parts[-2] in _PARTICLE_WORDS:
            surname_tokens = parts[-2:]
            given_tokens = parts[:-2]
        elif parts:
            surname_tokens = [parts[-1]]
            given_tokens = parts[:-1]
        else:
            surname_tokens = []
            given_tokens = []

    initials = tuple(token[0] for token in given_tokens if token)
    return ParsedName(
        original=name,
        normalized=normalized,
        surname=" ".join(surname_tokens),
        given_tokens=tuple(given_tokens),
        initials=initials,
    )


def _surname_similarity(a: ParsedName, b: ParsedName) -> float:
    if not a.surname or not b.surname:
        return 0.0
    return fuzz.ratio(a.surname, b.surname) / 100.0


def _initials_compatible(a: ParsedName, b: ParsedName) -> bool:
    if not a.given_tokens or not b.given_tokens:
        return True

    shortest = min(len(a.initials), len(b.initials))
    if shortest == 0:
        return True
    return a.initials[:shortest] == b.initials[:shortest]


def _given_names_compatible(a: ParsedName, b: ParsedName) -> bool:
    if not _initials_compatible(a, b):
        return False

    shared = min(len(a.given_tokens), len(b.given_tokens))
    for idx in range(shared):
        left = a.given_tokens[idx]
        right = b.given_tokens[idx]
        if left == right:
            continue
        if len(left) == 1 or len(right) == 1:
            continue
        if left[0] == right[0]:
            continue
        return False
    return True


def author_name_similarity(a: str, b: str) -> float:
    """Calculate author name similarity on a 0-1 scale."""
    na, nb = normalize_author_name(a), normalize_author_name(b)
    if not na or not nb:
        return 0.0

    parsed_a = _parse_author_name(a)
    parsed_b = _parse_author_name(b)
    full_score = fuzz.token_sort_ratio(na, nb) / 100.0
    surname_score = _surname_similarity(parsed_a, parsed_b)

    if surname_score > 0.9 and _given_names_compatible(parsed_a, parsed_b):
        if parsed_a.given_tokens and parsed_b.given_tokens:
            return max(full_score, 0.92)
        return max(full_score, 0.80)

    if surname_score > 0.9:
        return max(full_score, 0.75)

    return full_score


def _is_et_al(authors: list[str]) -> bool:
    """Return whether an author list is an abbreviated first-author form."""
    if not authors:
        return False
    joined = " ".join(authors).lower()
    return "et al" in joined or len(authors) == 1


def _first_author_matches(
    claimed: list[str],
    actual: list[str],
    threshold: float = 0.75,
) -> tuple[bool, str | None]:
    """Check whether the claimed first author appears in the actual authors."""
    if not claimed or not actual:
        return False, None
    for actual_name in actual:
        if author_name_similarity(claimed[0], actual_name) >= threshold:
            return True, actual_name
    return False, None


def author_list_similarity(
    claimed: list[str],
    actual: list[str],
    threshold: float = 0.80,
) -> tuple[float, list[tuple[str, str | None]]]:
    """Compare two author lists and return a score plus matched names."""
    if not claimed and not actual:
        return 1.0, []
    if not claimed or not actual:
        return 0.0, [(c, None) for c in claimed]

    if _is_et_al(claimed) and len(actual) > 1:
        matched, actual_name = _first_author_matches(claimed, actual)
        if matched:
            return 0.95, [(claimed[0], actual_name)]
        return 0.2, [(claimed[0], None)]

    matches: list[tuple[str, str | None]] = [(c, None) for c in claimed]
    remaining: list[tuple[int, str]] = list(enumerate(actual))

    for idx, claimed_name in enumerate(claimed):
        best_score = 0.0
        best_remaining_idx = -1
        for remaining_idx, (_, actual_name) in enumerate(remaining):
            score = author_name_similarity(claimed_name, actual_name)
            if score > best_score:
                best_score = score
                best_remaining_idx = remaining_idx

        if best_score >= threshold and best_remaining_idx >= 0:
            _, matched_actual = remaining.pop(best_remaining_idx)
            matches[idx] = (claimed_name, matched_actual)

    matched_count = sum(1 for _, matched in matches if matched is not None)
    total = max(len(claimed), len(actual))
    score = matched_count / total if total > 0 else 0.0

    return score, matches
