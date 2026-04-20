"""Landing-page metadata extraction for search API and browser fallbacks."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from io import BytesIO
from urllib.parse import urlparse, urlunparse

import httpx
from bs4 import BeautifulSoup
from loguru import logger
from pypdf import PdfReader

from src.models.schemas import RetrievedPaper

_SITE_DEFAULT_VENUES: dict[str, str] = {
    "arxiv.org": "arXiv",
    "aclanthology.org": "ACL Anthology",
    "openreview.net": "OpenReview",
    "semanticscholar.org": "Semantic Scholar",
    "dblp.org": "DBLP",
    "pubmed.ncbi.nlm.nih.gov": "PubMed",
    "pmc.ncbi.nlm.nih.gov": "PubMed Central",
    "europepmc.org": "Europe PMC",
    "ieeexplore.ieee.org": "IEEE Xplore",
    "dl.acm.org": "ACM Digital Library",
    "link.springer.com": "Springer",
    "sciencedirect.com": "ScienceDirect",
    "nature.com": "Nature",
}


def _normalize_space(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _extract_year(text: str | None) -> int | None:
    if not text:
        return None
    match = re.search(r"(19|20)\d{2}", text)
    if not match:
        return None
    try:
        return int(match.group())
    except ValueError:
        return None


def _meta_values(soup: BeautifulSoup, *names: str) -> list[str]:
    wanted = {name.lower() for name in names}
    values: list[str] = []
    for tag in soup.find_all("meta"):
        key = (tag.get("name") or tag.get("property") or "").strip().lower()
        if key in wanted:
            value = _normalize_space(tag.get("content"))
            if value:
                values.append(value)
    return values


def _first_meta_value(soup: BeautifulSoup, *names: str) -> str | None:
    values = _meta_values(soup, *names)
    return values[0] if values else None


def _json_ld_candidates(soup: BeautifulSoup) -> list[dict]:
    candidates: list[dict] = []
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = _normalize_space(script.string or script.get_text())
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            candidates.extend(item for item in data if isinstance(item, dict))
        elif isinstance(data, dict):
            candidates.append(data)
    return candidates


def _json_candidates_by_type(soup: BeautifulSoup, *script_ids: str) -> list[dict]:
    candidates: list[dict] = []
    wanted = set(script_ids)
    for script in soup.find_all("script"):
        if script.get("id") not in wanted:
            continue
        raw = _normalize_space(script.string or script.get_text())
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            candidates.append(data)
    return candidates


def _authors_from_json_ld(candidate: dict) -> list[str]:
    author = candidate.get("author")
    if not author:
        return []
    if isinstance(author, list):
        names = []
        for item in author:
            if isinstance(item, dict):
                name = _normalize_space(item.get("name"))
            else:
                name = _normalize_space(str(item))
            if name:
                names.append(name)
        return names
    if isinstance(author, dict):
        name = _normalize_space(author.get("name"))
        return [name] if name else []
    name = _normalize_space(str(author))
    return [name] if name else []


def _default_venue_for_host(host: str) -> str | None:
    host = host.lower()
    for suffix, venue in _SITE_DEFAULT_VENUES.items():
        if host == suffix or host.endswith(f".{suffix}"):
            return venue
    return None


def _find_text(soup: BeautifulSoup, selectors: list[str]) -> str:
    for selector in selectors:
        node = soup.select_one(selector)
        if node:
            text = _normalize_space(node.get_text(" ", strip=True))
            if text:
                return text
    return ""


def _find_texts(soup: BeautifulSoup, selectors: list[str]) -> list[str]:
    values: list[str] = []
    for selector in selectors:
        for node in soup.select(selector):
            text = _normalize_space(node.get_text(" ", strip=True))
            if text and text not in values:
                values.append(text)
    return values


def _strip_prefix(text: str, prefixes: list[str]) -> str:
    lowered = text.lower()
    for prefix in prefixes:
        if lowered.startswith(prefix.lower()):
            return text[len(prefix):].strip(" :")
    return text


def _paper_from_values(
    url: str,
    title: str = "",
    authors: list[str] | None = None,
    year: int | None = None,
    venue: str | None = None,
    doi: str | None = None,
    abstract: str | None = None,
) -> RetrievedPaper | None:
    title = _normalize_space(title)
    authors = [a for a in (authors or []) if _normalize_space(a)]
    abstract = _normalize_space(abstract) or None
    if not title:
        return None
    if not any([authors, year, venue, doi, abstract]):
        return None
    return RetrievedPaper(
        source="page_parser",
        title=title,
        authors=authors,
        year=year,
        venue=venue,
        doi=doi,
        abstract=abstract,
        url=url,
    )


def normalize_landing_page_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path

    if "aclanthology.org" in host and path.endswith(".pdf"):
        new_path = path[:-4]
        if not new_path.endswith("/"):
            new_path += "/"
        return urlunparse(parsed._replace(path=new_path, query="", fragment=""))

    if "arxiv.org" in host and path.startswith("/pdf/") and path.endswith(".pdf"):
        arxiv_id = path[len("/pdf/"):-4]
        return urlunparse(parsed._replace(path=f"/abs/{arxiv_id}", query="", fragment=""))

    return url


def _parse_arxiv_page(url: str, soup: BeautifulSoup) -> RetrievedPaper | None:
    title = _strip_prefix(
        _find_text(soup, ["h1.title", "h1.title.mathjax"]),
        ["Title"],
    )
    authors = _find_texts(soup, [".authors a", ".authors .author"])
    abstract = _strip_prefix(
        _find_text(soup, ["blockquote.abstract", "blockquote.abstract.mathjax"]),
        ["Abstract"],
    )
    year = _extract_year(_find_text(soup, [".dateline", ".submission-history"]))
    return _paper_from_values(
        url=url,
        title=title,
        authors=authors,
        year=year,
        venue="arXiv",
        abstract=abstract,
    )


def _parse_acl_page(url: str, soup: BeautifulSoup) -> RetrievedPaper | None:
    title = _find_text(soup, ["h2#title", "h2.card-title", "h1"])
    authors = _find_texts(soup, [".acl-paper-authors a", ".lead a", "p.lead a"])
    abstract = _find_text(soup, ["div.acl-abstract", "#abstract", ".card-body .acl-abstract"])
    year = _extract_year(_find_text(soup, [".acl-paper-details", ".card-footer", "title"]))
    doi = _find_text(soup, ["a[href*='doi.org']", "span.doi"])
    if doi and "doi.org/" in doi:
        doi = doi.split("doi.org/")[-1]
    return _paper_from_values(
        url=url,
        title=title,
        authors=authors,
        year=year,
        venue="ACL Anthology",
        doi=doi or None,
        abstract=abstract,
    )


def _parse_openreview_page(url: str, soup: BeautifulSoup) -> RetrievedPaper | None:
    for candidate in _json_candidates_by_type(soup, "__NEXT_DATA__"):
        forum = (
            candidate.get("props", {})
            .get("pageProps", {})
            .get("forumNote", {})
        )
        content = forum.get("content", {})
        title = _normalize_space(content.get("title"))
        authors = [
            _normalize_space(a)
            for a in content.get("authors", [])
            if _normalize_space(a)
        ]
        abstract = _normalize_space(content.get("abstract"))
        timestamp = forum.get("cdate") or forum.get("tcdate")
        year = None
        if isinstance(timestamp, (int, float)):
            year = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).year
        paper = _paper_from_values(
            url=url,
            title=title,
            authors=authors,
            year=year,
            venue="OpenReview",
            abstract=abstract,
        )
        if paper is not None:
            return paper
    return _paper_from_values(
        url=url,
        title=_find_text(soup, ["h2.note_content_title", "h1"]),
        authors=_find_texts(soup, [".note_content_field .author", ".forum-authors span"]),
        year=_extract_year(_find_text(soup, [".note_content_value", "time"])),
        venue="OpenReview",
        abstract=_find_text(soup, [".note_content_value", ".forum-abstract"]),
    )


def _parse_ieee_page(url: str, soup: BeautifulSoup) -> RetrievedPaper | None:
    for candidate in _json_candidates_by_type(soup, "xplGlobalDocumentMetadata"):
        title = _normalize_space(candidate.get("title"))
        authors = []
        for author in candidate.get("authors", []):
            if isinstance(author, dict):
                name = _normalize_space(author.get("name"))
            else:
                name = _normalize_space(str(author))
            if name:
                authors.append(name)
        paper = _paper_from_values(
            url=url,
            title=title,
            authors=authors,
            year=_extract_year(candidate.get("publicationYear")),
            venue=_normalize_space(candidate.get("publicationTitle")) or "IEEE Xplore",
            doi=_normalize_space(candidate.get("doi")) or None,
            abstract=_normalize_space(candidate.get("abstract")),
        )
        if paper is not None:
            return paper
    return _paper_from_values(
        url=url,
        title=_find_text(soup, ["h1.document-title", "h1"]),
        authors=_find_texts(soup, [".authors-info-container a.author-name", ".authors-container a"]),
        year=_extract_year(_find_text(soup, [".u-pb-1.stats-document-abstract-publishedIn", ".doc-abstract-publishedIn"])),
        venue=_find_text(soup, [".stats-document-abstract-publishedIn a", ".doc-abstract-publishedIn a"]) or "IEEE Xplore",
        doi=_find_text(soup, ["a.stats-document-abstract-doi", ".doc-abstract-doi"]),
        abstract=_find_text(soup, [".abstract-text", ".u-mb-1 .abstract-text"]),
    )


def _parse_acm_page(url: str, soup: BeautifulSoup) -> RetrievedPaper | None:
    abstract = _find_text(
        soup,
        ["section#abstract div[role='paragraph']", "section.abstract div[role='paragraph']", "#abstract div"],
    )
    return _paper_from_values(
        url=url,
        title=_find_text(soup, ["h1[property='name']", "h1.citation__title", "h1"]),
        authors=_find_texts(soup, [".author-data [property='author']", ".loa__author-name", ".author-name"]),
        year=_extract_year(_find_text(soup, [".core-date-published", ".issue-item__detail", ".epub-section__date"])),
        venue=_find_text(soup, [".issue-item__detail a", ".epub-section__title", ".issue-heading"]) or "ACM Digital Library",
        doi=_find_text(soup, ["a[href*='doi.org']", ".doi a"]),
        abstract=abstract,
    )


def _clean_title(title: str, host: str) -> str:
    title = _normalize_space(title)
    venue = _default_venue_for_host(host)
    if venue and title.endswith(f" - {venue}"):
        return title[: -(len(venue) + 3)].strip()
    return title


def _extract_pdf_abstract(text: str) -> str | None:
    normalized = text.replace("\r", "\n")
    match = re.search(
        r"\bAbstract\b[:\s]*([\s\S]{40,2000}?)(?:\n\s*(?:1\s+Introduction|Introduction|Keywords|Index Terms)\b|$)",
        normalized,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return _normalize_space(match.group(1)) or None


def _extract_pdf_authors(lines: list[str]) -> list[str]:
    if len(lines) < 2:
        return []
    candidate = lines[1]
    if "@" in candidate or len(candidate) > 200:
        return []
    authors = []
    for part in re.split(r",| and ", candidate):
        name = _normalize_space(part)
        if not name:
            continue
        words = name.split()
        if 1 < len(words) <= 5 and all(any(ch.isalpha() for ch in word) for word in words):
            authors.append(name)
    return authors


def parse_pdf_metadata(url: str, pdf_bytes: bytes) -> RetrievedPaper | None:
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
    except Exception:
        return None

    text_parts: list[str] = []
    for page in reader.pages[:3]:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text:
            text_parts.append(text)
    full_text = "\n".join(text_parts)
    if not full_text.strip():
        return None

    lines = [_normalize_space(line) for line in full_text.splitlines() if _normalize_space(line)]
    if not lines:
        return None

    return RetrievedPaper(
        source="page_parser",
        title=lines[0],
        authors=_extract_pdf_authors(lines),
        year=_extract_year(url) or _extract_year(full_text),
        venue=_default_venue_for_host(urlparse(url).netloc.lower()),
        doi=None,
        abstract=_extract_pdf_abstract(full_text),
        url=url,
    )


def parse_page_metadata(url: str, html: str) -> RetrievedPaper | None:
    """Extract paper metadata from a landing page HTML document."""
    soup = BeautifulSoup(html, "html.parser")
    host = urlparse(url).netloc.lower()

    if "arxiv.org" in host:
        paper = _parse_arxiv_page(url, soup)
        if paper is not None:
            return paper
    elif "aclanthology.org" in host:
        paper = _parse_acl_page(url, soup)
        if paper is not None:
            return paper
    elif "openreview.net" in host:
        paper = _parse_openreview_page(url, soup)
        if paper is not None:
            return paper
    elif "ieeexplore.ieee.org" in host:
        paper = _parse_ieee_page(url, soup)
        if paper is not None:
            return paper
    elif "dl.acm.org" in host:
        paper = _parse_acm_page(url, soup)
        if paper is not None:
            return paper

    title = _first_meta_value(soup, "citation_title", "og:title", "twitter:title")
    authors = _meta_values(soup, "citation_author")
    venue = _first_meta_value(
        soup,
        "citation_journal_title",
        "citation_conference_title",
        "prism.publicationname",
        "og:site_name",
    )
    doi = _first_meta_value(soup, "citation_doi", "prism.doi")
    abstract = _first_meta_value(
        soup,
        "citation_abstract",
        "description",
        "og:description",
        "twitter:description",
    )
    date = _first_meta_value(
        soup,
        "citation_date",
        "citation_publication_date",
        "prism.publicationdate",
        "article:published_time",
    )
    year = _extract_year(date)

    if not title:
        title_tag = soup.find("title")
        title = _normalize_space(title_tag.get_text()) if title_tag else ""

    if not authors or not year or not abstract or not doi:
        for candidate in _json_ld_candidates(soup):
            title = title or _normalize_space(candidate.get("headline") or candidate.get("name"))
            if not authors:
                authors = _authors_from_json_ld(candidate)
            if not year:
                year = _extract_year(candidate.get("datePublished") or candidate.get("dateCreated"))
            if not abstract:
                abstract = _normalize_space(candidate.get("description") or candidate.get("abstract")) or abstract
            if not doi:
                identifier = candidate.get("identifier")
                if isinstance(identifier, dict):
                    doi = _normalize_space(identifier.get("value")) or doi
                elif isinstance(identifier, str) and "10." in identifier:
                    doi = _normalize_space(identifier)

    title = _clean_title(title, host)
    venue = venue or _default_venue_for_host(host)

    if not title:
        return None
    if not any([authors, year, venue, doi, abstract]):
        return None

    return RetrievedPaper(
        source="page_parser",
        title=title,
        authors=authors,
        year=year,
        venue=venue,
        doi=doi,
        abstract=abstract,
        url=url,
    )


async def fetch_page_metadata(
    url: str,
    timeout: int = 15,
    proxy: str | None = None,
) -> RetrievedPaper | None:
    """Fetch a landing page or PDF and extract paper metadata."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    normalized_url = normalize_landing_page_url(url)
    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers=headers,
            proxy=proxy,
        ) as client:
            resp = await client.get(normalized_url)
            resp.raise_for_status()
            final_url = normalize_landing_page_url(str(resp.url))
            content_type = resp.headers.get("content-type", "").lower()
            if "application/pdf" in content_type or final_url.lower().endswith(".pdf"):
                paper = parse_pdf_metadata(final_url, resp.content)
            else:
                paper = parse_page_metadata(final_url, resp.text)
            if paper is not None:
                paper.url = final_url
            return paper
    except Exception as exc:
        logger.debug(f"[page_parser] failed to parse {url}: {exc}")
        return None
