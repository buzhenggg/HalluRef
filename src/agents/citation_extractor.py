"""Agent 1: 引用提取模块

从 LLM 生成的文本中提取所有引用, 输出结构化引用列表。
策略: 长文本按固定窗口切块, 并发提交给 LLM 进行结构化解析后合并去重。
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

from loguru import logger

from src.models.schemas import Citation, ParsedCitation
from src.utils.llm_client import LLMClient

CHUNK_SIZE = 10000
CONTEXT_SIZE = 500
MAX_EXTRACT_RETRIES = 3

_SYSTEM_PROMPT = (
    "You are an expert academic citation parser. "
    "Your task is to extract academic citations from text and return structured data."
)

_EXTRACT_PROMPT = """\
Extract all academic citations from the following text segment, including inline citations and bibliography/reference-list entries.

The input is split into three parts:
1. Previous context: use this only to complete citation information truncated at the boundary of the current segment. Do not extract new citations from this part.
2. Current segment: this is the **only** range from which citations may be extracted.
3. Following context: use this only to complete citation information truncated at the boundary of the current segment. Do not extract new citations from this part.

Important rules:
- The current segment may contain no academic citations. In that case, return an empty array `[]`.
- Do not invent or fabricate citations just to produce a non-empty result.
- Output only citations that appear in the current segment.
- Previous/following context may only be used to complete title / authors / year / venue / doi for citations that appear in the current segment.
- If a citation appears only in the previous context or following context, do not output it.
- If a title in the current segment is truncated but can be completed from the previous/following context, output the completed full title.
- If the complete title still cannot be recovered, skip that citation. Do not output a truncated title.

For each citation, extract:
- authors: author list, as complete as possible
- title: paper title (must be explicitly present in the input text; **do not infer or fabricate**; if the text does not clearly provide a title, omit that citation)
- year: publication year
- venue: journal or conference, if available
- doi: DOI, if available
- context: the sentence or local context where the citation appears

Return a JSON array. Each element must have this shape:
```json
[
  {{
    "authors": ["Author1", "Author2"],
    "title": "Paper Title",
    "year": 2023,
    "venue": "NeurIPS",
    "doi": null,
    "context": "The sentence where the citation appears in the original text"
  }}
]
```

**Numbered citation merge rule**:
If the text uses numbered citations such as `[1]` or `[12]` and includes a reference list, merge the inline number with the corresponding reference-list entry into one citation:
- `title / authors / year / venue / doi`: take these from the reference-list entry
- `context`: use the full sentence where the number first appears in the main text; if the same number appears multiple times in the current segment, join those contexts with " | "
- Output only one citation per number. Do not duplicate it.

If there are no citations in the current segment, return an empty array `[]`.

--- Previous Context Start ---
{previous_context}
--- Previous Context End ---

--- Current Segment Start ---
{current_text}
--- Current Segment End ---

--- Following Context Start ---
{next_context}
--- Following Context End ---
"""

_RETRY_INSTRUCTION = """\

Your previous response could not be accepted for this reason:
{error}

Please try again. Return only a valid JSON array with the required schema and no extra prose.
"""


@dataclass(frozen=True)
class TextChunk:
    current: str
    previous_context: str
    next_context: str


class CitationExtractor:
    """引用提取 Agent"""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    @classmethod
    def _split_text(cls, text: str) -> list[TextChunk]:
        if len(text) <= CHUNK_SIZE:
            return [TextChunk(text, "", "")]

        chunks: list[TextChunk] = []
        for start in range(0, len(text), CHUNK_SIZE):
            current = text[start:start + CHUNK_SIZE]
            if not current:
                continue
            previous_start = max(0, start - CONTEXT_SIZE)
            next_end = min(len(text), start + CHUNK_SIZE + CONTEXT_SIZE)
            chunks.append(TextChunk(
                current=current,
                previous_context=text[previous_start:start],
                next_context=text[start + CHUNK_SIZE:next_end],
            ))
            if start + CHUNK_SIZE >= len(text):
                break
        return chunks

    @staticmethod
    def _title_key(title: str) -> str:
        return re.sub(r"\s+", " ", title.casefold()).strip()

    @staticmethod
    def _validate_items(parsed) -> tuple[list[dict], str | None]:
        if not isinstance(parsed, list):
            return [], "Top-level JSON value must be an array."

        valid_items: list[dict] = []
        for index, item in enumerate(parsed):
            if not isinstance(item, dict):
                return [], f"Item {index} must be an object."

            authors = item.get("authors", [])
            if not isinstance(authors, list) or not all(isinstance(a, str) for a in authors):
                return [], f"Item {index}.authors must be an array of strings."

            title = item.get("title", "")
            if title is not None and not isinstance(title, str):
                return [], f"Item {index}.title must be a string or null."

            year = item.get("year")
            if year is not None and not isinstance(year, int):
                return [], f"Item {index}.year must be an integer or null."

            for field in ("venue", "doi", "context"):
                value = item.get(field)
                if value is not None and not isinstance(value, str):
                    return [], f"Item {index}.{field} must be a string or null."

            valid_items.append(item)

        return valid_items, None

    async def _extract_chunk(self, chunk: TextChunk, chunk_index: int) -> list[dict]:
        base_prompt = _EXTRACT_PROMPT.format(
            previous_context=chunk.previous_context,
            current_text=chunk.current,
            next_context=chunk.next_context,
        )
        last_error = ""
        for attempt in range(1, MAX_EXTRACT_RETRIES + 1):
            prompt = base_prompt
            if last_error:
                prompt += _RETRY_INSTRUCTION.format(error=last_error)
            try:
                parsed = await self.llm.chat_json(prompt, system=_SYSTEM_PROMPT)
            except Exception as e:
                last_error = f"JSON parsing or LLM call failed: {e}"
                logger.warning(
                    f"[Agent1] chunk {chunk_index} attempt {attempt} failed: {e}"
                )
                continue

            items, error = self._validate_items(parsed)
            if error is None:
                return items

            last_error = error
            logger.warning(
                f"[Agent1] chunk {chunk_index} attempt {attempt} invalid payload: {error}"
            )

        logger.error(
            f"[Agent1] chunk {chunk_index} failed after {MAX_EXTRACT_RETRIES} attempts: {last_error}"
        )
        return []

    @staticmethod
    def _item_to_citation(item: dict, citation_id: str) -> Citation | None:
        title = (item.get("title") or "").strip()
        if not title:
            logger.warning(
                f"[Agent1] skip citation without title: {item.get('context', '')[:80]}"
            )
            return None
        context = item.get("context", "") or ""
        return Citation(
            citation_id=citation_id,
            raw_text=context,
            parsed=ParsedCitation(
                authors=item.get("authors", []),
                title=title,
                year=item.get("year"),
                venue=item.get("venue"),
                doi=item.get("doi"),
            ),
            context=context,
        )

    @staticmethod
    def _merge_duplicate(existing: Citation, duplicate: Citation) -> None:
        if not existing.parsed.authors and duplicate.parsed.authors:
            existing.parsed.authors = duplicate.parsed.authors
        if existing.parsed.year is None and duplicate.parsed.year is not None:
            existing.parsed.year = duplicate.parsed.year
        if not existing.parsed.venue and duplicate.parsed.venue:
            existing.parsed.venue = duplicate.parsed.venue
        if not existing.parsed.doi and duplicate.parsed.doi:
            existing.parsed.doi = duplicate.parsed.doi

        if duplicate.context:
            contexts = [part.strip() for part in existing.context.split(" | ") if part.strip()]
            if duplicate.context not in contexts:
                contexts.append(duplicate.context)
                merged = " | ".join(contexts)
                existing.context = merged
                existing.raw_text = merged

    def _build_citations(self, items: list[dict]) -> list[Citation]:
        citations: list[Citation] = []
        by_title: dict[str, Citation] = {}
        skipped = 0

        for item in items:
            citation = self._item_to_citation(item, f"ref_{len(citations) + 1:03d}")
            if citation is None:
                skipped += 1
                continue

            key = self._title_key(citation.parsed.title)
            if key in by_title:
                self._merge_duplicate(by_title[key], citation)
                continue

            by_title[key] = citation
            citations.append(citation)

        for idx, citation in enumerate(citations, start=1):
            citation.citation_id = f"ref_{idx:03d}"

        logger.info(
            f"[Agent1] extracted {len(citations)} citations (skipped {skipped} without title)"
        )
        return citations

    async def extract(self, text: str) -> list[Citation]:
        """从文本中提取所有引用

        Args:
            text: LLM 生成的包含引用的学术文本

        Returns:
            结构化引用列表，LLM 调用失败时返回空列表
        """
        chunks = self._split_text(text)
        if len(chunks) > 1:
            logger.info(
                f"[Agent1] split input into {len(chunks)} chunks "
                f"(chunk_size={CHUNK_SIZE}, context_size={CONTEXT_SIZE})"
            )

        chunk_results = await asyncio.gather(
            *[
                self._extract_chunk(chunk, index)
                for index, chunk in enumerate(chunks, start=1)
            ]
        )
        items = [item for chunk_items in chunk_results for item in chunk_items]
        return self._build_citations(items)

