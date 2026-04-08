"""Agent 1: 引用提取智能体

从 LLM 生成的文本中提取所有引用, 输出结构化引用列表。
策略: 整篇文本一次性送入 LLM 进行结构化解析。
"""

from __future__ import annotations

import json

from loguru import logger

from src.models.schemas import Citation, ParsedCitation
from src.utils.llm_client import LLMClient

_SYSTEM_PROMPT = "你是一个学术引用解析专家。你的任务是从文本中提取所有学术引用并输出结构化数据。"

_EXTRACT_PROMPT = """\
请从以下学术文本中提取所有引用(包括行内引用和参考文献列表中的引用)。

对每条引用, 提取以下信息:
- authors: 作者列表 (尽可能完整)
- title: 论文标题 (必须来自文本中明确给出的内容; **严禁推断或编造**; 若文本中没有明确标题, 请直接省略该条引用, 不要输出)
- year: 发表年份
- venue: 发表期刊/会议 (如有)
- doi: DOI (如有)
- context: 引用出现的上下文句子
- claim: **可选字段**。仅当正文中明确出现"作者借该引用表达的观点 / 声称该论文做了什么"时才填写;
         若正文未对该引用作出任何具体声称 (例如这条引用只出现在文末参考文献列表, 或正文中只是
         单纯标注编号而没有展开论述), **必须留空字符串 ""**。
         **严禁**根据论文标题、摘要或常识推测 claim, 也**不要**把论文标题改写后填进 claim。
         claim 必须严格来自输入文本本身。

请以JSON数组格式返回, 每个元素格式如下:
```json
[
  {{
    "authors": ["Author1", "Author2"],
    "title": "Paper Title",
    "year": 2023,
    "venue": "NeurIPS",
    "doi": null,
    "context": "原文中引用出现的句子",
    "claim": ""
  }}
]
```

**编号引用合并规则**:
若文本采用编号引用 (如 `[1]`、`[12]` 等) 并在文末附参考文献列表, 请将**行内出现的编号**与**文末对应编号的条目**合并为同一条引用:
- `title / authors / year / venue / doi`: 取自文末参考文献条目
- `context`: 取该编号在正文中首次出现的整句; 若该编号在正文出现多次, 用 " | " 拼接多句
- `claim`: 描述正文中作者借该引用所表达的观点 / 声称
- 同一编号**只输出一条引用**, 不要重复

如果文本中没有任何引用, 返回空数组 `[]`。

--- 文本开始 ---
{text}
--- 文本结束 ---
"""


class CitationExtractor:
    """引用提取 Agent"""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def extract(self, text: str) -> list[Citation]:
        """从文本中提取所有引用

        Args:
            text: LLM 生成的包含引用的学术文本

        Returns:
            结构化引用列表，LLM 调用失败时返回空列表
        """
        prompt = _EXTRACT_PROMPT.format(text=text)
        try:
            parsed = await self.llm.chat_json(prompt, system=_SYSTEM_PROMPT)
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"[Agent1] LLM parse failed: {e}")
            return []

        if not isinstance(parsed, list):
            logger.warning("[Agent1] LLM returned non-list, wrapping")
            parsed = [parsed]

        citations: list[Citation] = []
        skipped = 0
        for item in parsed:
            title = (item.get("title") or "").strip()
            if not title:
                skipped += 1
                logger.warning(
                    f"[Agent1] skip citation without title: {item.get('context', '')[:80]}"
                )
                continue
            cid = f"ref_{len(citations) + 1:03d}"
            citations.append(Citation(
                citation_id=cid,
                raw_text=item.get("context", ""),
                parsed=ParsedCitation(
                    authors=item.get("authors", []),
                    title=title,
                    year=item.get("year"),
                    venue=item.get("venue"),
                    doi=item.get("doi"),
                ),
                context=item.get("context", ""),
                claim=item.get("claim", ""),
            ))

        logger.info(
            f"[Agent1] extracted {len(citations)} citations (skipped {skipped} without title)"
        )
        return citations
