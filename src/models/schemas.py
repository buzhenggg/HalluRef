"""HalluRef 数据模型定义"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── 异常 ──────────────────────────────────────────────────

class InputTooLargeError(ValueError):
    """输入文本超出 max_input_chars 上限"""

    def __init__(self, length: int, limit: int):
        self.length = length
        self.limit = limit
        super().__init__(
            f"输入文本长度 {length} 字符，超出上限 {limit} 字符。请分段提交。"
        )


# ── 枚举类型 ──────────────────────────────────────────────

class HallucinationType(str, Enum):
    """幻觉引用分类"""
    FABRICATED = "FABRICATED"              # 完全捏造
    METADATA_ERROR = "METADATA_ERROR"      # 信息篡改 (title 或 authors 不匹配)
    VERIFIED = "VERIFIED"                  # 引用正确
    VERIFIED_MINOR = "VERIFIED_MINOR"      # 引用基本正确 (title 与 authors 匹配, 仅 year/venue 等小字段有误)
    UNVERIFIABLE = "UNVERIFIABLE"          # 无法验证 (LLM 看不出 / LLM 调用失败)


class MatchConfidence(str, Enum):
    """检索匹配置信度"""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NONE = "NONE"


class FieldMatchStatus(str, Enum):
    """字段匹配状态"""
    MATCH = "MATCH"
    MISMATCH = "MISMATCH"
    PARTIAL = "PARTIAL"
    UNKNOWN = "UNKNOWN"


# ── 引用提取 (Agent 1) ───────────────────────────────────

class ParsedCitation(BaseModel):
    """LLM文本中解析出的引用元数据"""
    authors: list[str] = Field(default_factory=list, description="作者列表")
    title: str = Field(default="", description="论文标题")
    year: Optional[int] = Field(default=None, description="发表年份")
    venue: Optional[str] = Field(default=None, description="发表venue/期刊")
    doi: Optional[str] = Field(default=None, description="DOI")


class Citation(BaseModel):
    """一条完整的引用记录"""
    citation_id: str = Field(description="引用ID, 如 ref_001")
    raw_text: str = Field(description="引用的原始文本")
    parsed: ParsedCitation = Field(description="解析后的结构化元数据")
    context: str = Field(default="", description="引用所在的上下文句子")


# ── 文献检索 (Agent 2) ───────────────────────────────────

class RetrievedPaper(BaseModel):
    """从学术数据库检索到的论文信息"""
    source: str = Field(description="数据源, 如 semantic_scholar / crossref")
    title: str = Field(default="")
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    abstract: Optional[str] = None
    url: Optional[str] = None


class RetrievalResult(BaseModel):
    """某条引用的检索结果"""
    citation_id: str
    found: bool = Field(description="是否找到匹配论文")
    confidence: MatchConfidence = MatchConfidence.NONE
    best_match: Optional[RetrievedPaper] = None
    all_candidates: list[RetrievedPaper] = Field(default_factory=list)
    debug_log: str = Field(default="", description="检索链路调试摘要")


# ── 元数据比对 (Agent 3) ──────────────────────────────────

class FieldComparison(BaseModel):
    """单个字段的比对结果"""
    field: str
    claimed: str = Field(description="LLM声称的值")
    actual: str = Field(description="真实值")
    status: FieldMatchStatus = FieldMatchStatus.UNKNOWN
    similarity: float = Field(default=0.0, description="相似度分数 0-1")


class MetadataComparisonResult(BaseModel):
    """元数据比对汇总"""
    citation_id: str
    fields: list[FieldComparison] = Field(default_factory=list)
    mismatch_count: int = 0
    has_major_mismatch: bool = False


# ── 综合报告 (Agent 4) ────────────────────────────────────

class CitationVerdict(BaseModel):
    """单条引用的最终判定"""
    citation_id: str
    raw_text: str = ""
    verdict: HallucinationType = HallucinationType.UNVERIFIABLE
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: str = Field(default="", description="判定依据")
    suggestion: str = Field(default="", description="修改建议")

    # 中间结果引用
    retrieval: Optional[RetrievalResult] = None
    metadata: Optional[MetadataComparisonResult] = None

class DetectionReport(BaseModel):
    """完整检测报告"""
    total_citations: int = 0
    verified: int = 0
    verified_minor: int = 0
    fabricated: int = 0
    metadata_error: int = 0
    unverifiable: int = 0
    details: list[CitationVerdict] = Field(default_factory=list)
