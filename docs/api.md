# HalluRef API 文档

## Agent 2: ReferenceRetriever — 文献检索核查

### 概述

对 LLM 生成文本中提取的引用，按级联策略调用学术数据库 / 检索 API / Google Scholar 直接爬虫进行检索验证。搜索 API 命中网页链接后，可进一步用站点专属解析器补全作者、年份、venue、DOI、摘要等结构化字段。

### 检索策略

- Tier 1 学术 API **伪并行**：先并行调用 OpenAlex + CrossRef；仅当这两个来源未达到命中阈值时，再并行调用 arXiv + Semantic Scholar。Semantic Scholar 必须配置 `retriever.semantic_scholar.api_key` 才会启用
- Tier 2 `scholar_search`：Search API 形式的 Google Scholar 搜索
  - 若已配置 Search API，则走 `Serper/SerpAPI Scholar`
  - 若未配置 Search API，则跳过该层
- Tier 3 `google_scholar_direct`：仅在前两层未命中时，用 `GoogleScholarRetriever` 直接 HTTP 爬取 Google Scholar
- 任一 tier 命中阈值即停，后续 tier 跳过
- 对**多条引用**的批量验证，使用 `asyncio.Semaphore` 控制并发数，避免触发 API 限流

#### 编排层调试开关

为方便本地调试，`src/agents/orchestrator.py` 内置了硬编码开关，不依赖 `config.yaml`：

- 学术 API 开关：控制 `OpenAlex / CrossRef / arXiv / Semantic Scholar`
- 搜索 API 开关：控制 `Serper / SerpAPI` 的 scholar 检索

关闭某一组后，对应检索器不会注入 `CascadeRetriever`，可用于快速隔离问题。

### 类: `ReferenceRetriever`

#### `__init__(retrievers, title_exact_threshold, title_fuzzy_threshold, max_concurrent)`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `retrievers` | `list[BaseRetriever]` | 必填 | 检索器列表 |
| `title_exact_threshold` | `float` | 0.95 | 精确匹配阈值 |
| `title_fuzzy_threshold` | `float` | 0.85 | 模糊匹配阈值 |
| `max_concurrent` | `int` | 5 | 批量验证时的最大并发引用数 |

#### 串行节流默认值

- `ReferenceRetriever` 默认在相邻两条引用之间随机等待 `1 ~ 3` 秒
- 该节流在 `verify()` 入口生效，用于降低对外部检索 API 的访问压力并避免固定节奏

#### `verify(citation: Citation) -> RetrievalResult`

验证单条引用。并发调用所有检索器，合并结果后选出最佳匹配。

- **输入**: `Citation` 对象（含 title, authors, year）
- **输出**: `RetrievalResult`（含 found, confidence, best_match, all_candidates, debug_log）
- **行为**: 所有检索器并发执行，单个失败不阻塞其他

#### `RetrievalResult.debug_log`

用于前端与用户调试检索链路的一段文本摘要，重点面向 `found=false` 的情况。

内容通常包括：

- 实际尝试过的检索层，例如 `academic / scholar_search / google_scholar_direct`
- 每层里调用了哪些检索器
- 是“未检索到候选 / 候选分数不够”还是“API/爬虫调用失败”
- 若最终未命中，会在 `CitationVerdict.evidence` 中附带这段摘要，便于用户排查

#### `verify_batch(citations: list[Citation]) -> list[RetrievalResult]`

批量验证多条引用。使用信号量控制并发数。

- **输入**: `list[Citation]`
- **输出**: `list[RetrievalResult]`（顺序与输入一致）
- **并发控制**: 最多 `max_concurrent` 条引用同时验证

### 检索 API 结果增强（站点专属解析）

Tier 2 / Tier 3 若只拿到普通搜索结果链接（如 Google 搜索结果），会尝试对目标页面做二次解析，以补齐作者、年份、venue、DOI 等结构化字段。

- **优先站点**：`arxiv.org`、`aclanthology.org`、`openreview.net`、`semanticscholar.org`、`dblp.org`、`pubmed.ncbi.nlm.nih.gov`、`pmc.ncbi.nlm.nih.gov`、`europepmc.org`、`ieeexplore.ieee.org`、`dl.acm.org`、`link.springer.com`、`sciencedirect.com`、`nature.com`、`doi.org`
- **提取顺序**：URL 归一化（优先把 PDF 跳回论文页）→ 按域名分发到站点专属 parser → `citation_*` / `prism.*` / `og:*` / `twitter:*` meta 标签 → JSON-LD → PDF 兜底读取 → 页面正文启发式解析
- **用途**：避免普通搜索结果只返回 `title/snippet/link`，导致作者为空从而误判 `METADATA_ERROR`

#### `enrich_result_link(url) -> RetrievedPaper | None`

- **输入**：检索结果 URL
- **输出**：补齐元数据的 `RetrievedPaper`；若无法解析则返回 `None`
- **调用位置**：`SerperRetriever` / `SerpApiRetriever`
- **代理支持**：可复用统一代理配置；若设置 `retriever.proxy.server`，则落地页抓取与 Google Scholar 直接爬虫均走同一代理

#### 已支持的站点专属解析与 PDF 归一化

当前 `src/retrievers/page_metadata.py` 采用“**域名分发 + 通用兜底**”模式：

- 先根据域名进入专属 parser，优先读取该站点常见的前端结构或页面内嵌数据
- 专属 parser 失败后，再回退到标准 meta / JSON-LD 提取
- 若最终资源仍是 PDF，则进入 PDF 文本兜底读取

已实现的深度站点专属 parser：

- `arxiv.org`：读取摘要页标题块、作者块、摘要块
- `aclanthology.org`：读取 ACL 论文页标题/作者/摘要区域，并支持 PDF URL 回跳
- `openreview.net`：读取页面内嵌脚本数据与论文信息块
- `ieeexplore.ieee.org`：读取 IEEE 页面内嵌 JSON 或文献信息块
- `dl.acm.org`：读取 ACM Digital Library 标题/作者/摘要块

其余站点当前仍主要依赖标准 meta / JSON-LD，但已具备站点识别、默认 venue 与 PDF 兜底能力。

| 站点 | 当前支持 | 说明 |
|------|----------|------|
| `arxiv.org` | URL 归一化 + meta / JSON-LD | `pdf/*` 自动归一化到 `abs/*`，默认 venue=`arXiv` |
| `aclanthology.org` | URL 归一化 + meta / JSON-LD | `*.pdf` 自动跳回论文页，默认 venue=`ACL Anthology` |
| `openreview.net` | meta / JSON-LD | 默认 venue=`OpenReview` |
| `semanticscholar.org` | meta / JSON-LD | 默认 venue=`Semantic Scholar` |
| `dblp.org` | meta / JSON-LD | 默认 venue=`DBLP` |
| `pubmed.ncbi.nlm.nih.gov` | meta / JSON-LD | 默认 venue=`PubMed` |
| `pmc.ncbi.nlm.nih.gov` | meta / JSON-LD | 默认 venue=`PubMed Central` |
| `europepmc.org` | meta / JSON-LD | 默认 venue=`Europe PMC` |
| `ieeexplore.ieee.org` | meta / JSON-LD | 默认 venue=`IEEE Xplore` |
| `dl.acm.org` | meta / JSON-LD | 默认 venue=`ACM Digital Library` |
| `link.springer.com` | meta / JSON-LD | 默认 venue=`Springer` |
| `sciencedirect.com` | meta / JSON-LD | 默认 venue=`ScienceDirect` |
| `nature.com` | meta / JSON-LD | 默认 venue=`Nature` |
| `doi.org` | 重定向复用 | 先跳转落地页，再复用对应站点解析器 |
| `*.pdf` 其他站点 | PDF 兜底读取 | 读取 PDF 前几页文本，抽取标题 / 摘要 / 年份 / 作者（若可识别） |

### 数据源

| 检索器 | 数据源 | 认证方式 | 限流 |
|--------|--------|----------|------|
| `OpenAlexRetriever` | OpenAlex | mailto (polite pool) | 有 mailto: 10 req/s |
| `CrossRefRetriever` | CrossRef | mailto (polite pool) | 有 mailto: 更宽松 |
| `ArxivRetriever` | arXiv | 无 | 内置进程级 3.5s 冷却，任意两次 arXiv HTTP 请求至少间隔 3.5 秒 |
| `SemanticScholarRetriever` | Semantic Scholar | x-api-key (必填) | 有 key 10 req/s |
| `SerperRetriever` | Serper Google Scholar | X-API-KEY (必填) | 注册即送 2500 次免费查询 |
| `SerpApiRetriever` | SerpAPI Google Scholar | API key (必填) | 100 次/月 |
| `GoogleScholarRetriever` | Google Scholar 直接爬虫 | 无 | 建议配置代理并串行节流 |

#### Google Scholar 直接爬虫

`GoogleScholarRetriever` 使用 `httpx + BeautifulSoup` 直接请求 `https://scholar.google.com/scholar`，解析结果页中的标题、作者、年份、venue、摘要片段和链接。它作为最终兜底层运行，仅在 `academic` 和 `scholar_search` 都未达到命中阈值时调用。

爬虫会检测 CAPTCHA、unusual traffic、sorry page、异常短响应等拦截信号；遇到拦截或解析失败时返回空候选，不中断整条检测流水线。

#### 统一代理配置

面向中国大陆等网络受限环境，检索链路支持统一代理配置：

```yaml
retriever:
  proxy:
    server: "http://127.0.0.1:7890"
```

行为：

- `src/retrievers/page_metadata.py` 在抓取落地页 HTML / PDF 时使用同一代理
- `GoogleScholarRetriever` 直接爬取 Google Scholar 时使用同一代理
- 若未配置 `retriever.proxy.server`，则保持直连

### CascadeRetriever — 级联检索编排器

`src/retrievers/cascade.py` 提供 `CascadeRetriever`，按 tier 顺序级联调用各检索接口。

#### 调用顺序

```
Tier 1a OpenAlex + CrossRef                               并行                     ─┐
                                                                                  │ 未命中
Tier 1b arXiv + Semantic Scholar                         并行                     │
                                                                                  ├─ 任一 tier 命中阈值 → 立即返回
Tier 2  Scholar Search  (Serper/SerpAPI Scholar)                                  │
                                                                                  │
Tier 3  Google Scholar Direct (GoogleScholarRetriever)                            ─┘
```

#### 规则

- **早停**：任一 tier 的最佳匹配达到 `title_fuzzy_threshold` 即返回，后续 tier 不调用
- **跳过未配置**：每个检索器的 `is_configured()` 返回 False 时自动跳过该检索器；某 tier 全部检索器都不可用则整 tier 跳过
- **Tier 1 伪并行**：OpenAlex / CrossRef 先并行；未命中阈值时再并行 arXiv / Semantic Scholar。这样优先使用较稳定的结构化来源，并减少 arXiv 429 限流风险
- **Scholar API 优先**：先跑 `scholar_search`；只要 Scholar API 已配置，`google_scholar_direct` 不启用
- **同层二选一**：同一搜索意图下 Serper 优先，未配置时回落 SerpAPI
- **站点增强**：搜索结果若包含论文落地页，优先走站点专属解析器补全元数据
- **失败鲁棒**：单检索器异常 / 超时被吞掉，不阻塞同 tier 其他检索器或后续 tier
- **候选累积**：未命中时所有 tier 的候选会累积保留，供下游元数据/内容比对
- **来源优先级 (tiebreak)**：分数相同时按 `openalex > crossref > arxiv > semantic_scholar > serper_scholar > serpapi_scholar > google_scholar` 选最佳

#### `is_configured()` 检查

| 检索器 | 配置完整条件 |
|--------|-------------|
| `OpenAlexRetriever` | 始终 True (mailto 可选) |
| `CrossRefRetriever` | 始终 True |
| `ArxivRetriever` | 始终 True |
| `SemanticScholarRetriever` | `api_key` 非空 |
| `SerperRetriever` | `api_key` 非空 |
| `SerpApiRetriever` | `api_key` 非空 |
| `GoogleScholarRetriever` | 始终 True |
| `BaseRetriever` 默认 | True，子类可覆盖 |

#### `search(title, authors, year) -> CascadeSearchResult`

```python
@dataclass
class CascadeSearchResult:
    found: bool
    confidence: MatchConfidence
    best_match: RetrievedPaper | None
    score: float
    hit_tier: str | None      # 'academic_primary' / 'academic_secondary' / 'scholar_search' / 'google_scholar_direct' / None
    candidates: list[RetrievedPaper]
    tiers_run: list[str]      # 实际执行过的 tier 列表
```

### 检索器统一接口

所有检索器（除 `GoogleScholarRetriever` 直爬实现外）继承 `BaseRetriever`，
统一暴露：

- `search_by_title(title: str) -> list[RetrievedPaper]`
- `search_by_author_year(authors: list[str], year: int | None) -> list[RetrievedPaper]`
- `search(title=..., authors=..., year=...)` — 综合两阶段检索（标题优先）
- `close()` — 释放 httpx client

每个接口独立成文件，便于单独测试与替换：

| 文件 | 类 | 端点 |
|------|----|------|
| `src/retrievers/openalex.py` | `OpenAlexRetriever` | `GET https://api.openalex.org/works` |
| `src/retrievers/crossref.py` | `CrossRefRetriever` | `GET https://api.crossref.org/works` |
| `src/retrievers/semantic_scholar.py` | `SemanticScholarRetriever` | `GET https://api.semanticscholar.org/graph/v1/paper/search` |
| `src/retrievers/arxiv.py` | `ArxivRetriever` | `GET http://export.arxiv.org/api/query` (Atom XML) |
| `src/retrievers/serper.py` | `SerperRetriever` | `POST https://google.serper.dev/scholar` 或 `/search` |
| `src/retrievers/serpapi.py` | `SerpApiRetriever` | `GET https://serpapi.com/search.json` (`engine=google_scholar` 或 `google`) |
| `src/retrievers/google_scholar.py` | `GoogleScholarRetriever` | `GET https://scholar.google.com/scholar` |
| `src/retrievers/page_metadata.py` | 站点专属解析器 | 解析论文落地页 HTML / meta / JSON-LD |

---

## 输入长度限制

为防止单次调用消耗过多 LLM token，所有入口（REST / SSE / CLI）均会在执行前检查输入文本长度：

- 配置项：`detection.max_input_chars`（默认 `100000`）
- 超限时抛出 `InputTooLargeError`，流水线立即终止，不执行任何 Agent
- REST `/api/detect` 返回 **HTTP 413 Payload Too Large**，body：
  ```json
  {"message": "输入文本长度 XXXXX 字符，超出上限 100000 字符。请分段提交。"}
  ```
- SSE `/api/detect/stream` 推送 `error` 事件后关闭连接
- CLI 打印错误并 `sys.exit(1)`

## Agent 1: CitationExtractor — 引用提取

仅使用 LLM 进行结构化解析（已移除正则预提取与降级逻辑）。输入文本仍先受 `detection.max_input_chars` 整体上限约束；未超限的文本进入 Agent 1 后，会按固定字符窗口处理：短文本直接单次提交，超过 10000 字符的长文本按 `chunk_size=10000` 拆分为多个**当前片段**。每个当前片段会额外携带前序/后续上下文用于补全边界截断信息，但 prompt 明确要求只从当前片段中提取引用。

当前默认面向英文文档，上下文长度固定为前后各 500 字符。多个 chunk 会并发请求 LLM，实际并发由现有 `LLMClient` semaphore 控制。

每个 chunk 要求返回 JSON 数组，每条仅含 `authors / title / year / venue / doi / context` 等检测所需字段。Agent 1 会检查返回值必须是 JSON 数组，且每个条目的字段类型必须符合要求：`authors` 为字符串数组，`title/context/venue/doi` 为字符串或 `null`，`year` 为整数或 `null`。若 JSON 解析、整体结构或字段类型不合法，会重新调用模型，单个 chunk 最多尝试 3 次。3 次仍失败时，该 chunk 返回空列表，不影响其他 chunk；所有 chunk 结果合并后，仅按标题去重并重新生成连续的 `citation_id`。

**Title 必填策略**：prompt 明确要求**不得推断**标题，文本未给出标题的引用直接由 LLM 省略；解析阶段还会再次过滤 `title` 为空 / 空白字符串的条目（记录 warning 日志），保证下游 Agent 2 拿到的每条 `Citation` 都有可检索的 title。

**编号引用合并策略**：针对 `[1]` / `[12]` 等编号式引用 + 文末参考文献列表的场景，prompt 要求 LLM 把行内编号与文末对应条目合并为同一条引用 —— 元数据 (`title/authors/year/venue/doi`) 取自文末，`context` 取该编号在正文中首次出现的整句（多次出现用 ` | ` 拼接）；同一编号只输出一条。

## Agent 4: ReportGenerator — 综合研判分类

当前系统聚焦**引用真实性与元数据正确性**，不再执行内容一致性核查。`HallucinationType` 共 5 个标签：

| 标签 | 含义 | 触发条件 |
|------|------|----------|
| `FABRICATED` | 完全捏造 | 检索不到论文 |
| `METADATA_ERROR` | 信息篡改 | 论文存在，但 **title 或 authors** 字段 MISMATCH（关键字段错） |
| `UNVERIFIABLE` | 无法验证 | orchestrator 处理异常，或检索命中但缺失 `authors` 等关键字段，导致无法完成关键元数据核验 |
| `VERIFIED_MINOR` | 引用基本正确 | 论文 + 标题 + 作者都对，但 year/venue 等次要字段 MISMATCH |
| `VERIFIED` | 引用正确 | 论文存在，且关键字段与次要字段检查通过 |

决策树短路顺序：`FABRICATED → METADATA_ERROR → UNVERIFIABLE → VERIFIED_MINOR → VERIFIED`。

字段分组定义：
- **关键字段** (`_KEY_FIELDS`): `title`, `authors`
- **次要字段** (`_MINOR_FIELDS`): `year`, `venue`

作者姓名匹配采用平衡召回策略：支持全名、首字母缩写、倒置名、`et al.` 和常见后缀差异，例如 `Z. Guo` / `Guo, Z.` 可匹配 `Zhijiang Guo`，但不同姓氏或同姓不同首字母仍会保守拒绝。

`DetectionReport` 在原有 5 个计数字段基础上新增 `verified_minor: int`。

## REST API (FastAPI)

### `POST /api/detect`

检测文本中的幻觉引用。

- **Request Body**: `{ "text": "包含引用的学术文本" }`
- **Response**: `{ "report": DetectionReport }`

### `POST /api/detect/stream`

以 SSE（Server-Sent Events）流式返回检测结果，每完成一条引用立即推送。

- **Request Body**: `{ "text": "包含引用的学术文本" }`
- **Response**: `text/event-stream`，逐行推送以下事件：

| 事件类型 | data 结构 | 说明 |
|----------|-----------|------|
| `extraction_done` | `{ "total": int, "citations": [{ citation_id, raw_text }] }` | 引用提取完成，告知前端总数 |
| `citation_verdict` | `CitationVerdict` (JSON) | 单条引用完成全部检测，含 verdict / evidence / retrieval / metadata |
| `report_complete` | `DetectionReport` (JSON) | 所有引用处理完毕，含汇总统计；`details` 按初始 `citation_id` 顺序排列，前端据此重排最终列表 |
| `error` | `{ "message": string }` | 流水线异常 |

示例事件流：
```
event: extraction_done
data: {"total": 3, "citations": [{"citation_id": "ref_001", "raw_text": "..."}]}

event: citation_verdict
data: {"citation_id": "ref_001", "verdict": "VERIFIED", ...}

event: citation_verdict
data: {"citation_id": "ref_002", "verdict": "FABRICATED", ...}

event: citation_verdict
data: {"citation_id": "ref_003", "verdict": "METADATA_ERROR", ...}

event: report_complete
data: {"total_citations": 3, "verified": 1, "fabricated": 1, ...}
```

### `POST /api/extract-file`

上传 PDF 或现代 Word `.docx` 文件，后端只提取纯文本并返回给前端；不会直接启动检测流程，也不会持久化保存上传文件。

- **Request Body**: `multipart/form-data`
- **字段**: `file`
- **支持格式**: `.pdf`, `.docx`
- **Response**:
  ```json
  {
    "filename": "paper.pdf",
    "content_type": "application/pdf",
    "text": "...",
    "char_count": 12345,
    "max_input_chars": 100000,
    "over_limit": false
  }
  ```

错误响应：

| 状态码 | 场景 |
|------|------|
| `400` | 未上传文件或文件为空 |
| `413` | 上传文件字节数超过 `detection.max_upload_bytes` |
| `415` | 不支持的文件格式，例如旧版 `.doc` |
| `422` | 文件解析失败，或未提取到可用文本 |

说明：

- 上传大小上限由 `detection.max_upload_bytes` 控制，默认 `10485760` 字节。
- 提取文本长度仍受 `detection.max_input_chars` 控制；接口不会截断文本，而是通过 `over_limit` 告知前端是否超过检测上限。

### `GET /api/health`

健康检查。

- **Response**: `{ "status": "ok" }`

### `GET /api/retrieval/config`

返回后端当前实际生效的检索配置状态和检测输入限制。前端启动时会读取其中的 `detection.max_input_chars`，用于保持字符计数器和检测按钮状态与后端配置一致；接口只检查配置与编排策略，不发起真实外部网络探测。

- **Method**: `GET`
- **Response**:
  ```json
  {
    "policy": "academic -> scholar_search -> google_scholar_direct; direct crawler is enabled only when no Scholar API is configured.",
    "detection": {
      "max_input_chars": 100000,
      "max_upload_bytes": 10485760
    },
    "tiers": [
      {
        "tier": "academic",
        "label": "Tier 1 Academic APIs",
        "active": true,
        "providers": [
          {"name": "OpenAlex", "source": "openalex", "configured": true, "active": true, "status": "active", "message": "已启用"},
          {"name": "CrossRef", "source": "crossref", "configured": true, "active": true, "status": "active", "message": "已启用"},
          {"name": "arXiv", "source": "arxiv", "configured": true, "active": true, "status": "active", "message": "已启用"},
          {"name": "Semantic Scholar", "source": "semantic_scholar", "configured": false, "active": false, "status": "error", "message": "Missing API key"}
        ]
      },
      {
        "tier": "scholar_search",
        "label": "Tier 2 Google Scholar Search APIs",
        "active": true,
        "providers": [
          {"name": "Serper Scholar", "source": "serper_scholar", "configured": true, "active": true, "status": "active", "message": "API key configured"},
          {"name": "SerpAPI Scholar", "source": "serpapi_scholar", "configured": false, "active": false, "status": "error", "message": "Missing API key"}
        ]
      },
      {
        "tier": "google_scholar_direct",
        "label": "Tier 3 Direct Google Scholar Crawler",
        "active": false,
        "providers": [
          {"name": "Google Scholar Direct", "source": "google_scholar", "configured": true, "active": false, "status": "standby", "message": "Disabled because a Scholar API is configured"}
        ]
      }
    ]
  }
  ```
- **状态语义**：
  - `active`：配置完整且会被当前检索链路使用。
  - `standby`：配置完整，但因优先级策略暂不启用，例如二层 Scholar API 已配置时跳过三层直爬。
  - `error`：必要配置缺失或无效，例如 API key 为空。
  - `disabled`：被代码级检索开关禁用。

当前检索顺序：

1. Tier 1a `academic_primary`：OpenAlex + CrossRef 并行。
2. Tier 1b `academic_secondary`：仅在 1a 未命中时调用 arXiv + Semantic Scholar，其中 Semantic Scholar 需要 API key。
3. Tier 2 `scholar_search`：Serper/SerpAPI Google Scholar API。只要任一 Scholar API 配置完整，Tier 3 不会启用。
4. Tier 3 `google_scholar_direct`：直接 `GoogleScholarRetriever` 爬虫，仅在没有可用 Scholar API key 时启用。

---

## Web 前端

单页应用，由 FastAPI 提供静态文件服务（`app/static/index.html`）。

- 调用 `POST /api/detect/stream` 进行流式检测
- Tailwind CSS 构建，暗色模式支持
- 结果按判定类型颜色编码展示
- 前端展示检索结果来源 `source`
- 前端不展示 `confidence`
- 前端判定类型与后端 `HallucinationType` 保持一致，仅展示：
  `VERIFIED / VERIFIED_MINOR / FABRICATED / METADATA_ERROR / UNVERIFIABLE`
- 前端不再展示“观点歪曲 / 内容一致性核查”相关文案或字段
- 输入框附近提示用户优先提交参考文献 / 引用部分，而不是整篇全文，以减少模型资源消耗并提升引用识别精度
- 流水线进度展示与当前后端阶段一致：
  `引用提取 → 文献检索 → 元数据比对 → 综合研判`

## Author Matching Policy Update

Author matching is intentionally recall-oriented for common citation formats. Single-first-author forms such as `Vaswani et al.` are treated as a high confidence author match when the first author can be found in the retrieved author list. Multi-author lists are matched as unordered sets by best available name pair, so retrieved author order differences do not create an author mismatch as long as corresponding authors are present. Empty author lists still do not match non-empty author lists.

## LLM Author Consistency Fallback

Author comparison first uses the local rule-based matcher. Clear rule matches and clear rule mismatches are resolved locally. Ambiguous author scores may call the configured LLM as a fallback judge using a strict JSON prompt that compares only List A claimed authors and List B retrieved authors. A successful LLM response maps `is_consistent=true` to authors `MATCH` with similarity `0.95`, and `is_consistent=false` to authors `MISMATCH` with similarity `0.0`. LLM call failures, invalid JSON, or missing fields fall back to the original rule score so detection continues.
