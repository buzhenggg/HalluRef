# HalluRef API 文档

## Agent 2: ReferenceRetriever — 文献检索核查

### 概述

对 LLM 生成文本中提取的引用，按级联策略调用学术数据库 / Web 搜索 / 爬虫多源进行检索验证。

### 检索策略

- Tier 1 学术 API **并行**：OpenAlex + CrossRef 通过 `asyncio.gather` 同时调用
- Tier 2 Web 搜索 **二选一**：Serper 优先，未配置时回落 SerpAPI
- Tier 3 Google Scholar 直爬兜底
- 任一 tier 命中阈值即停，后续 tier 跳过
- 对**多条引用**的批量验证，使用 `asyncio.Semaphore` 控制并发数，避免触发 API 限流

### 类: `ReferenceRetriever`

#### `__init__(retrievers, title_exact_threshold, title_fuzzy_threshold, max_concurrent)`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `retrievers` | `list[BaseRetriever]` | 必填 | 检索器列表 |
| `title_exact_threshold` | `float` | 0.95 | 精确匹配阈值 |
| `title_fuzzy_threshold` | `float` | 0.85 | 模糊匹配阈值 |
| `max_concurrent` | `int` | 5 | 批量验证时的最大并发引用数 |

#### `verify(citation: Citation) -> RetrievalResult`

验证单条引用。并发调用所有检索器，合并结果后选出最佳匹配。

- **输入**: `Citation` 对象（含 title, authors, year）
- **输出**: `RetrievalResult`（含 found, confidence, best_match, all_candidates）
- **行为**: 所有检索器并发执行，单个失败不阻塞其他

#### `verify_batch(citations: list[Citation]) -> list[RetrievalResult]`

批量验证多条引用。使用信号量控制并发数。

- **输入**: `list[Citation]`
- **输出**: `list[RetrievalResult]`（顺序与输入一致）
- **并发控制**: 最多 `max_concurrent` 条引用同时验证

### Google Scholar 补查（Fallback）

当 OpenAlex + CrossRef 主力检索均未找到匹配论文时，自动调用 Google Scholar 进行补查。

- **实现**: `GoogleScholarRetriever`，基于 `httpx + BeautifulSoup` 直接爬取 `scholar.google.com`
- **调用时机**: 仅当主力检索 `found=False` 时触发，避免不必要的爬取
- **原生异步**: 复用项目已有的 `httpx.AsyncClient`，无需线程池包装
- **代理支持**: 支持 HTTP/SOCKS 代理，避免 Google 封禁 IP。在 `config.yaml` 中配置
- **中文友好**: 实测 Google Scholar 对中文文献覆盖良好（核心期刊命中率高）
- **解析字段**: 标题 / 作者 / 期刊 / 年份 / 摘要片段 / 引用数 / 链接

#### `GoogleScholarRetriever`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `proxy` | `str \| None` | `None` | HTTP/SOCKS5 代理地址，如 `http://127.0.0.1:7890` |
| `timeout` | `int` | 15 | 单次检索超时秒数 |
| `max_results` | `int` | 3 | 每次检索返回的最大结果数 |

#### ReferenceRetriever 新增参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `fallback_retriever` | `GoogleScholarRetriever \| None` | `None` | Google Scholar 补查检索器 |

#### 补查流程

```
verify(citation):
  1. 并发调用 OpenAlex + CrossRef
  2. 选出最佳匹配, 判定 found
  3. 若 found=False 且 fallback_retriever 存在:
     a. 调用 Google Scholar 检索
     b. 合并候选, 重新选出最佳匹配
  4. 返回 RetrievalResult
```

### 数据源

| 检索器 | 数据源 | 认证方式 | 限流 |
|--------|--------|----------|------|
| `OpenAlexRetriever` | OpenAlex | mailto (polite pool) | 有 mailto: 10 req/s |
| `CrossRefRetriever` | CrossRef | mailto (polite pool) | 有 mailto: 更宽松 |
| `SemanticScholarRetriever` | Semantic Scholar | x-api-key (可选) | 无 key 1 req/s, 有 key 10 req/s |
| `ArxivRetriever` | arXiv | 无 | 建议 3s 间隔 |
| `SerperRetriever` | Serper Google Scholar | X-API-KEY (必填) | 注册即送 2500 次免费查询 |
| `GoogleScholarRetriever` | Google Scholar (直爬) | 代理 (可选) | 需配代理避免封 IP |

### CascadeRetriever — 级联检索编排器

`src/retrievers/cascade.py` 提供 `CascadeRetriever`，按 tier 顺序级联调用各检索接口。

#### 调用顺序

```
Tier 1  OpenAlex + CrossRef       并行                    ─┐
                                                            ├─ 任一 tier 命中阈值 → 立即返回
Tier 2  Serper → SerpAPI          二选一 (Serper 优先)    │
                                                            │
Tier 3  Google Scholar 直爬                                 ─┘
```

#### 规则

- **早停**：任一 tier 的最佳匹配达到 `title_fuzzy_threshold` 即返回，后续 tier 不调用
- **跳过未配置**：每个检索器的 `is_configured()` 返回 False 时自动跳过该检索器；某 tier 全部检索器都不可用则整 tier 跳过
- **Tier 1 并行**：OpenAlex 和 CrossRef 通过 `asyncio.gather` 同时调用，候选合并后统一打分
- **Tier 2 二选一**：Serper 优先，未配置时回落 SerpAPI
- **失败鲁棒**：单检索器异常 / 超时被吞掉，不阻塞同 tier 其他检索器或后续 tier
- **候选累积**：未命中时所有 tier 的候选会累积保留，供下游元数据/内容比对
- **来源优先级 (tiebreak)**：分数相同时按 `openalex > crossref > serper > google_scholar` 选最佳

#### `is_configured()` 检查

| 检索器 | 配置完整条件 |
|--------|-------------|
| `OpenAlexRetriever` | 始终 True (mailto 可选) |
| `CrossRefRetriever` | 始终 True |
| `SerperRetriever` | `api_key` 非空 |
| `GoogleScholarRetriever` | 始终 True (proxy 可选) |
| `BaseRetriever` 默认 | True，子类可覆盖 |

#### `search(title, authors, year) -> CascadeSearchResult`

```python
@dataclass
class CascadeSearchResult:
    found: bool
    confidence: MatchConfidence
    best_match: RetrievedPaper | None
    score: float
    hit_tier: str | None      # 'academic' / 'serper' / 'scholar' / None
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
| `src/retrievers/serper.py` | `SerperRetriever` | `POST https://google.serper.dev/scholar` |
| `src/retrievers/google_scholar.py` | `GoogleScholarRetriever` | 直接爬取 `scholar.google.com/scholar` |

---

## Agent 4: ContentChecker — 内容一致性核查

### 跳过策略

当引用未包含对论文贡献的具体声明（`citation.claim` 为空）时，跳过 LLM 内容核查，直接返回 `content_check = None`。此时 Agent 5 的决策树将跳过内容检查环节，仅根据检索和元数据结果判定。

---

## 输入长度限制

为防止单次调用消耗过多 LLM token，所有入口（REST / SSE / CLI）均会在执行前检查输入文本长度：

- 配置项：`detection.max_input_chars`（默认 `20000`）
- 超限时抛出 `InputTooLargeError`，流水线立即终止，不执行任何 Agent
- REST `/api/detect` 返回 **HTTP 413 Payload Too Large**，body：
  ```json
  {"message": "输入文本长度 XXXXX 字符，超出上限 20000 字符。请分段提交。"}
  ```
- SSE `/api/detect/stream` 推送 `error` 事件后关闭连接
- CLI 打印错误并 `sys.exit(1)`

## Agent 1: CitationExtractor — 引用提取

仅使用 LLM 进行结构化解析（已移除正则预提取与降级逻辑）。整篇文本一次性送入 LLM，要求返回 JSON 数组，每条含 `authors / title / year / venue / doi / context / claim`。LLM 调用失败时返回空列表，不再使用正则兜底。

**Title 必填策略**：prompt 明确要求**不得推断**标题，文本未给出标题的引用直接由 LLM 省略；解析阶段还会再次过滤 `title` 为空 / 空白字符串的条目（记录 warning 日志），保证下游 Agent 2 拿到的每条 `Citation` 都有可检索的 title。

**编号引用合并策略**：针对 `[1]` / `[12]` 等编号式引用 + 文末参考文献列表的场景，prompt 要求 LLM 把行内编号与文末对应条目合并为同一条引用 —— 元数据 (`title/authors/year/venue/doi`) 取自文末，`context` 取该编号在正文中首次出现的整句（多次出现用 ` | ` 拼接），`claim` 取正文中作者借该引用表达的观点；同一编号只输出一条。

## Agent 5: ReportGenerator — 综合研判分类

`HallucinationType` 共 6 个标签：

| 标签 | 含义 | 触发条件 |
|------|------|----------|
| `FABRICATED` | 完全捏造 | 检索不到论文 |
| `METADATA_ERROR` | 信息篡改 | 论文存在，但 **title 或 authors** 字段 MISMATCH（关键字段错） |
| `MISREPRESENTED` | 观点歪曲 | 论文 + 关键元数据正确，但 LLM 内容核查判 INCONSISTENT / EXAGGERATED |
| `UNVERIFIABLE` | 无法验证 | **仅** orchestrator 层处理整条引用时抛异常的兜底（不再因 LLM 看不出 / 摘要缺失进入此档） |
| `VERIFIED_MINOR` | 引用基本正确 | 论文 + 标题 + 作者都对，但 year/venue 等次要字段 MISMATCH |
| `VERIFIED` | 引用正确 | 全部检查通过；或论文存在 + 关键字段正确且无内容核查可做（无 claim / 摘要缺失） |

决策树短路顺序：`FABRICATED → METADATA_ERROR → MISREPRESENTED → UNVERIFIABLE → VERIFIED_MINOR → VERIFIED`。

字段分组定义：
- **关键字段** (`_KEY_FIELDS`): `title`, `authors`
- **次要字段** (`_MINOR_FIELDS`): `year`, `venue`

**摘要缺失的处理**：Agent 4 在论文已找到但 `best_match.abstract` 为空时直接返回 `None`（不再返回 UNVERIFIABLE），由 Agent 5 走 VERIFIED / VERIFIED_MINOR 分支并在 evidence 中注明"摘要缺失，跳过内容核查"。

**LLM 内容存疑的处理**：当 Agent 4 的 LLM 返回 `consistency=UNVERIFIABLE`（仅凭摘要无法定论），Agent 5 **不会**因此把整条引用判为 UNVERIFIABLE 标签 —— 论文真实存在 + 关键元数据正确说明引用本身没出错，只是 LLM 拿摘要看不出 claim 真假。处理方式：仍按元数据情况落 VERIFIED / VERIFIED_MINOR，evidence 附注 "内容声明存疑 (LLM 仅凭摘要无法定论, 建议人工核实)"，并把完整 LLM reasoning 通过 `content_check` 字段返回前端供详情面板展示。

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
| `citation_verdict` | `CitationVerdict` (JSON) | 单条引用完成全部检测，含 verdict / evidence / retrieval / metadata / content_check |
| `report_complete` | `DetectionReport` (JSON) | 所有引用处理完毕，含汇总统计 |
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

### `GET /api/health`

健康检查。

- **Response**: `{ "status": "ok" }`

---

## Web 前端

单页应用，由 FastAPI 提供静态文件服务（`app/static/index.html`）。

- 调用 `POST /api/detect` 进行检测
- Tailwind CSS 构建，暗色模式支持
- 结果按判定类型颜色编码展示
