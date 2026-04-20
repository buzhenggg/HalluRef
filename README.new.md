# HalluRef

面向 LLM 生成学术文本的引用真实性检测系统。

HalluRef 聚焦两件事：

- 这条引用对应的论文是否真实存在
- 标题、作者、年份、venue 等元数据是否正确

当前版本已经移除了“内容一致性/观点是否忠于原文”的判断，系统边界更明确，重点放在“幻觉引用”和“元数据错误”上。

## 核心能力

- LLM 引用抽取：从整段文本中提取结构化引用单元
- 三级级联检索：Academic APIs -> Scholar Search API -> Google Scholar Direct
- Search API 优先，直接 Google Scholar 爬虫兜底
- 站点专属页面解析：补全普通搜索结果里的作者、年份、venue、DOI、摘要等字段
- PDF 归一化与兜底读取：优先从论文落地页提取，必要时读取 PDF
- 多界面支持：FastAPI、SSE 流式接口、Gradio、CLI

## 当前判定标签

| 标签 | 含义 |
|---|---|
| `VERIFIED` | 引用存在，关键字段和次要字段均基本正确 |
| `VERIFIED_MINOR` | 引用存在，标题/作者正确，但年份或 venue 等次要字段有误 |
| `METADATA_ERROR` | 引用存在，但标题或作者等关键字段不一致 |
| `FABRICATED` | 未检索到对应论文，疑似捏造引用 |
| `UNVERIFIABLE` | 流程异常导致无法完成验证 |

## 整体流程

```text
输入文本
  -> Agent 1: CitationExtractor
  -> Agent 2: ReferenceRetriever
       Tier 1  academic
         - OpenAlex
         - CrossRef
         - arXiv
       Tier 2  scholar_search
         - Serper / SerpAPI Scholar
       Tier 3  google_scholar_direct
         - GoogleScholarRetriever 直接爬取 Google Scholar
  -> Agent 3: MetadataComparator
  -> Agent 4: ReportGenerator
  -> DetectionReport
```

检索策略要点：

- Tier 1 并行调用 `OpenAlex + CrossRef + arXiv`
- `Scholar Search` API 优先于直接爬虫兜底
- 命中阈值后立即早停
- Search API 结果如果命中论文落地页，会继续做页面元数据增强

## 站点专属解析与 PDF 支持

项目目前支持“域名分发 + 通用兜底”的页面解析策略。

已经做了深度专属解析的站点：

- `arxiv.org`
- `aclanthology.org`
- `openreview.net`
- `ieeexplore.ieee.org`
- `dl.acm.org`

已经具备站点识别、默认 venue 补全或通用 meta/JSON-LD 解析能力的站点：

- `semanticscholar.org`
- `dblp.org`
- `pubmed.ncbi.nlm.nih.gov`
- `pmc.ncbi.nlm.nih.gov`
- `europepmc.org`
- `link.springer.com`
- `sciencedirect.com`
- `nature.com`
- `doi.org`

PDF 处理策略：

- `arxiv.org/pdf/*` 会自动归一化到 `arxiv.org/abs/*`
- `aclanthology.org/*.pdf` 会优先跳回论文落地页
- 若仍然只有 PDF 可用，会读取前几页文本做标题、摘要、年份、作者等兜底提取

## 快速开始

### 1. 创建环境

按项目约定使用 conda，环境名固定为 `halluref`：

```bash
conda create -n halluref python=3.11 -y
conda activate halluref
pip install -r requirements.txt
```

### 2. 配置

编辑 `config/config.yaml`：

```yaml
llm:
  base_url: "https://api.deepseek.com"
  api_key: "sk-xxx"
  model: "deepseek-chat"

retriever:
  proxy:
    server: "http://127.0.0.1:7890"

  openalex:
    mailto: "your@email.com"

  crossref:
    mailto: "your@email.com"

  serper:
    api_key: "your-serper-key"

  serpapi:
    api_key: "your-serpapi-key"

  google_scholar:
    timeout: 15
    max_results: 3

detection:
  max_input_chars: 100000
```

说明：

- `llm` 是必需项
- `OpenAlex` 和 `CrossRef` 默认即可使用，但建议配置 `mailto`
- `retriever.proxy.server` 会同时作用于：
  - Google Scholar 直接爬虫
  - 论文落地页 HTML / PDF 抓取

### 3. 运行

```bash
# FastAPI
uvicorn app.api:app --host 127.0.0.1 --port 8000

# Gradio
python app/ui.py

# CLI
python -m src.agents.orchestrator --input your_text.txt --output report.json
```

## 输出示例

```json
{
  "total_citations": 3,
  "verified": 1,
  "verified_minor": 1,
  "fabricated": 1,
  "metadata_error": 0,
  "unverifiable": 0,
  "details": [
    {
      "citation_id": "ref_001",
      "verdict": "FABRICATED",
      "confidence": 0.93,
      "evidence": "在所有数据源中均未检索到该论文",
      "suggestion": "建议核实引用来源或替换为真实文献"
    }
  ]
}
```

## API 与前端

详见 `docs/api.md`。

当前提供：

- `POST /api/detect`
- `POST /api/detect/stream`
- `GET /api/health`

其中 `/api/detect/stream` 会逐条推送 citation 级结果，适合前端边检索边展示。

## 项目结构

```text
src/
├─ agents/         # 引用抽取、检索验证、元数据比对、报告生成
├─ retrievers/     # 学术 API、搜索 API、Google Scholar 爬虫、页面解析、级联编排
├─ models/         # Pydantic 数据模型
└─ utils/          # LLM 客户端、相似度计算、姓名匹配等

app/
├─ api.py          # FastAPI
├─ ui.py           # Gradio
└─ static/         # Web 前端静态资源

config/
└─ config.yaml     # 主配置

tests/             # 测试
docs/
└─ api.md          # 接口与策略文档
```
