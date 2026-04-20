<p align="center">
  <h1 align="center">🔍 HalluRef</h1>
  <p align="center">
    <strong>面向 LLM 生成学术文本的幻觉引用检测框架</strong>
  </p>
</p>

![image-20260420183714776](/img/img1.png)


---

大语言模型在生成文献综述、研究现状等学术文本时，经常会产生**幻觉引用**。HalluRef 聚焦**引用真实性与元数据正确性**，用于自动检测引用是否真实存在，以及标题、作者、年份、venue 等元数据是否可靠。

当前版本已移除“观点歪曲 / 内容一致性核查”相关判断，系统边界更明确：只判断**论文是否存在**与**引用元数据是否正确**。

| 类型 | 说明 |
|:--|:--|
| **完全捏造** | 未检索到对应论文，疑似虚构作者、标题或出版信息 |
| **信息篡改** | 论文存在，但标题、作者、年份或 venue 等字段与真实记录不一致 |

HalluRef 通过 4 阶段流水线，为每条引用输出结构化判定结果。

## ✨ 核心功能

- **引用提取**：LLM 结构化解析文本，支持 `Author (year)`、`(Author, year)`、`[N]` 等格式；长文本会按 10000 字符当前片段切块，并携带前后各 500 字符上下文补全截断引用，最后按标题去重合并；输入长度受 `max_input_chars` 限制（默认 100000，超限直接拒绝）
- **两段级联检索**：Academic APIs -> Google Search；其中 Google Search 优先走 Search API，未配置 API key 时才启用 Google Scholar Direct 爬虫兜底
- **站点元数据增强**：对 arXiv、ACL Anthology、OpenReview、IEEE、ACM 等论文落地页做专属解析，补全作者、年份、venue、DOI、摘要等字段
- **PDF 归一化与兜底读取**：优先跳回论文落地页；必要时读取 PDF 前几页文本做元数据提取
- **元数据比对**：基于 `rapidfuzz` 相似度算法校验标题、作者、年份、venue
- **五类判定标签**：`VERIFIED` · `VERIFIED_MINOR` · `FABRICATED` · `METADATA_ERROR` · `UNVERIFIABLE`
- **多种调用方式**：FastAPI REST、SSE 流式接口、Web 单页前端、命令行

## 快速开始

### 1. 安装

按项目约定使用 conda，环境名固定为 `halluref`：

```bash
git clone <repo_url>
cd Hallucinated_citations_detection

conda create -n halluref python=3.11 -y
conda activate halluref

pip install -r requirements.txt
```

### 2. 配置

复制或编辑 `config/config.yaml`：

```yaml
llm:
  base_url: "https://api.deepseek.com"   # 任意 OpenAI 兼容端点
  api_key: "sk-xxx"
  model: "deepseek-chat"
  temperature: 0.1
  max_tokens: 8192

retriever:
  proxy:
    server: "http://127.0.0.1:7890"      # 可选，供谷歌学术直爬使用

  openalex:
    mailto: "your@email.com"             # 可选，进入 polite pool

  crossref:
    mailto: "your@email.com"             # 可选

  semantic_scholar:
    api_key: ""                          # 可选；配置后启用

  serper:
    api_key: "your-serper-key"           # 可选；Serper Scholar 优先

  serpapi:
    api_key: ""                          # 可选；Serper 未配置时作为备选

  google_scholar:
    timeout: 15
    max_results: 3

detection:
  max_input_chars: 100000                # 单次检测最大字符数，超限拒绝
```

> **最低要求：** 需配置 `llm` 接口。OpenAlex、CrossRef、arXiv 默认即可使用；强烈建议为 OpenAlex / CrossRef 配置 `mailto`，只用输入自己的邮箱即可，以获得更稳定的访问体验。

> **代理说明：** `retriever.proxy.server` 会同时作用于 Google Scholar 直接爬虫与论文落地页 HTML / PDF 抓取。

### 3. 运行

```bash
# Web 界面（推荐）
uvicorn app.api:app --host 127.0.0.1 --port 8000

# 命令行
python -m src.agents.orchestrator --input your_text.txt --output report.json
```

启动后浏览器打开：

```text
http://127.0.0.1:8000
```

## 检索流水线

HalluRef 采用 **两段级联检索策略**，未配置的源会自动跳过；任一段命中 `MEDIUM` 或 `HIGH` 置信度后立即早停。第二段中，Google Scholar Direct 只是兜底路径：只要已配置 Serper 或 SerpAPI，直接爬虫就不会接入本次检索。

```text
第 1 层 Academic APIs（并行）       OpenAlex + CrossRef + arXiv + Semantic Scholar
              │ 未命中
第 2 层  Google Search              Serper Scholar -> SerpAPI Scholar -> Google Scholar Direct
                                    Search API 二选一；Direct 仅在未配置 Search API 时启用
```

### 第 1 层：学术结构化 API

| 检索器 | 数据源 | 官网 / 文档 | 认证 | 额度策略 |
|:--|:--|:--|:--|:--|
| `OpenAlexRetriever` | OpenAlex | [openalex.org](https://openalex.org/) / [Pricing](https://help.openalex.org/hc/en-us/articles/24397762024087-Pricing) | `mailto`（建议） | 免费 API 公开额度为 100,000 calls/day，最多 10 req/s；付费订阅可获得更高额度 |
| `CrossRefRetriever` | CrossRef | [REST API](https://www.crossref.org/documentation/retrieve-metadata/rest-api/) / [Access](https://www.crossref.org/documentation/retrieve-metadata/rest-api/access-and-authentication/) | `mailto`（建议） | Public pool 5 req/s、并发 1；Polite pool 10 req/s、并发 3；Plus pool 150 req/s |
| `ArxivRetriever` | arXiv | [API docs](https://arxiv.org/help/api) / [Terms](https://arxiv.org/help/api/tou) | 无 | 免费；官方 ToU 要求连续请求时不超过每 3 秒 1 次；本项目内置进程级 3.5s 冷却 |
| `SemanticScholarRetriever` | Semantic Scholar | [API](https://www.semanticscholar.org/product/api) / [Tutorial](https://www.semanticscholar.org/product/api/tutorial) | `x-api-key`（本项目中配置后启用） | 多数端点可公共访问但会受共享限流影响；API key 初始额度通常为 1 RPS，可申请更高额度 |

### 第 2 层：Google Search

| 检索器 | 数据源 | 官网 / 文档 | 认证 | 额度策略 |
|:--|:--|:--|:--|:--|
| `SerperRetriever` | Serper Google Scholar | [serper.dev](https://serper.dev/) | API key 必填 | 注册可获得 2,500 次免费查询；付费采用 credits/top-up 模式，Starter 示例为 50k credits、50 QPS |
| `SerpApiRetriever` | SerpAPI Google Scholar | [serpapi.com](https://serpapi.com/) / [Pricing](https://serpapi.com/pricing) | API key 必填 | Free plan 为 250 searches/month、50 throughput/hour；付费套餐按月搜索量与小时吞吐计费 |
| `GoogleScholarRetriever` | Google Scholar Direct | [Google Scholar](https://scholar.google.com/) | 无 API key，需要配置代理 | Google Scholar 没有官方公开 API；本项目仅在未配置 Serper/SerpAPI 时使用直爬兜底，容易触发 CAPTCHA / unusual traffic 拦截 |

> 额度策略会随服务商政策变化，以上仅作为配置选择参考；实际生产使用前请以各平台官网说明为准。

## 判定标签

| 标签 | 含义 |
|:--|:--|
| `VERIFIED` | 论文存在，标题、作者及年份 / venue 等字段基本正确 |
| `VERIFIED_MINOR` | 论文存在，标题和作者正确，但年份或 venue 等次要字段有误 |
| `METADATA_ERROR` | 论文存在，但标题或作者等关键字段不一致 |
| `FABRICATED` | 未检索到对应论文，疑似捏造引用 |
| `UNVERIFIABLE` | 流程异常或关键信息不足，无法完成验证 |

## API 与前端

详见 `docs/api.md`。当前提供：

| 接口 | 方法 | 说明 |
|:--|:--|:--|
| `/api/detect` | `POST` | 一次性返回完整检测报告 |
| `/api/detect/stream` | `POST` | SSE 流式推送抽取结果、逐条引用判定与最终报告 |
| `/api/health` | `GET` | 健康检查 |
| `/api/retrieval/config` | `GET` | 返回当前实际生效的检索配置状态 |

`/api/detect/stream` 事件顺序：

```text
extraction_done -> citation_verdict（逐条） -> report_complete
```

## 输出格式

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

## 项目结构

```text
src/
├── agents/            # 引用抽取、文献检索、元数据比对、报告生成
├── retrievers/        # 学术 API、搜索 API、Google Scholar 爬虫、页面解析、级联编排
├── utils/             # LLM 客户端、相似度计算、姓名匹配、配置读取
└── models/            # Pydantic 数据模型

app/
├── api.py             # FastAPI 后端与静态页面入口
└── static/            # Web 前端

config/
├── config.yaml         # 主配置文件
└── config.example.yaml # 配置示例

docs/
└── api.md              # 接口与检索策略文档

tests/                  # 测试用例与调试脚本
```

---

<p align="center">
  <sub>如果 HalluRef 对你的研究有帮助，欢迎点个 ⭐ 支持一下</sub>
</p>
