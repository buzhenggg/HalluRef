<p align="center">
  <h1 align="center">🔍 HalluRef</h1>
  <p align="center">
    <strong>面向 LLM 生成学术文本的多智能体幻觉引用检测框架</strong>
  </p>
</p>


---

大语言模型在生成文献综述时经常产生幻觉引用。HalluRef 可自动检测并分类三类引用错误：

| 类型 | 说明 |
|:--|:--|
| **完全捏造** | 引用的论文根本不存在——作者、标题或期刊均为虚构 |
| **信息篡改** | 论文存在，但元数据有误（年份、作者、期刊等） |
| **观点歪曲** | 论文存在，但对其观点的引述失真或被夸大 |

HalluRef 通过 5 阶段智能体流水线，为每条引用输出结构化判定结果。

## ✨ 核心功能

- **引用提取** — LLM 结构化解析整篇文本，支持 `Author (year)`、`(Author, year)`、`[N]` 等格式；输入长度受 `max_input_chars` 限制（默认 20000，超限直接拒绝）
- **三层级联检索** — 学术 API → Web 搜索 → 爬虫兜底；命中即停
- **元数据比对** — 基于 `rapidfuzz` 相似度算法校验标题/作者/年份/期刊
- **内容一致性核查** — LLM NLI 判断引用声明是否忠于论文摘要
- **五类判定** — `VERIFIED` · `FABRICATED` · `METADATA_ERROR` · `MISREPRESENTED` · `UNVERIFIABLE`
- **多种界面** — FastAPI + SSE 流式单页应用、Gradio 界面、命令行

## 快速开始

### 1. 安装

```bash
git clone <repo_url>
cd Hallucinated_citations_detection

conda create -n halluref python=3.11 -y
conda activate halluref

pip install -r requirements.txt
```

### 2. 配置

编辑 `config/config.yaml`：

```yaml
llm:
  base_url: "https://api.deepseek.com"   # 任意 OpenAI 兼容端点
  api_key: "sk-xxx"
  model: "deepseek-chat"

retriever:
  openalex:
    mailto: "your@email.com"             # 可选，进入 polite pool
  crossref:
    mailto: "your@email.com"             # 可选
  serper:
    api_key: "xxx"                       # 可选 — https://serper.dev 注册送 2500 次
  google_scholar:
    enabled: true
    proxy: "http://127.0.0.1:7890"       # 建议配代理避免封 IP

detection:
  max_input_chars: 20000                 # 单次检测最大字符数，超限拒绝
```

> **最低要求：** 需配置 llm 接口，OpenAlex + CrossRef 即可运行（免费、无需注册），但最好配置 mailto，即输入自己的邮箱，否则限速严重。其余检索源均为可选。

> **输入长度限制：** 默认上限 20000 字符。不建议设置太高，越高模型识别引用准确率越低。

### 3. 运行

```bash
# Web 界面（推荐）
uvicorn app.api:app --host 127.0.0.1 --port 8000

# Gradio 界面
python app/ui.py

# 命令行
python -m src.agents.orchestrator --input your_text.txt --output report.json
```

## 检索流水线

HalluRef 采用 **三层级联检索策略**——任一层命中即停，未配置的源自动跳过。

```
第 1 层 ─ 学术 API（并行）      OpenAlex + CrossRef
             │ 未命中
第 2 层 ─ Web 搜索              Serper → SerpAPI（二选一，Serper 优先）
             │ 未命中
第 3 层 ─ 爬虫兜底              Google Scholar（httpx + BeautifulSoup）
```

### 第 1 层 — 学术结构化 API（免费，并行）

OpenAlex 与 CrossRef 通过 `asyncio.gather` 并发调用，候选合并后统一打分。

| 检索器 | 数据源 | 认证 | 备注 |
|:--|:--|:--|:--|
| `OpenAlexRetriever` | OpenAlex | `mailto`（可选） | 2 亿+ 论文，10 req/s |
| `CrossRefRetriever` | CrossRef | `mailto`（可选） | DOI 权威库，polite pool |

> Semantic Scholar / arXiv 检索器代码已实现 (`src/retrievers/`)，可单独使用，但默认未接入级联流水线（无 API key 时 SS 限流严重）。

### 第 2 层 — Web 搜索 API（付费/有免费额度）

| 检索器 | 数据源 | 认证 | 免费额度 |
|:--|:--|:--|:--|
| `SerperRetriever` | Serper（Google Scholar） | API key 必填 | 2,500 次 |
| `SerpApiRetriever` | SerpAPI（Google） | API key 必填 | 100 次/月 |

### 第 3 层 — 爬虫兜底

| 检索器 | 数据源 | 方式 | 备注 |
|:--|:--|:--|:--|
| `GoogleScholarRetriever` | Google Scholar | 直接爬取 | 支持 HTTP/SOCKS5 代理，中文文献覆盖佳 |

## LLM 后端

| 智能体 | 用途 | 协议 |
|:--|:--|:--|
| Agent 1 | 引用结构化解析 | OpenAI 兼容 |
| Agent 4 | 内容 NLI 核查 | OpenAI 兼容 |

可替换为 **OpenAI / Qwen / Moonshot / Kimi / Ollama** 等任意 OpenAI 兼容端点，只需修改 `config.yaml` 中的 `llm` 部分。

## 输出格式

```json
{
  "total_citations": 3,
  "verified": 1,
  "fabricated": 1,
  "metadata_error": 1,
  "details": [
    {
      "citation_id": "ref_001",
      "verdict": "FABRICATED",
      "confidence": 0.95,
      "evidence": "在所有学术数据源中均未检索到该论文",
      "suggestion": "建议删除该引用或替换为真实文献"
    }
  ]
}
```

## 项目结构

```
src/
├── agents/            # 5 个流水线组件
├── retrievers/        # 7 个检索器 + 级联编排
├── utils/             # LLM 客户端、相似度、姓名匹配
└── models/            # Pydantic 数据模型
app/
├── api.py             # FastAPI 后端
├── static/            # Web 前端
└── ui.py              # Gradio 备选界面
config/config.yaml     # 主配置文件
tests/                 # 测试用例
```

---

<p align="center">
  <sub>如果 HalluRef 对你的研究有帮助，欢迎点个 ⭐ 支持一下</sub>
</p>
