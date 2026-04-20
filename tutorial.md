# HalluRef 教程

## 一、项目是什么

**HalluRef** 是一个用于检测大语言模型（LLM）生成学术文本中**幻觉引用**的多智能体事实核查框架。针对 LLM 在写文献综述/研究现状时常见的三类引用问题：

- **完全捏造**：虚构作者、标题、期刊的不存在论文
- **信息篡改**：论文存在但元数据错误（年份、作者、期刊等）
- **观点歪曲**：论文存在但对其观点的引述失真（夸大/扭曲）

HalluRef 通过 5 个组件流水线自动化识别并分类，输出结构化检测报告。

## 二、核心功能

1. **引用提取**：正则 + LLM 结构化解析，支持 `Author (year)` / `(Author, year)` / `[N]` 多种格式
2. **级联文献检索**：3 层级联策略，命中即停，覆盖学术 API / Web 搜索 / 爬虫兜底
3. **元数据比对**：rapidfuzz 相似度算法校验标题/作者/年份/期刊
4. **元数据比对**：校验标题、作者、年份、venue 等字段是否一致
5. **分类决策**：输出 5 类判定 —— VERIFIED / VERIFIED_MINOR / FABRICATED / METADATA_ERROR / UNVERIFIABLE
6. **前端**：FastAPI + SSE 流式单页应用
7. **命令行模式**：直接对输入文本文件生成 JSON 报告

## 三、快速启动

### 1. 环境准备

```bash
git clone <repo_url>
cd Hallucinated_citations_detection

# 创建 conda 虚拟环境（项目强制要求名为 halluref）
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
    mailto: "your@email.com"             # 可选，进 polite pool
  crossref:
    mailto: "your@email.com"             # 可选
  serper:
    api_key: "xxx"                       # 可选，https://serper.dev 注册送 2500 次
  proxy:
    server: "http://127.0.0.1:7890"       # 统一代理，供直爬和落地页解析使用

  google_scholar:
    timeout: 15
    max_results: 3
```

> 只要 OpenAlex + CrossRef 可用即可运行，其它检索源均可选。

### 3. 三种运行方式

**① Web 服务（推荐）**
```bash
uvicorn app.api:app --host 127.0.0.1 --port 8000
```
浏览器打开 http://127.0.0.1:8000 ，点击「加载示例」体验流式检测。

**② 命令行**
```bash
python -m src.agents.orchestrator --input your_text.txt --output report.json
```

## 四、REST API 速览

| 接口 | 方法 | 说明 |
|---|---|---|
| `/api/detect` | POST | 一次性返回完整检测报告，body: `{"text": "..."}` |
| `/api/detect/stream` | POST | SSE 流式推送，事件：`extraction_done` → `citation_verdict` (逐条) → `report_complete` |
| `/api/health` | GET | 健康检查 |

示例：
```bash
curl -X POST http://127.0.0.1:8000/api/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "Smith et al. (2020) proposed a novel method..."}'
```

## 五、支持的学术数据库与搜索 API

HalluRef 采用 **3 层级联检索策略**，任一层命中即停，未配置的源自动跳过：

```
Tier 1  学术 API 并行   OpenAlex + CrossRef
            │ 未命中 ↓
Tier 2  Web 搜索串行    Serper → SerpAPI  (二选一)
            │ 未命中 ↓
Tier 3  爬虫兜底        Google Scholar 直爬
```

### Tier 1：学术结构化 API（免费、无需注册）

| 检索器 | 数据源 | 端点 | 认证 | 说明 |
|---|---|---|---|---|
| `OpenAlexRetriever` | **OpenAlex** | `api.openalex.org/works` | mailto（可选） | 2 亿+ 论文，倒排索引重建摘要，10 req/s |
| `CrossRefRetriever` | **CrossRef** | `api.crossref.org/works` | mailto（可选） | DOI 权威库，polite pool 限流宽松 |
| `SemanticScholarRetriever` | **Semantic Scholar** | `api.semanticscholar.org/graph/v1/paper/search` | x-api-key（可选） | 无 key 1 req/s，有 key 10 req/s |
| `ArxivRetriever` | **arXiv** | `export.arxiv.org/api/query` | 无 | 预印本，Atom XML，建议 3s 间隔 |

### Tier 2：Web 搜索 API（付费/有免费额度）

| 检索器 | 数据源 | 端点 | 认证 | 免费额度 |
|---|---|---|---|---|
| `SerperRetriever` | **Serper**（Google Scholar） | `POST google.serper.dev/scholar` | X-API-KEY 必填 | 注册送 2500 次 |
| `SerpApiRetriever` | **SerpAPI**（Google） | `GET serpapi.com/search.json` | api_key 必填 | 100 次/月 |

> Tier 2 中 Serper 优先，仅当 Serper 未配置时启用 SerpAPI。

### Tier 3：爬虫兜底

| 检索器 | 数据源 | 方式 | 备注 |
|---|---|---|---|
| `GoogleScholarRetriever` | **Google Scholar** | `httpx + BeautifulSoup` 直爬 | 支持 HTTP/SOCKS5 代理，中文文献覆盖佳 |

### 匹配评分

- 基于 `rapidfuzz` 计算标题相似度
- 阈值：`≥0.95 HIGH` / `≥0.85 MEDIUM` / `≥0.6 LOW`
- HIGH/MEDIUM 视为命中
- 同分 tiebreak：`openalex > crossref > arxiv > semantic_scholar > serper_scholar > serpapi_scholar > google_scholar`

### 鲁棒性保障

- 未配置检索器（缺 api_key）自动跳过
- 单源超时/异常被吞掉，不阻塞其他源
- 指数退避重试（429/5xx）+ 请求间隔限流
- 标题检索无结果时自动按 author+year 回退

## 六、LLM 接口

| 用途 | 协议 | 默认 |
|---|---|---|
| Agent 1 引用结构化解析 | OpenAI 兼容（base_url + api_key + model） | DeepSeek |
| Agent 4 内容 NLI 核查 | 同上 | DeepSeek |

可替换为 **OpenAI / Qwen / Moonshot / Kimi / Ollama** 等任意 OpenAI 兼容端点，只需修改 `config.yaml` 的 `llm` 段。

## 七、输出示例

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

## 八、项目结构

```
src/
├── agents/          # 5 个流水线组件
├── retrievers/      # 7 个检索器 + 级联编排
├── utils/           # LLM 客户端 / 相似度 / 姓名匹配
└── models/          # Pydantic 数据模型
app/
├── api.py           # FastAPI 后端
├── static/          # Web 前端
config/config.yaml   # 主配置
tests/               # 测试用例
```

