"""压力测试: 各检索器在 0.5 req/s 下的稳定性

对每个接口连续发起 N 次真实请求, 间隔 2s (0.5 req/s),
记录: 成功数 / 失败数 / 平均延迟 / 错误信息

运行:
    conda run -n halluref python tests/bench_retrievers_rate.py
可选环境变量:
    SERPER_API_KEY  - 启用 Serper 测试
    S2_API_KEY      - Semantic Scholar 用 key 加速 (可选)
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrievers.arxiv import ArxivRetriever
from src.retrievers.crossref import CrossRefRetriever
from src.retrievers.openalex import OpenAlexRetriever
from src.retrievers.semantic_scholar import SemanticScholarRetriever
from src.retrievers.serper import SerperRetriever


REQUEST_INTERVAL = 2.0   # 2 秒一次 = 0.5 req/s
N_REQUESTS = 5

# 5 个真实存在的论文标题, 覆盖不同领域
QUERIES = [
    "Attention Is All You Need",
    "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    "Deep Residual Learning for Image Recognition",
    "LoRA: Low-Rank Adaptation of Large Language Models",
    "Denoising Diffusion Probabilistic Models",
]


@dataclass
class BenchResult:
    name: str
    latencies: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    n_results: list[int] = field(default_factory=list)

    @property
    def success(self) -> int:
        return len(self.latencies)

    @property
    def fail(self) -> int:
        return len(self.errors)

    @property
    def avg_latency(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0

    @property
    def max_latency(self) -> float:
        return max(self.latencies) if self.latencies else 0.0

    @property
    def total_results(self) -> int:
        return sum(self.n_results)


async def bench_one(name: str, retriever, queries: list[str]) -> BenchResult:
    res = BenchResult(name=name)
    print(f"\n[{name}] starting {len(queries)} requests @ {1/REQUEST_INTERVAL:.2f} req/s ...")
    for i, q in enumerate(queries):
        if i > 0:
            await asyncio.sleep(REQUEST_INTERVAL)
        t0 = time.perf_counter()
        try:
            papers = await retriever.search_by_title(q)
            dt = time.perf_counter() - t0
            res.latencies.append(dt)
            res.n_results.append(len(papers))
            print(f"  [{name}] #{i+1} OK   {dt:6.2f}s  results={len(papers)}  '{q[:40]}'")
        except Exception as e:
            dt = time.perf_counter() - t0
            err = f"{type(e).__name__}: {e}"
            res.errors.append(err)
            print(f"  [{name}] #{i+1} FAIL {dt:6.2f}s  {err}")
    try:
        await retriever.close()
    except Exception:
        pass
    return res


def print_summary(results: list[BenchResult]) -> None:
    print("\n" + "=" * 78)
    print(f"{'Retriever':<22} {'OK':>4} {'FAIL':>5} {'avg(s)':>8} {'max(s)':>8} {'#hits':>6}")
    print("-" * 78)
    for r in results:
        print(
            f"{r.name:<22} {r.success:>4} {r.fail:>5} "
            f"{r.avg_latency:>8.2f} {r.max_latency:>8.2f} {r.total_results:>6}"
        )
    print("=" * 78)
    for r in results:
        if r.errors:
            print(f"\n[{r.name}] errors:")
            for e in r.errors:
                print(f"  - {e}")


async def main():
    retrievers: list[tuple[str, object]] = []

    # OpenAlex (mailto 进入 polite pool)
    retrievers.append((
        "openalex",
        OpenAlexRetriever(mailto="buzheng202108@gmail.com"),
    ))

    # CrossRef
    retrievers.append((
        "crossref",
        CrossRefRetriever(mailto="buzheng202108@gmail.com"),
    ))

    # Semantic Scholar
    retrievers.append((
        "semantic_scholar",
        SemanticScholarRetriever(api_key=os.getenv("S2_API_KEY")),
    ))

    # arXiv
    retrievers.append((
        "arxiv",
        ArxivRetriever(),
    ))

    # Serper (需 key) — 优先 env var, 否则从 config/config.yaml 读
    serper_key = os.getenv("SERPER_API_KEY")
    if not serper_key:
        try:
            from src.utils.config import load_config
            serper_key = (load_config().get("retriever", {}).get("serper", {}) or {}).get("api_key") or None
        except Exception:
            serper_key = None
    if serper_key:
        retrievers.append((
            "serper",
            SerperRetriever(api_key=serper_key),
        ))
    else:
        print("[serper] skipped — set SERPER_API_KEY to enable")

    # 顺序运行 (而非并发), 避免互相干扰
    results = []
    for name, r in retrievers:
        results.append(await bench_one(name, r, QUERIES[:N_REQUESTS]))

    print_summary(results)


if __name__ == "__main__":
    asyncio.run(main())
