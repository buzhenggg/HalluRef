"""GoogleScholarRetriever 真实端到端基准测试

用新的 httpx 直接爬取实现, 验证速度和成功率
手动运行: python tests/bench_google_scholar.py
"""

import asyncio
import time

from src.retrievers.google_scholar import GoogleScholarRetriever

PROXY = "http://127.0.0.1:7890"

QUERIES = [
    "Attention Is All You Need",
    "BERT Pre-training of Deep Bidirectional Transformers",
    "Deep Residual Learning for Image Recognition",
    "Generative Adversarial Networks Goodfellow",
    "知识图谱构建综述",
    "深度学习在医学影像中的应用",
]


async def main():
    print(f"\nProxy: {PROXY}\n")
    retriever = GoogleScholarRetriever(proxy=PROXY, timeout=20, max_results=3)

    print("=" * 70)
    print("Sequential — using new GoogleScholarRetriever")
    print("=" * 70)
    success = 0
    timings = []
    t_total = time.time()
    for i, q in enumerate(QUERIES, 1):
        t0 = time.time()
        papers = await retriever.search_by_title(q)
        elapsed = time.time() - t0
        timings.append(elapsed)
        if papers:
            success += 1
            p = papers[0]
            print(f"[{i}/{len(QUERIES)}] OK {elapsed:5.2f}s ({len(papers)} hits) — {q[:45]}")
            print(f"        title:   {p.title[:70]}")
            print(f"        authors: {', '.join(p.authors[:3])}")
            print(f"        year:    {p.year}    venue: {(p.venue or '')[:50]}")
            if p.abstract:
                print(f"        abstr:   {p.abstract[:70]}...")
        else:
            print(f"[{i}/{len(QUERIES)}] EMPTY {elapsed:5.2f}s — {q[:45]}")
    total = time.time() - t_total

    print("-" * 70)
    print(f"Total: {total:.2f}s    Success: {success}/{len(QUERIES)}")
    print(f"Per-query: avg={sum(timings)/len(timings):.2f}s min={min(timings):.2f}s max={max(timings):.2f}s")

    await retriever.close()


if __name__ == "__main__":
    asyncio.run(main())
