"""Semantic Scholar 稳定性测试

策略: 串行检索 N 条不同标题，每次调用结束后随机 sleep 1-2s，
统计成功 / 失败 / 命中率。
"""

from __future__ import annotations

import asyncio
import random
import time

from src.retrievers.semantic_scholar import SemanticScholarRetriever


TEST_TITLES = [
    "Attention Is All You Need",
    "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    "Language Models are Few-Shot Learners",
    "LLaMA: Open and Efficient Foundation Language Models",
    "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
    "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    "REALM: Retrieval-Augmented Language Model Pre-Training",
    "Deep Residual Learning for Image Recognition",
    "ImageNet Classification with Deep Convolutional Neural Networks",
    "Generative Adversarial Networks",
    "Adam: A Method for Stochastic Optimization",
    "Dropout: A Simple Way to Prevent Neural Networks from Overfitting",
    "Sequence to Sequence Learning with Neural Networks",
    "Distributed Representations of Words and Phrases and their Compositionality",
    "Long Short-Term Memory",
]


async def main():
    retriever = SemanticScholarRetriever()
    success = 0
    fail = 0
    hit = 0
    timings: list[float] = []

    print(f"\n开始测试 Semantic Scholar API（{len(TEST_TITLES)} 条查询，每次后随机等待 1-2s）\n")
    print(f"{'#':<4}{'状态':<8}{'耗时':<10}{'命中':<8}标题")
    print("-" * 90)

    for i, title in enumerate(TEST_TITLES, 1):
        t0 = time.perf_counter()
        try:
            results = await retriever.search_by_title(title)
            elapsed = time.perf_counter() - t0
            timings.append(elapsed)
            success += 1
            hit_flag = "✓" if results else "✗"
            if results:
                hit += 1
            print(f"{i:<4}{'OK':<8}{elapsed*1000:>6.0f}ms  {hit_flag:<8}{title[:60]}")
        except Exception as e:
            elapsed = time.perf_counter() - t0
            fail += 1
            print(f"{i:<4}{'FAIL':<8}{elapsed*1000:>6.0f}ms  {'-':<8}{title[:60]}  → {type(e).__name__}: {e}")

        if i < len(TEST_TITLES):
            wait = random.uniform(1.0, 2.0)
            await asyncio.sleep(wait)

    await retriever.close()

    print("-" * 90)
    print(f"\n汇总:")
    print(f"  总查询: {len(TEST_TITLES)}")
    print(f"  成功:   {success} ({success/len(TEST_TITLES)*100:.0f}%)")
    print(f"  失败:   {fail}")
    print(f"  命中:   {hit} ({hit/len(TEST_TITLES)*100:.0f}%)")
    if timings:
        print(f"  平均耗时: {sum(timings)/len(timings)*1000:.0f}ms")
        print(f"  最快/最慢: {min(timings)*1000:.0f}ms / {max(timings)*1000:.0f}ms")


if __name__ == "__main__":
    asyncio.run(main())
