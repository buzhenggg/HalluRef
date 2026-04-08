"""直接 HTTP 爬取 Google Scholar 测试

不依赖 scholarly 库, 用 httpx + BeautifulSoup 解析
测试返回内容、限速、被封时机
"""

import asyncio
import time
import httpx
from bs4 import BeautifulSoup

PROXY = "http://127.0.0.1:7890"
BASE = "https://scholar.google.com/scholar"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

QUERIES = [
    "Attention Is All You Need",
    "BERT Pre-training of Deep Bidirectional Transformers",
    "Deep Residual Learning for Image Recognition",
    "Generative Adversarial Networks Goodfellow",
    "ImageNet Classification Krizhevsky",
    "Adam Method Stochastic Optimization Kingma",
    "知识图谱构建综述",  # 中文测试
    "深度学习在医学影像中的应用",
]


def parse_results(html: str) -> list[dict]:
    """解析 Google Scholar 搜索结果页"""
    soup = BeautifulSoup(html, "html.parser")
    results = []
    for div in soup.select("div.gs_ri"):
        # 标题
        title_tag = div.select_one("h3.gs_rt")
        title = title_tag.get_text(strip=True) if title_tag else ""
        link = ""
        if title_tag and title_tag.find("a"):
            link = title_tag.find("a").get("href", "")

        # 作者/年份/期刊行: "Author1, Author2 - Venue, Year - publisher"
        meta_tag = div.select_one("div.gs_a")
        meta = meta_tag.get_text(" ", strip=True) if meta_tag else ""

        # 摘要片段
        snippet_tag = div.select_one("div.gs_rs")
        snippet = snippet_tag.get_text(" ", strip=True) if snippet_tag else ""

        # 引用数
        cited_by = ""
        for fl in div.select("div.gs_fl a"):
            t = fl.get_text(strip=True)
            if t.startswith("Cited by") or t.startswith("被引用次数"):
                cited_by = t

        results.append({
            "title": title,
            "link": link,
            "meta": meta,
            "snippet": snippet,
            "cited_by": cited_by,
        })
    return results


def detect_block(html: str) -> str | None:
    """检测是否被 Google 封禁/挑战"""
    if "captcha" in html.lower() or "unusual traffic" in html.lower():
        return "CAPTCHA challenge"
    if "Our systems have detected unusual traffic" in html:
        return "Unusual traffic block"
    if "sorry/index" in html:
        return "Sorry page (blocked)"
    if len(html) < 1000:
        return f"Suspiciously short response ({len(html)} bytes)"
    return None


async def fetch_one(client: httpx.AsyncClient, query: str) -> dict:
    """单次请求"""
    t0 = time.time()
    try:
        r = await client.get(
            BASE,
            params={"q": query, "hl": "en"},
            headers=HEADERS,
            follow_redirects=True,
        )
        elapsed = time.time() - t0

        block = detect_block(r.text)
        if block:
            return {"query": query, "elapsed": elapsed, "status": r.status_code,
                    "blocked": block, "results": []}

        results = parse_results(r.text)
        return {"query": query, "elapsed": elapsed, "status": r.status_code,
                "blocked": None, "results": results, "html_size": len(r.text)}
    except Exception as e:
        return {"query": query, "elapsed": time.time() - t0, "status": -1,
                "blocked": f"exception: {e}", "results": []}


async def bench_sequential():
    print("=" * 70)
    print("Sequential test — 8 queries (1s gap between)")
    print("=" * 70)
    async with httpx.AsyncClient(proxy=PROXY, timeout=20) as client:
        success = 0
        blocked = 0
        empty = 0
        timings = []
        total_start = time.time()
        for i, q in enumerate(QUERIES, 1):
            r = await fetch_one(client, q)
            timings.append(r["elapsed"])
            if r["blocked"]:
                blocked += 1
                print(f"[{i}/8] BLOCKED  {r['elapsed']:5.2f}s — {r['blocked']} — {q[:45]}")
            elif r["results"]:
                success += 1
                print(f"[{i}/8] OK       {r['elapsed']:5.2f}s ({len(r['results'])} hits, {r.get('html_size',0)//1024}KB) — {q[:45]}")
                top = r["results"][0]
                print(f"        → {top['title'][:70]}")
                print(f"        → {top['meta'][:70]}")
                if top["snippet"]:
                    print(f"        → {top['snippet'][:70]}...")
                if top["cited_by"]:
                    print(f"        → {top['cited_by']}")
            else:
                empty += 1
                print(f"[{i}/8] EMPTY    {r['elapsed']:5.2f}s — {q[:45]}")
            await asyncio.sleep(1)
        total = time.time() - total_start

    print("-" * 70)
    print(f"Total:    {total:.2f}s")
    print(f"Success:  {success}/8")
    print(f"Blocked:  {blocked}/8")
    print(f"Empty:    {empty}/8")
    if timings:
        print(f"Per-req:  avg={sum(timings)/len(timings):.2f}s "
              f"min={min(timings):.2f}s max={max(timings):.2f}s")
    print()
    return blocked == 0


async def bench_burst():
    print("=" * 70)
    print("Burst test — 6 queries with NO gap, watch for blocking")
    print("=" * 70)
    async with httpx.AsyncClient(proxy=PROXY, timeout=20) as client:
        for i in range(6):
            q = QUERIES[i % len(QUERIES)]
            r = await fetch_one(client, q)
            status = r["blocked"] or (f"OK ({len(r['results'])} hits)" if r["results"] else "EMPTY")
            print(f"  burst[{i+1}] {r['elapsed']:5.2f}s  HTTP {r['status']:3d}  {status}")
    print()


async def bench_concurrent():
    print("=" * 70)
    print("Concurrent test — 4 queries in parallel")
    print("=" * 70)
    async with httpx.AsyncClient(proxy=PROXY, timeout=20) as client:
        t0 = time.time()
        results = await asyncio.gather(*[fetch_one(client, q) for q in QUERIES[:4]])
        elapsed = time.time() - t0

    for r in results:
        status = r["blocked"] or (f"OK ({len(r['results'])} hits)" if r["results"] else "EMPTY")
        print(f"  {r['elapsed']:5.2f}s  HTTP {r['status']:3d}  {status} — {r['query'][:50]}")
    print("-" * 70)
    print(f"Total: {elapsed:.2f}s")
    print()


async def main():
    print(f"\nProxy: {PROXY}\n")
    ok = await bench_sequential()
    if not ok:
        print(">>> Sequential test got blocked, stopping further tests <<<")
        return
    print("Sleeping 5s before concurrent test...\n")
    await asyncio.sleep(5)
    await bench_concurrent()
    print("Sleeping 5s before burst test...\n")
    await asyncio.sleep(5)
    await bench_burst()


if __name__ == "__main__":
    asyncio.run(main())
