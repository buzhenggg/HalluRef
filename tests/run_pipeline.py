"""运行完整流水线并计时"""

import asyncio
import time
from pathlib import Path

from src.agents.orchestrator import HalluRefPipeline


async def main():
    text = Path("tests/sample_input.txt").read_text(encoding="utf-8")
    pipeline = HalluRefPipeline()

    start = time.perf_counter()
    try:
        report = await pipeline.run(text)
    finally:
        await pipeline.close()
    elapsed = time.perf_counter() - start

    print()
    print("=" * 60)
    print(f"Total time: {elapsed:.2f}s")
    print(f"Citations:  {report.total_citations}")
    print(f"Verified:   {report.verified}")
    print(f"Fabricated: {report.fabricated}")
    print(f"MetaError:  {report.metadata_error}")
    print(f"Misrepresented: {report.misrepresented}")
    print(f"Unverifiable:   {report.unverifiable}")
    print("=" * 60)
    for d in report.details:
        raw = d.raw_text[:70].replace("\n", " ")
        print(f"  {d.citation_id}: {d.verdict.value:<16} conf={d.confidence:.2f}  {raw}...")


asyncio.run(main())
