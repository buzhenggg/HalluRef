"""Gradio 前端 — 可视化检测报告"""

from __future__ import annotations

import asyncio
import json

import gradio as gr

from src.agents.orchestrator import HalluRefPipeline

pipeline: HalluRefPipeline | None = None


def get_pipeline() -> HalluRefPipeline:
    global pipeline
    if pipeline is None:
        pipeline = HalluRefPipeline()
    return pipeline


def detect(text: str) -> tuple[str, str]:
    """运行检测并返回 (摘要, 详细报告JSON)"""
    if not text.strip():
        return "请输入包含引用的学术文本", ""

    p = get_pipeline()
    report = asyncio.run(p.run(text))

    # 摘要
    summary_lines = [
        f"## 检测结果摘要",
        f"- 总引用数: **{report.total_citations}**",
        f"- 引用正确 ✅: **{report.verified}**",
        f"- 引用基本正确 ☑️: **{report.verified_minor}**",
        f"- 完全捏造 🚫: **{report.fabricated}**",
        f"- 信息篡改 ⚠️: **{report.metadata_error}**",
        f"- 观点歪曲 📝: **{report.misrepresented}**",
        f"- 无法验证 ❓: **{report.unverifiable}**",
        "",
        "---",
        "",
    ]

    # 逐条详情
    for v in report.details:
        icon = {
            "VERIFIED": "✅",
            "VERIFIED_MINOR": "☑️",
            "FABRICATED": "🚫",
            "METADATA_ERROR": "⚠️",
            "MISREPRESENTED": "📝",
            "UNVERIFIABLE": "❓",
        }.get(v.verdict.value, "")

        summary_lines.append(
            f"### {v.citation_id} {icon} {v.verdict.value} (置信度: {v.confidence:.0%})"
        )
        if v.raw_text:
            summary_lines.append(f"> {v.raw_text[:200]}")
        summary_lines.append(f"**依据**: {v.evidence}")
        if v.suggestion:
            summary_lines.append(f"**建议**: {v.suggestion}")
        summary_lines.append("")

    summary_md = "\n".join(summary_lines)
    detail_json = report.model_dump_json(indent=2, ensure_ascii=False)

    return summary_md, detail_json


def main():
    with gr.Blocks(title="HalluRef - 幻觉引用检测") as demo:
        gr.Markdown("# HalluRef — 基于多智能体事实核查的 LLM 幻觉引用检测")
        gr.Markdown("输入 LLM 生成的包含引用的学术文本, 系统将自动检测幻觉引用。")

        with gr.Row():
            with gr.Column(scale=1):
                text_input = gr.Textbox(
                    label="输入文本",
                    placeholder="粘贴包含引用的学术文本...",
                    lines=15,
                )
                detect_btn = gr.Button("开始检测", variant="primary")

            with gr.Column(scale=1):
                summary_output = gr.Markdown(label="检测结果")

        detail_output = gr.Code(label="详细报告 (JSON)", language="json")

        detect_btn.click(
            fn=detect,
            inputs=[text_input],
            outputs=[summary_output, detail_output],
        )

    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
