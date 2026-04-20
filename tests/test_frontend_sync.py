"""前端静态页与后端判定模型保持一致。"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_HTML = PROJECT_ROOT / "app" / "static" / "index.html"


def test_frontend_only_uses_supported_verdicts():
    html = INDEX_HTML.read_text(encoding="utf-8")

    assert "VERIFIED_MINOR" in html
    assert "MISREPRESENTED" not in html


def test_frontend_removes_content_check_copy():
    html = INDEX_HTML.read_text(encoding="utf-8")

    assert "内容核查" not in html
    assert "内容一致性" not in html
    assert "观点歪曲" not in html


def test_frontend_progress_matches_current_pipeline():
    html = INDEX_HTML.read_text(encoding="utf-8")

    assert "引用提取" in html
    assert "文献检索" in html
    assert "元数据比对" in html
    assert "综合研判" in html


def test_frontend_shows_source_and_hides_confidence_copy():
    html = INDEX_HTML.read_text(encoding="utf-8")

    assert "来源" in html
    assert "置信度" not in html


def test_frontend_reads_detection_limit_from_backend_config():
    html = INDEX_HTML.read_text(encoding="utf-8")

    assert "retrieval-config-panel" not in html
    assert "/api/retrieval/config" in html
    assert "data.detection?.max_input_chars" in html
    assert "MAX_INPUT_CHARS = limit" in html


def test_frontend_reorders_with_final_report_on_completion():
    html = INDEX_HTML.read_text(encoding="utf-8")

    assert "report_complete(report)" in html
    assert "renderFullCitations(report)" in html


def test_frontend_warns_against_uploading_full_text_when_refs_suffice():
    html = INDEX_HTML.read_text(encoding="utf-8")

    assert "建议优先粘贴或上传参考文献 / 引用部分" in html
    assert "降低引用识别精准度" in html


def test_static_architecture_page_removed():
    assert not (PROJECT_ROOT / "app" / "static" / "architecture.html").exists()
