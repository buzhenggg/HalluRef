"""Tests for PDF/DOCX upload text extraction API."""

from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.api import app
from src.utils.document_text import extract_document_text


def _docx_bytes() -> bytes:
    docx = pytest.importorskip("docx")

    document = docx.Document()
    document.add_paragraph("Paragraph one.")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Table A"
    table.cell(0, 1).text = "Table B"
    stream = BytesIO()
    document.save(stream)
    return stream.getvalue()


def test_docx_upload_extracts_paragraphs_and_table_text():
    with TestClient(app) as client:
        resp = client.post(
            "/api/extract-file",
            files={
                "file": (
                    "paper.docx",
                    _docx_bytes(),
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["filename"] == "paper.docx"
    assert "Paragraph one." in data["text"]
    assert "Table A" in data["text"]
    assert "Table B" in data["text"]
    assert data["char_count"] == len(data["text"])
    assert data["over_limit"] is False


def test_pdf_upload_uses_pdf_reader_and_returns_text():
    fake_page = SimpleNamespace(extract_text=lambda: "PDF page text")
    fake_reader = SimpleNamespace(pages=[fake_page])

    with patch("src.utils.document_text.PdfReader", return_value=fake_reader):
        with TestClient(app) as client:
            resp = client.post(
                "/api/extract-file",
                files={"file": ("paper.pdf", b"%PDF fake", "application/pdf")},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["filename"] == "paper.pdf"
    assert data["text"] == "PDF page text"
    assert data["content_type"] == "application/pdf"


def test_unsupported_doc_returns_415():
    with TestClient(app) as client:
        resp = client.post(
            "/api/extract-file",
            files={"file": ("legacy.doc", b"legacy", "application/msword")},
        )

    assert resp.status_code == 415
    assert "Unsupported" in resp.json()["message"]


def test_empty_extraction_returns_422():
    with patch("src.utils.document_text.extract_document_text", return_value=""):
        with TestClient(app) as client:
            resp = client.post(
                "/api/extract-file",
                files={"file": ("paper.pdf", b"%PDF fake", "application/pdf")},
            )

    assert resp.status_code == 422


def test_oversized_upload_returns_413():
    original = app.state.max_upload_bytes if hasattr(app.state, "max_upload_bytes") else None
    app.state.max_upload_bytes = 4
    try:
        with TestClient(app) as client:
            resp = client.post(
                "/api/extract-file",
                files={"file": ("paper.pdf", b"12345", "application/pdf")},
            )
    finally:
        if original is None:
            delattr(app.state, "max_upload_bytes")
        else:
            app.state.max_upload_bytes = original

    assert resp.status_code == 413


def test_over_limit_flag_uses_input_limit():
    original = app.state.max_input_chars if hasattr(app.state, "max_input_chars") else None
    app.state.max_input_chars = 3
    try:
        with patch("src.utils.document_text.extract_document_text", return_value="abcd"):
            with TestClient(app) as client:
                resp = client.post(
                    "/api/extract-file",
                    files={"file": ("paper.pdf", b"%PDF fake", "application/pdf")},
                )
    finally:
        if original is None:
            delattr(app.state, "max_input_chars")
        else:
            app.state.max_input_chars = original

    assert resp.status_code == 200
    data = resp.json()
    assert data["text"] == "abcd"
    assert data["char_count"] == 4
    assert data["max_input_chars"] == 3
    assert data["over_limit"] is True


def test_document_text_normalizes_blank_lines():
    text = extract_document_text(
        "paper.pdf",
        "application/pdf",
        b"data",
        pdf_reader_factory=lambda _: SimpleNamespace(
            pages=[
                SimpleNamespace(extract_text=lambda: " A\r\n\r\n\r\nB "),
                SimpleNamespace(extract_text=lambda: " C "),
            ]
        ),
    )

    assert text == "A\n\nB\n\nC"
