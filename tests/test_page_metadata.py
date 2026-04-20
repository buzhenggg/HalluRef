"""站点专属页面元数据解析测试"""

import pytest

from src.retrievers.page_metadata import (
    fetch_page_metadata,
    normalize_landing_page_url,
    parse_page_metadata,
    parse_pdf_metadata,
)


def test_parse_arxiv_meta_tags():
    html = """
    <html><head>
      <meta name="citation_title" content="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding">
      <meta name="citation_author" content="Jacob Devlin">
      <meta name="citation_author" content="Ming-Wei Chang">
      <meta name="citation_author" content="Kenton Lee">
      <meta name="citation_author" content="Kristina Toutanova">
      <meta name="citation_date" content="2018/10/11">
      <meta name="citation_abstract" content="We introduce a new language representation model called BERT.">
      <meta name="citation_pdf_url" content="https://arxiv.org/pdf/1810.04805.pdf">
    </head><body></body></html>
    """
    paper = parse_page_metadata("https://arxiv.org/abs/1810.04805", html)

    assert paper is not None
    assert paper.source == "page_parser"
    assert paper.title.startswith("BERT:")
    assert paper.authors[0] == "Jacob Devlin"
    assert paper.year == 2018
    assert "language representation" in (paper.abstract or "").lower()


def test_parse_acl_anthology_meta_tags():
    html = """
    <html><head>
      <meta name="citation_title" content="Attention Is All You Need">
      <meta name="citation_author" content="Ashish Vaswani">
      <meta name="citation_author" content="Noam Shazeer">
      <meta name="citation_publication_date" content="2017">
      <meta name="citation_conference_title" content="Advances in Neural Information Processing Systems">
      <meta name="citation_doi" content="10.5555/3295222.3295349">
      <meta name="description" content="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.">
    </head><body></body></html>
    """
    paper = parse_page_metadata("https://aclanthology.org/P17-1011/", html)

    assert paper is not None
    assert paper.title == "Attention Is All You Need"
    assert paper.authors[:2] == ["Ashish Vaswani", "Noam Shazeer"]
    assert paper.year == 2017
    assert paper.venue == "Advances in Neural Information Processing Systems"
    assert paper.doi == "10.5555/3295222.3295349"


def test_parse_unknown_site_without_metadata_returns_none():
    html = "<html><head><title>Example</title></head><body>No metadata</body></html>"
    paper = parse_page_metadata("https://example.com/paper", html)
    assert paper is None


def test_normalize_acl_pdf_url_to_landing_page():
    assert normalize_landing_page_url(
        "https://aclanthology.org/2025.acl-long.17.pdf"
    ) == "https://aclanthology.org/2025.acl-long.17/"


def test_normalize_arxiv_pdf_url_to_abs_page():
    assert normalize_landing_page_url(
        "https://arxiv.org/pdf/1810.04805.pdf"
    ) == "https://arxiv.org/abs/1810.04805"


def test_parse_semantic_scholar_page_assigns_default_venue():
    html = """
    <html><head>
      <meta name="citation_title" content="Chain-of-Thought Prompting Elicits Reasoning in Large Language Models">
      <meta name="citation_author" content="Jason Wei">
      <meta name="citation_publication_date" content="2022/01/01">
    </head><body></body></html>
    """
    paper = parse_page_metadata("https://www.semanticscholar.org/paper/abc", html)

    assert paper is not None
    assert paper.venue == "Semantic Scholar"
    assert paper.year == 2022


def test_parse_arxiv_dom_without_meta_tags():
    html = """
    <html><body>
      <h1 class="title mathjax">Title: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding</h1>
      <div class="authors">Authors: <a>Jacob Devlin</a>, <a>Ming-Wei Chang</a></div>
      <blockquote class="abstract mathjax">
        Abstract: We introduce a new language representation model called BERT.
      </blockquote>
      <div class="dateline">(Submitted on 11 Oct 2018)</div>
    </body></html>
    """
    paper = parse_page_metadata("https://arxiv.org/abs/1810.04805", html)

    assert paper is not None
    assert paper.title.startswith("BERT:")
    assert paper.authors == ["Jacob Devlin", "Ming-Wei Chang"]
    assert paper.year == 2018
    assert paper.venue == "arXiv"


def test_parse_openreview_script_payload_without_meta_tags():
    html = """
    <html><body>
      <script id="__NEXT_DATA__" type="application/json">
      {
        "props": {
          "pageProps": {
            "forumNote": {
              "content": {
                "title": "OpenReview Paper Title",
                "authors": ["Alice Smith", "Bob Jones"],
                "abstract": "This is an OpenReview abstract."
              },
              "cdate": 1711929600000
            }
          }
        }
      }
      </script>
    </body></html>
    """
    paper = parse_page_metadata("https://openreview.net/forum?id=abc123", html)

    assert paper is not None
    assert paper.title == "OpenReview Paper Title"
    assert paper.authors == ["Alice Smith", "Bob Jones"]
    assert paper.year == 2024
    assert paper.venue == "OpenReview"


def test_parse_ieee_embedded_json_without_meta_tags():
    html = """
    <html><body>
      <script type="application/json" id="xplGlobalDocumentMetadata">
      {
        "title": "IEEE Paper",
        "authors": [{"name": "Jane Doe"}, {"name": "John Roe"}],
        "publicationYear": "2023",
        "publicationTitle": "IEEE Transactions on Pattern Analysis and Machine Intelligence",
        "doi": "10.1109/5.771073",
        "abstract": "An IEEE abstract."
      }
      </script>
    </body></html>
    """
    paper = parse_page_metadata("https://ieeexplore.ieee.org/document/123456", html)

    assert paper is not None
    assert paper.title == "IEEE Paper"
    assert paper.authors == ["Jane Doe", "John Roe"]
    assert paper.year == 2023
    assert paper.doi == "10.1109/5.771073"
    assert paper.venue == "IEEE Transactions on Pattern Analysis and Machine Intelligence"


def test_parse_acm_dom_without_meta_tags():
    html = """
    <html><body>
      <h1 property="name">ACM Paper</h1>
      <div class="author-data">
        <span property="author">Alice Smith</span>
        <span property="author">Bob Jones</span>
      </div>
      <section id="abstract">
        <div role="paragraph">An ACM abstract.</div>
      </section>
      <span class="core-date-published">2021</span>
      <div class="issue-item__detail"><a>Proceedings of the ACM on Management of Data</a></div>
    </body></html>
    """
    paper = parse_page_metadata("https://dl.acm.org/doi/10.1145/1234567", html)

    assert paper is not None
    assert paper.title == "ACM Paper"
    assert paper.authors == ["Alice Smith", "Bob Jones"]
    assert paper.year == 2021
    assert paper.abstract == "An ACM abstract."
    assert paper.venue == "Proceedings of the ACM on Management of Data"


def test_parse_pubmed_page_assigns_default_venue():
    html = """
    <html><head>
      <meta name="citation_title" content="A PubMed Indexed Paper">
      <meta name="citation_author" content="Alice Smith">
      <meta name="citation_publication_date" content="2021/06/01">
    </head><body></body></html>
    """
    paper = parse_page_metadata("https://pubmed.ncbi.nlm.nih.gov/12345678/", html)

    assert paper is not None
    assert paper.venue == "PubMed"


def test_parse_pdf_metadata_extracts_title_abstract_and_year(monkeypatch):
    class FakePage:
        def extract_text(self):
            return (
                "A Strong Paper Title\n"
                "Alice Smith, Bob Jones\n\n"
                "Abstract\n"
                "This paper proposes a practical PDF parsing fallback.\n"
                "Proceedings of ACL 2025\n"
            )

    class FakePdfReader:
        def __init__(self, _stream):
            self.pages = [FakePage()]

    monkeypatch.setattr("src.retrievers.page_metadata.PdfReader", FakePdfReader)

    paper = parse_pdf_metadata(
        "https://example.org/paper.pdf",
        b"%PDF-1.4 fake bytes",
    )

    assert paper is not None
    assert paper.title == "A Strong Paper Title"
    assert paper.authors == ["Alice Smith", "Bob Jones"]
    assert paper.year == 2025
    assert "practical pdf parsing fallback" in (paper.abstract or "").lower()


@pytest.mark.asyncio
async def test_fetch_page_metadata_passes_proxy_to_httpx_client(monkeypatch):
    captured = {}

    class FakeResponse:
        headers = {"content-type": "text/html"}
        text = """
        <html><head>
          <meta name="citation_title" content="Proxy Paper">
          <meta name="citation_author" content="Alice Smith">
          <meta name="citation_publication_date" content="2024">
        </head></html>
        """
        content = text.encode("utf-8")
        url = "https://example.org/paper"

        def raise_for_status(self):
            return None

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url):
            captured["url"] = url
            return FakeResponse()

    monkeypatch.setattr("src.retrievers.page_metadata.httpx.AsyncClient", FakeAsyncClient)

    paper = await fetch_page_metadata(
        "https://example.org/paper",
        proxy="http://127.0.0.1:7890",
    )

    assert paper is not None
    assert paper.title == "Proxy Paper"
    assert captured["proxy"] == "http://127.0.0.1:7890"
