"""工具模块单元测试"""

import pytest

from src.utils.text_similarity import normalize_text, title_similarity, venue_similarity
from src.utils.name_matcher import (
    normalize_author_name,
    author_name_similarity,
    author_list_similarity,
)


# ── text_similarity ───────────────────────────────────────

class TestTitleSimilarity:
    def test_identical(self):
        assert title_similarity("Attention Is All You Need", "Attention Is All You Need") > 0.99

    def test_case_insensitive(self):
        assert title_similarity("attention is all you need", "ATTENTION IS ALL YOU NEED") > 0.99

    def test_similar(self):
        score = title_similarity(
            "A Novel Framework for Text Generation",
            "A Novel Framework for Text Generation Using Transformers",
        )
        assert score > 0.7

    def test_different(self):
        score = title_similarity(
            "Attention Is All You Need",
            "ImageNet Classification with Deep Convolutional Neural Networks",
        )
        assert score < 0.5

    def test_empty(self):
        assert title_similarity("", "something") == 0.0


class TestVenueSimilarity:
    def test_abbreviation(self):
        score = venue_similarity("NeurIPS", "Neural Information Processing Systems")
        assert score > 0.9

    def test_same(self):
        assert venue_similarity("ICML", "ICML") > 0.99


# ── name_matcher ──────────────────────────────────────────

class TestAuthorNameSimilarity:
    def test_same_name(self):
        assert author_name_similarity("John Smith", "John Smith") > 0.99

    def test_last_first_format(self):
        score = author_name_similarity("Smith, John", "John Smith")
        assert score > 0.7

    def test_initials(self):
        score = author_name_similarity("J. Smith", "John Smith")
        assert score > 0.6

    def test_initial_plus_surname_matches_full_given_name(self):
        assert author_name_similarity("Z. Guo", "Zhijiang Guo") >= 0.9
        assert author_name_similarity("M. Schlichtkrull", "Michael Schlichtkrull") >= 0.9
        assert author_name_similarity("A. Vlachos", "Andreas Vlachos") >= 0.9

    def test_inverted_initial_name_matches_full_name(self):
        assert author_name_similarity("Guo, Z.", "Zhijiang Guo") >= 0.9

    def test_multiple_initials_and_suffix_match_without_suffix_penalty(self):
        assert author_name_similarity("E. C. M. Boyle, III", "E C M Boyle") >= 0.9

    def test_same_surname_different_initial_is_not_high_confidence(self):
        assert author_name_similarity("Z. Guo", "Michael Guo") < 0.8

    def test_different_surname_is_low_confidence(self):
        assert author_name_similarity("Z. Guo", "Zhijiang Wang") < 0.4


class TestAuthorListSimilarity:
    def test_identical_list(self):
        score, _ = author_list_similarity(
            ["John Smith", "Jane Doe"],
            ["John Smith", "Jane Doe"],
        )
        assert score > 0.9

    def test_partial_overlap(self):
        score, matches = author_list_similarity(
            ["John Smith", "Jane Doe", "Bob Brown"],
            ["John Smith", "Jane Doe"],
        )
        assert 0.3 < score < 0.9

    def test_empty(self):
        score, _ = author_list_similarity([], [])
        assert score == 1.0

    def test_initial_author_list_matches_full_names(self):
        score, matches = author_list_similarity(
            ["Z. Guo", "M. Schlichtkrull", "A. Vlachos"],
            ["Zhijiang Guo", "Michael Schlichtkrull", "Andreas Vlachos"],
        )
        assert score >= 0.8
        assert matches == [
            ("Z. Guo", "Zhijiang Guo"),
            ("M. Schlichtkrull", "Michael Schlichtkrull"),
            ("A. Vlachos", "Andreas Vlachos"),
        ]

    def test_author_list_matches_out_of_order(self):
        score, matches = author_list_similarity(
            ["M. Schlichtkrull", "A. Vlachos", "Z. Guo"],
            ["Zhijiang Guo", "Michael Schlichtkrull", "Andreas Vlachos"],
        )

        assert score >= 0.8
        assert matches == [
            ("M. Schlichtkrull", "Michael Schlichtkrull"),
            ("A. Vlachos", "Andreas Vlachos"),
            ("Z. Guo", "Zhijiang Guo"),
        ]

    def test_et_al_first_author_is_high_confidence_match(self):
        score, matches = author_list_similarity(
            ["Vaswani et al."],
            [
                "Ashish Vaswani",
                "Noam Shazeer",
                "Niki Parmar",
                "Jakob Uszkoreit",
                "Llion Jones",
                "Aidan N. Gomez",
                "Lukasz Kaiser",
                "Illia Polosukhin",
            ],
        )

        assert score >= 0.9
        assert matches == [("Vaswani et al.", "Ashish Vaswani")]

    def test_obvious_author_count_mismatch_is_not_full_match(self):
        score, _ = author_list_similarity(
            ["Z. Guo", "M. Schlichtkrull", "A. Vlachos"],
            ["Zhijiang Guo", "Michael Schlichtkrull"],
        )
        assert score < 0.8
