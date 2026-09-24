"""
These tests exercise the extraction *algorithm* (column banding, card
boundaries, headline/label pairing) against small hand-built fake pages
rather than real rendered PDFs. pdfplumber's own word/line extraction is an
external, already-tested library; what this session's bugs lived in was the
pairing/leftover logic layered on top of it, so that's what these tests
target directly.

FakePage/FakeRegion implement just the pdfplumber surface pdf_table_extraction
actually calls: .chars, .lines, .width, .height, .crop(), .filter(),
.extract_words(), .extract_text(). Each "char" dict here represents one
already-formed word token (text/x0/x1/top/bottom/size) -- word-level, not
glyph-level -- since word grouping itself isn't the logic under test.
"""
import pdfplumber

from src.ingestion.pdf_table_extraction import (
    _body_size,
    _extract_card_text,
    _find_card_boundaries,
    _find_column_bands,
    extract_structured_text,
)


def word(text, x0, top, x1=None, bottom=None, size=10.0):
    return {
        "text": text,
        "x0": x0,
        "top": top,
        "x1": x1 if x1 is not None else x0 + len(text) * size * 0.5,
        "bottom": bottom if bottom is not None else top + size,
        "size": size,
    }


class FakeRegion:
    def __init__(self, chars, lines=None):
        self.chars = chars
        self.lines = lines or []

    def filter(self, predicate):
        return FakeRegion([c for c in self.chars if predicate(c)], self.lines)

    def extract_words(self):
        return [dict(c) for c in self.chars]

    def extract_text(self):
        ordered = sorted(self.chars, key=lambda c: (round(c["top"]), c["x0"]))
        return " ".join(c["text"] for c in ordered)


class FakePage(FakeRegion):
    def __init__(self, chars, lines=None, width=600, height=800):
        super().__init__(chars, lines)
        self.width = width
        self.height = height

    def crop(self, bbox):
        x0, top, x1, bottom = bbox
        cropped = [
            c for c in self.chars
            if c["x0"] >= x0 - 0.01 and c["x1"] <= x1 + 0.01
            and c["top"] >= top - 0.01 and c["bottom"] <= bottom + 0.01
        ]
        return FakeRegion(cropped, self.lines)


# ---------------------------------------------------------------------------
# _body_size
# ---------------------------------------------------------------------------

def test_body_size_returns_most_common_size():
    page = FakePage([word("a", 0, 0, size=9.0), word("b", 0, 10, size=9.0), word("c", 0, 20, size=24.0)])
    assert _body_size(page) == 9.0


def test_body_size_defaults_when_no_chars():
    page = FakePage([])
    assert _body_size(page) == 9.0


# ---------------------------------------------------------------------------
# _find_column_bands
# ---------------------------------------------------------------------------

def test_find_column_bands_single_column_when_no_wide_gap():
    words = [word("a", 10, 0), word("b", 60, 0), word("c", 110, 0)]
    page = FakePage(words, width=600)
    assert _find_column_bands(page) == [(0, page.width)]


def test_find_column_bands_splits_on_wide_middle_gap():
    # The gap must straddle the page's horizontal middle band (0.3-0.7 of
    # width) to count -- a wide gap out near either edge does not.
    left_words = [word("left", 10, 0, x1=280)]
    right_words = [word("right", 320, 0, x1=590)]
    page = FakePage(left_words + right_words, width=600)

    bands = _find_column_bands(page)

    assert len(bands) == 2
    assert bands[0][0] == 0
    assert bands[1][1] == 600
    assert bands[0][1] < bands[1][0]


def test_find_column_bands_empty_page_is_single_band():
    page = FakePage([], width=600)
    assert _find_column_bands(page) == [(0, 600)]


# ---------------------------------------------------------------------------
# _find_card_boundaries
# ---------------------------------------------------------------------------

def test_find_card_boundaries_no_rules_is_one_segment():
    page = FakePage([], lines=[], width=600, height=800)
    assert _find_card_boundaries(page, (0, 600)) == [(0.0, 800)]


def test_find_card_boundaries_splits_on_horizontal_rules_within_band():
    rule = {"height": 0.0, "x0": 0, "x1": 590, "top": 400}
    page = FakePage([], lines=[rule], width=600, height=800)

    boundaries = _find_card_boundaries(page, (0, 600))

    assert boundaries == [(0.0, 400), (400, 800)]


def test_find_card_boundaries_ignores_rules_outside_band_or_too_short():
    short_rule = {"height": 0.0, "x0": 0, "x1": 50, "top": 400}  # spans < 30% of band width
    outside_rule = {"height": 0.0, "x0": 700, "x1": 900, "top": 200}  # outside band bbox
    page = FakePage([], lines=[short_rule, outside_rule], width=600, height=800)

    boundaries = _find_card_boundaries(page, (0, 600))

    assert boundaries == [(0.0, 800)]


# ---------------------------------------------------------------------------
# _extract_card_text -- the two bugs found/fixed this session
# ---------------------------------------------------------------------------

def test_extract_card_text_pairs_full_multiword_label_not_just_nearest_word():
    # Regression test: the original bug grabbed only the single nearest label
    # word ("NPAT2: $5,412m") instead of the full label phrase
    # ("Statutory NPAT2: $5,412m").
    headline = word("$5,412m", 100, 50, x1=160, bottom=68, size=20.0)
    label_words = [
        word("Statutory", 100, 70, x1=140, bottom=80, size=9.0),
        word("NPAT2", 145, 70, x1=175, bottom=80, size=9.0),
    ]
    page = FakePage([headline] + label_words, width=600, height=800)

    table_text, leftover_text = _extract_card_text(page, (0, 600), (0, 800), body_size=9.0)

    assert table_text == "Statutory NPAT2: $5,412m"
    assert leftover_text is None


def test_extract_card_text_preserves_unconsumed_text_as_leftover():
    # Regression test for the data-loss bug: a region with one successful
    # pairing must not silently drop unrelated body text (e.g. a footnote)
    # sharing the same region.
    headline = word("$5,412m", 100, 50, x1=160, bottom=68, size=20.0)
    label = word("NPAT", 100, 70, x1=135, bottom=80, size=9.0)
    footnote = word("Footnote:", 300, 700, x1=350, bottom=710, size=9.0)
    footnote2 = word("see", 355, 700, x1=380, bottom=710, size=9.0)
    page = FakePage([headline, label, footnote, footnote2], width=600, height=800)

    table_text, leftover_text = _extract_card_text(page, (0, 600), (0, 800), body_size=9.0)

    assert table_text == "NPAT: $5,412m"
    assert leftover_text is not None
    assert "Footnote:" in leftover_text
    assert "see" in leftover_text


def test_extract_card_text_returns_none_when_no_headline_words():
    page = FakePage([word("just body text", 0, 0, size=9.0)], width=600, height=800)
    table_text, leftover_text = _extract_card_text(page, (0, 600), (0, 800), body_size=9.0)
    assert table_text is None
    assert leftover_text is None


def test_extract_card_text_falls_back_to_narrative_when_headline_has_no_nearby_label():
    # A large token with no label anywhere near it (e.g. a section masthead)
    # must not force the whole region into "table" treatment.
    masthead = word("Section Title", 0, 0, x1=200, bottom=20, size=20.0)
    unrelated_text = word("Unrelated body paragraph.", 0, 500, x1=250, bottom=515, size=9.0)
    page = FakePage([masthead, unrelated_text], width=600, height=800)

    table_text, leftover_text = _extract_card_text(page, (0, 600), (0, 800), body_size=9.0)

    assert table_text is None
    assert leftover_text is None


def test_extract_card_text_headline_above_upper_ratio_bound_is_excluded():
    # A ratio-bounded check: a token far larger than any real stat-card number
    # (e.g. a 38pt page masthead against a 9pt body) must not be treated as
    # a headline number at all.
    oversized_masthead = word("MASTHEAD", 0, 0, x1=200, bottom=38, size=38.0)  # ratio ~4.2x body
    label = word("some label", 0, 40, x1=100, bottom=50, size=9.0)
    page = FakePage([oversized_masthead, label], width=600, height=800)

    table_text, leftover_text = _extract_card_text(page, (0, 600), (0, 800), body_size=9.0)

    assert table_text is None
    assert leftover_text is None


# ---------------------------------------------------------------------------
# extract_structured_text -- end-to-end orchestration, pdfplumber.open mocked
# ---------------------------------------------------------------------------

def test_extract_structured_text_assembles_narrative_and_tables_across_pages(monkeypatch, tmp_path):
    headline = word("$5,412m", 100, 50, x1=160, bottom=68, size=20.0)
    label = word("NPAT", 100, 70, x1=135, bottom=80, size=9.0)
    # Extra body-sized filler (far from the headline, so it doesn't get
    # pulled into the pairing) so body-size detection has more than one
    # 9.0-size sample to outweigh the single 20.0-size headline on a tie.
    filler1 = word("filler", 100, 750, x1=140, bottom=760, size=9.0)
    filler2 = word("more filler", 100, 765, x1=170, bottom=775, size=9.0)
    table_page = FakePage([headline, label, filler1, filler2], width=600, height=800)

    narrative_words = [word("Plain narrative sentence.", 0, 0, x1=250, bottom=15, size=9.0)]
    narrative_page = FakePage(narrative_words, width=600, height=800)

    class FakePdf:
        pages = [table_page, narrative_page]
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    monkeypatch.setattr(pdfplumber, "open", lambda path: FakePdf())

    fake_pdf_path = tmp_path / "fake.pdf"
    fake_pdf_path.write_bytes(b"%PDF-1.4 fake")

    result = extract_structured_text(str(fake_pdf_path))

    assert result["table_blocks"] == ["NPAT: $5,412m"]
    assert "Plain narrative sentence." in result["narrative_text"]
