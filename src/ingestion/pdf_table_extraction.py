import pdfplumber
from collections import Counter

# All thresholds below are structural (page-relative), not tuned to any
# specific document's coordinates or content:
#   - HEADLINE_SIZE_RATIO range: a token counts as a "headline number" if its
#     font size falls within this multiple of the page's most common (body)
#     size — bounded above as well as below, so a one-off page masthead/title
#     (much larger than any repeated stat-card number) doesn't get swept in.
#   - COLUMN_GUTTER_MIN_WIDTH: minimum horizontal gap between words, near the
#     page's horizontal middle, to treat as a two-column boundary.
#   - LABEL_MAX_GAP_RATIO: a label only pairs with a headline number if it
#     sits within this multiple of the headline's own font size beneath it —
#     keeps pairing local to one stat card instead of reaching across
#     unrelated text.
HEADLINE_SIZE_RATIO_MIN = 1.4
HEADLINE_SIZE_RATIO_MAX = 2.5
COLUMN_GUTTER_MIN_WIDTH = 20
LABEL_MAX_GAP_RATIO = 1.5
LABEL_LINE_TOLERANCE = 3  # points of `top` difference still counted as "the same line"


def _body_size(page) -> float:
    sizes = Counter(round(c["size"], 1) for c in page.chars)
    return sizes.most_common(1)[0][0] if sizes else 9.0


def _find_column_bands(page) -> list:
    """Splits the page into one or two horizontal x-ranges ('column bands'),
    based on the widest gap between word x-positions near the page's
    horizontal middle. No gap of sufficient width -> single-column page."""
    words = page.extract_words()
    if not words:
        return [(0, page.width)]

    xs = sorted(set([round(w["x0"]) for w in words] + [round(w["x1"]) for w in words]))
    mid_lo, mid_hi = page.width * 0.3, page.width * 0.7

    best_gap = 0
    gap_at = None
    for a, b in zip(xs, xs[1:]):
        if mid_lo <= a <= mid_hi and (b - a) > best_gap:
            best_gap = b - a
            gap_at = (a, b)

    if gap_at and best_gap >= COLUMN_GUTTER_MIN_WIDTH:
        return [(0, gap_at[0]), (gap_at[1], page.width)]
    return [(0, page.width)]


def _find_card_boundaries(page, band) -> list:
    """Horizontal rule lines within this column band split it into vertical
    segments ('cards'). No rules found -> the whole band is one segment."""
    x0, x1 = band
    band_width = x1 - x0

    rules = [
        line for line in page.lines
        if line["height"] < 0.5  # horizontal line
        and line["x0"] >= x0 - 5 and line["x1"] <= x1 + 5
        and (line["x1"] - line["x0"]) > band_width * 0.3  # spans a meaningful part of the band
    ]

    ys = sorted(set([0.0, page.height] + [r["top"] for r in rules]))
    return list(zip(ys, ys[1:]))


def _extract_card_text(page, band, y_range, body_size: float):
    """Pairs headline-size numbers with the label text sitting below them,
    within one card region. Returns (table_text, leftover_text):
    - table_text is None if no headline-size token found a real label nearby
      anywhere in the region (not a label:value card at all).
    - leftover_text is whatever label-sized text in the region was NOT
      consumed by a pairing -- real narrative content (an explanatory
      paragraph, an unrelated footnote list sharing the region with one
      incidental larger-font snippet) must never be silently dropped just
      because *some* pairing succeeded elsewhere in the same region."""
    x0, x1 = band
    top, bottom = y_range
    if bottom - top < 1:
        return None, None

    region = page.crop((x0, top, x1, bottom))
    headline_min = body_size * HEADLINE_SIZE_RATIO_MIN
    headline_max = body_size * HEADLINE_SIZE_RATIO_MAX

    headline_region = region.filter(lambda obj: headline_min <= obj.get("size", 0) <= headline_max)
    headline_words = headline_region.extract_words() if headline_region.chars else []
    if not headline_words:
        return None, None

    label_region = region.filter(lambda obj: obj.get("size", 0) < headline_min)
    label_words = label_region.extract_words() if label_region.chars else []

    def word_key(w):
        return (w["text"], round(w["x0"]), round(w["top"]))

    lines = []
    any_paired = False
    consumed = set()
    for hw in headline_words:
        max_gap = hw["bottom"] - hw["top"]  # headline token's own font height
        nearest = [
            lw for lw in label_words
            if hw["bottom"] < lw["top"] <= hw["bottom"] + max_gap * LABEL_MAX_GAP_RATIO
            and not (lw["x1"] < hw["x0"] or lw["x0"] > hw["x1"])  # horizontal overlap
        ]
        if nearest:
            closest_top = min(lw["top"] for lw in nearest)
            same_line = sorted(
                (lw for lw in nearest if abs(lw["top"] - closest_top) <= LABEL_LINE_TOLERANCE),
                key=lambda lw: lw["x0"],
            )
            label_text = " ".join(lw["text"] for lw in same_line)
            lines.append(f"{label_text}: {hw['text']}")
            any_paired = True
            consumed.update(word_key(lw) for lw in same_line)
        else:
            lines.append(hw["text"])

    if not any_paired:
        # No headline-size token found a real label anywhere nearby -- this
        # region isn't a label:value stat card (more likely a section heading
        # or masthead sharing space with body text). Treat the whole region
        # as narrative instead.
        return None, None

    leftover_words = [lw for lw in label_words if word_key(lw) not in consumed]
    leftover_text = None
    if leftover_words:
        leftover_sorted = sorted(leftover_words, key=lambda w: (round(w["top"]), w["x0"]))
        leftover_text = " ".join(w["text"] for w in leftover_sorted)

    return "\n".join(lines), leftover_text


def extract_structured_text(file_path: str) -> dict:
    """Splits each page into column bands, each band into rule-separated
    cards, and each card into either a headline-number/label block (a
    'table_block') or, if it has no headline-size tokens, plain narrative
    text. Returns {"narrative_text": str, "table_blocks": list[str]}."""
    narrative_parts = []
    table_blocks = []

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            body_size = _body_size(page)
            bands = _find_column_bands(page)
            page_narrative = []

            for band in bands:
                for y_range in _find_card_boundaries(page, band):
                    card_text, leftover_text = _extract_card_text(page, band, y_range, body_size)
                    if card_text:
                        table_blocks.append(card_text)
                        if leftover_text and leftover_text.strip():
                            page_narrative.append(leftover_text)
                    else:
                        region = page.crop((band[0], y_range[0], band[1], y_range[1]))
                        text = region.extract_text() or ""
                        if text.strip():
                            page_narrative.append(text)

            narrative_parts.append("\n".join(page_narrative))

    return {"narrative_text": "\n".join(narrative_parts), "table_blocks": table_blocks}
