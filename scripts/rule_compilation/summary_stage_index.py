"""Chapter index for the Summary Stage 2018 manual.

Pure code, no LLM. Summary Stage is a single-item manual (NAACCR 764) whose
structure is unusually regular: a run of site-agnostic general chapters followed
by ~90 site chapters, each of which opens with a code preamble naming the
ICD-O-3 topography and morphology ranges the chapter covers, e.g.

    ## GUM

    8000-8700, 8982 C030-C031, C039, C062 C030 Upper gum C031 Lower gum ...

    **Note 1: Sources used in the development of this chapter**

This module turns that structure into a list of ``Chapter`` records carrying the
line range to compile and the ``RuleApplicability`` defaults to compile it with,
so the driver never has to guess a per-chapter ``--sites``/``--histologies``.
Codes are read only from the preamble (heading up to the first "Note N:" line):
the chapter bodies enumerate distant sites and metastatic nodes, and scooping
those into ``applies_to`` would make every chapter match every case.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from cipoc.models import RuleApplicability

from .segment import Section, segment_markdown

# The NAACCR item this manual governs, in full. Every unit compiled from it is
# forced to this item rather than trusting the model to infer it per section.
SUMMARY_STAGE_ITEM_ID = 764

# The manual is effective for cases diagnosed 1/1/2018 and forward, and each new
# version supersedes the last for that whole range. This is the units' own
# dx_date_min, which is what keeps them applicable to a 2019 case even though
# the manifest dates the edition to its 2025 publication (see MANIFEST_ENTRY).
EFFECTIVE_DX_DATE_MIN = "2018-01-01"

MANIFEST_ENTRY = {
    "title": "SEER Summary Stage 2018 (v3.3)",
    "family": "SEER",
    # Edition recency, NOT the effective date. resolve_precedence() ranks same-family
    # manuals by publication_date, so dating this to the 2018 effective date would let
    # the SPCSM 2024 summary of Summary Stage outrank the dedicated manual and silently
    # drop every unit compiled here. The effective date rides on each unit's
    # applies_to.dx_date_min instead, which scope_coding_context() honours ahead of the
    # manual-level temporal filter.
    "publication_date": "2025-11-01",
    "effective_note": (
        "Published November 2025; effective for all cases diagnosed 2018-01-01 and "
        "forward. publication_date records edition recency for precedence; unit-level "
        "applies_to.dx_date_min carries the 2018 effective date."
    ),
}

# Region boundaries in the manual, matched as case-insensitive exact headings.
GENERAL_REGION_START = "summary stage"
SITE_REGION_START = "head and neck"
APPENDIX_PREFIX = "appendix"

# Front matter that carries no coding guidance.
_SKIP_HEADINGS = ("table of contents", "summary stage 2018 general coding instructions")

# A chapter preamble is the code block between the heading and the chapter's
# first "Note N:" line. That Note line is the only reliable terminator: every
# real site chapter opens its notes with "Note 1: Sources used in the development
# of this chapter", while the system-level guidance sections (Distinguishing "In
# Situ" and "Localized" Tumors, Bladder Anatomic Structures, ...) have no code
# header and no Note line at all. Requiring the terminator is therefore also how
# a guidance section is recognised: no Note line means no code header, so the
# chapter stays unscoped instead of picking up whatever codes its prose tables
# happen to mention.
_NOTE_LINE = re.compile(r"^\W*note\s*\d*\s*:", re.IGNORECASE)
# Generous enough for the longest genuine header (Brain lists two full behavior-split
# histology blocks before its Note 1), tight enough that a section without a header
# is not scanned to its end.
_MAX_PREAMBLE_LINES = 40

# Exclusion clauses name codes the chapter does NOT cover -- "C009 Lip, NOS
# (excludes skin of lip C440)", "8720-8790 [except C500]". Blanked before parsing
# so the excluded code is not read as part of the chapter's scope.
_EXCLUSION = re.compile(
    r"[(\[]\s*(?:exclud\w*|except\w*)\b[^)\]]*[)\]]",
    re.IGNORECASE,
)

_SITE_RANGE = re.compile(r"C(\d{3})\s*-\s*C?(\d{3})")
_SITE_CODE = re.compile(r"C(\d{3})")
_MORPH_RANGE = re.compile(r"(?<!\d)(\d{4})\s*-\s*(\d{4})(?!\d)")
_MORPH_CODE = re.compile(r"(?<!\d)(\d{4})(?!\d)")

# ICD-O-3 morphology occupies 8000-9993. Bounding the scan to that window is what
# keeps year spans -- "2018-2024", "2025+", "(2023+)" -- out of the histology list.
_MORPH_MIN, _MORPH_MAX = 8000, 9999
# ICD-O-3 topography runs C000-C809; C76x-C80x are ill-defined/unknown sites and
# are legitimate chapter scopes (Ill-Defined Other), so the whole span is allowed.
_SITE_MIN, _SITE_MAX = 0, 809


@dataclass(frozen=True)
class Chapter:
    """One compilable chapter of the Summary Stage manual."""

    heading: str
    site_group: str
    scope: str  # "general" (site-agnostic) or "site"
    system: str | None  # parent body-system heading, when the chapter sits under one
    start_line: int  # 0-indexed, inclusive
    end_line: int
    sites: tuple[str, ...]
    histologies: tuple[str, ...]

    @property
    def applicability(self) -> RuleApplicability:
        """Default applies_to for every unit compiled from this chapter.

        Empty site/histology lists become None rather than ``[]`` so scoping
        widens: a chapter we could not scope must not exclude itself from every
        case.
        """
        return RuleApplicability(
            sites=list(self.sites) or None,
            histologies=list(self.histologies) or None,
            dx_date_min=EFFECTIVE_DX_DATE_MIN,
        )


def slugify_chapter(heading: str) -> str:
    """Chapter heading -> output file stem, e.g. 'LARYNX SUPRAGLOTTIC' -> 'larynx_supraglottic'."""
    return re.sub(r"[^a-z0-9]+", "_", heading.casefold()).strip("_")


def _merge_ranges(values: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Collapse overlapping/adjacent numeric spans into a minimal sorted list."""
    merged: list[tuple[int, int]] = []
    for low, high in sorted(values):
        if merged and low <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], high))
        else:
            merged.append((low, high))
    return merged


def _spans(text: str, range_re: re.Pattern, code_re: re.Pattern, low: int, high: int) -> list[tuple[int, int]]:
    """Numeric spans in ``text``, reading explicit ranges first then bare codes.

    Ranges are blanked out before the bare-code scan so 'C030-C031' does not also
    register C030 and C031 as separate codes -- harmless for matching, but it
    would leave the merged output noisier than the source.
    """
    spans: list[tuple[int, int]] = []
    for match in range_re.finditer(text):
        start, end = int(match.group(1)), int(match.group(2))
        if low <= start <= high and low <= end <= high and start <= end:
            spans.append((start, end))
    remainder = range_re.sub(" ", text)
    for match in code_re.finditer(remainder):
        code = int(match.group(1))
        if low <= code <= high:
            spans.append((code, code))
    return spans


def parse_sites(preamble: str) -> tuple[str, ...]:
    """ICD-O-3 topography ranges named in a chapter preamble, merged and sorted."""
    merged = _merge_ranges(_spans(preamble, _SITE_RANGE, _SITE_CODE, _SITE_MIN, _SITE_MAX))
    return tuple(
        f"C{low:03d}" if low == high else f"C{low:03d}-C{high:03d}" for low, high in merged
    )


def parse_histologies(preamble: str) -> tuple[str, ...]:
    """ICD-O-3 morphology ranges named in a chapter preamble, merged and sorted."""
    merged = _merge_ranges(_spans(preamble, _MORPH_RANGE, _MORPH_CODE, _MORPH_MIN, _MORPH_MAX))
    return tuple(str(low) if low == high else f"{low}-{high}" for low, high in merged)


def chapter_preamble(body: str) -> str | None:
    """The code block at the head of a chapter, or None if it has no code header.

    Returns the lines between the heading and the chapter's first "Note N:" line.
    None means no terminator was found within ``_MAX_PREAMBLE_LINES``, which for
    this manual identifies a guidance section rather than a site chapter.
    """
    kept: list[str] = []
    for line in body.splitlines():
        if not line.strip():
            continue
        if _NOTE_LINE.match(line):
            return _EXCLUSION.sub(" ", "\n".join(kept))
        kept.append(line)
        if len(kept) >= _MAX_PREAMBLE_LINES:
            return None
    return None


def _is_stub(section: Section) -> bool:
    """True for a body-system divider heading that carries no content of its own."""
    return not section.body.strip()


def build_index(
    source_path: str | Path,
    *,
    include_appendices: bool = False,
) -> list[Chapter]:
    """Index every compilable chapter of the Summary Stage manual, in document order.

    Chapters before the site region are marked ``scope="general"`` and left
    unscoped by site/histology -- they are the stage-code definitions and general
    coding instructions that govern every case. Chapters in the site region are
    scoped from their own code preamble. Body-system dividers ('# HEAD AND NECK')
    carry no body and are recorded only as the ``system`` of the chapters beneath
    them. Appendices are reference tables rather than coding guidance and are
    excluded unless ``include_appendices`` is set.
    """
    sections = segment_markdown(Path(source_path).read_text(), max_heading_level=2)

    def index_of(predicate) -> int | None:
        return next((i for i, s in enumerate(sections) if predicate(s)), None)

    general_start = index_of(lambda s: s.heading.casefold() == GENERAL_REGION_START)
    site_start = index_of(lambda s: s.heading.casefold() == SITE_REGION_START)
    if general_start is None or site_start is None:
        raise ValueError(
            f"{source_path} does not look like the Summary Stage manual: missing "
            f"{GENERAL_REGION_START!r} or {SITE_REGION_START!r} heading."
        )

    chapters: list[Chapter] = []
    system: str | None = None
    for position, section in enumerate(sections[general_start:], start=general_start):
        heading = section.heading
        folded = heading.casefold()

        if folded.startswith(APPENDIX_PREFIX):
            if not include_appendices:
                continue
        if any(folded.startswith(skip) for skip in _SKIP_HEADINGS):
            continue

        in_site_region = position >= site_start
        if in_site_region and section.level == 1 and _is_stub(section):
            system = heading
            continue
        if _is_stub(section):
            continue

        if in_site_region:
            preamble = chapter_preamble(section.body) or ""
            sites = parse_sites(preamble)
            histologies = parse_histologies(preamble)
            # A level-1 chapter with its own content (Breast, Bone) is its own
            # body system rather than a member of the preceding one.
            parent = None if section.level == 1 else system
        else:
            sites = histologies = ()
            parent = None

        chapters.append(
            Chapter(
                heading=heading,
                site_group=slugify_chapter(heading),
                scope="site" if in_site_region else "general",
                system=parent,
                start_line=section.start_line,
                end_line=section.end_line,
                sites=sites,
                histologies=histologies,
            )
        )
    return chapters


def sections_for(chapter: Chapter, sections: list[Section]) -> list[Section]:
    """The segmented sections a chapter spans, for the tagging pass.

    Chapters are indexed at heading level 2 but compiled from a finer
    segmentation, so a long chapter is tagged as several sections rather than one
    oversized LLM call.
    """
    return [
        section
        for section in sections
        if chapter.start_line <= section.start_line <= chapter.end_line
    ]
