"""Tests for the Summary Stage 2018 chapter index.

The index is what tells the compiler which lines of the manual are a chapter and
what ICD-O-3 scope to compile them under, so an error here silently mis-scopes
rules rather than failing loudly. These tests pin the parsing against the real
manual: the codes asserted below are read off the chapter preambles in
``documents/markdown/Summary-Stage_v3.3.md``.
"""

import unittest
from pathlib import Path

from scripts.rule_compilation.summary_stage_index import (
    EFFECTIVE_DX_DATE_MIN,
    MANIFEST_ENTRY,
    build_index,
    chapter_preamble,
    parse_histologies,
    parse_sites,
    slugify_chapter,
)

SOURCE = Path("documents/markdown/Summary-Stage_v3.3.md")


class PreambleTests(unittest.TestCase):
    def test_preamble_ends_at_the_first_note_line(self):
        body = "8000-8700 C500\n\n**Note 1: Sources used**\n\n- C700 mentioned in a note\n"
        self.assertEqual(chapter_preamble(body).strip(), "8000-8700 C500")

    def test_missing_note_terminator_yields_no_preamble(self):
        """A guidance section has no code header; it must not be scoped from its prose."""
        body = "| PRIMARY SITE | ICD-O |\n| LIP | C00_ |\n| HARD PALATE | C050 |\n"
        self.assertIsNone(chapter_preamble(body))

    def test_exclusion_clauses_are_stripped(self):
        body = "8720-8790 [except C500] C509 Lip, NOS (excludes skin of lip C440)\n**Note 1:**\n"
        preamble = chapter_preamble(body)
        self.assertEqual(parse_sites(preamble), ("C509",))
        self.assertEqual(parse_histologies(preamble), ("8720-8790",))


class CodeParsingTests(unittest.TestCase):
    def test_ranges_and_codes_merge_into_minimal_spans(self):
        self.assertEqual(
            parse_sites("C500-C506, C508-C509 C500 Nipple C501 Central portion"),
            ("C500-C506", "C508-C509"),
        )

    def test_year_spans_are_not_read_as_histologies(self):
        """'2018-2024' and '(2023+)' are effective-date notes, not morphology codes."""
        self.assertEqual(parse_histologies("2018-2024 8000-8700 (2023+) 9671"), ("8000-8700", "9671"))

    def test_non_code_text_yields_nothing(self):
        self.assertEqual(parse_sites("Note: see the chapter above"), ())
        self.assertEqual(parse_histologies("Schema Discriminator 1: 0, 3, 9"), ())

    def test_slugify_chapter(self):
        self.assertEqual(slugify_chapter("LARYNX SUPRAGLOTTIC"), "larynx_supraglottic")
        self.assertEqual(slugify_chapter("BONE (INCLUDING JOINTS)"), "bone_including_joints")


class IndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SOURCE.exists():
            raise unittest.SkipTest(f"{SOURCE} not present")
        cls.chapters = build_index(SOURCE)
        cls.by_group = {c.site_group: c for c in cls.chapters}

    def test_site_groups_are_unique(self):
        """Duplicate stems would have chapters overwrite each other's output file."""
        self.assertEqual(len(self.by_group), len(self.chapters))

    def test_general_chapters_come_first_and_are_unscoped(self):
        scopes = [c.scope for c in self.chapters]
        self.assertEqual(scopes.count("general"), scopes.index("site"))
        for chapter in self.chapters:
            if chapter.scope == "general":
                self.assertEqual(chapter.sites, ())
                self.assertEqual(chapter.histologies, ())

    def test_general_region_covers_the_stage_code_definitions(self):
        general = {c.site_group for c in self.chapters if c.scope == "general"}
        self.assertIn("code_0_in_situ", general)
        self.assertIn("code_7_distant", general)
        self.assertIn("ambiguous_terminology", general)
        self.assertIn("how_to_assign_summary_stage", general)

    def test_body_system_dividers_are_not_chapters(self):
        """'# HEAD AND NECK' carries no content; it is only the parent of what follows."""
        self.assertNotIn("head_and_neck", self.by_group)
        self.assertNotIn("female_genital_system", self.by_group)
        self.assertEqual(self.by_group["lip"].system, "HEAD AND NECK")

    def test_level_one_chapter_with_content_is_its_own_system(self):
        """Breast is a '#' heading with a full chapter body, not a divider."""
        breast = self.by_group["breast"]
        self.assertEqual(breast.scope, "site")
        self.assertIsNone(breast.system)
        self.assertEqual(breast.sites, ("C500-C506", "C508-C509"))

    def test_appendices_are_excluded_by_default(self):
        self.assertFalse([c for c in self.chapters if c.site_group.startswith("appendix")])
        with_appendices = build_index(SOURCE, include_appendices=True)
        self.assertTrue([c for c in with_appendices if c.site_group.startswith("appendix")])

    def test_chapter_scopes_match_the_manual(self):
        cases = {
            "lip": (("C003-C005", "C008-C009"), "8982"),
            "prostate": (("C619",), "8720-8790"),
            "melanoma_uvea": (("C693-C694",), "8720-8790"),
            "medulloblastoma": (("C700-C729", "C753"), "9508"),
            "pleural_mesothelioma": (("C340-C343", "C348-C349", "C384"), "9050-9053"),
        }
        for group, (sites, histology) in cases.items():
            with self.subTest(group=group):
                self.assertEqual(self.by_group[group].sites, sites)
                self.assertIn(histology, self.by_group[group].histologies)

    def test_brain_header_is_read_past_its_long_histology_block(self):
        """Brain lists two behavior-split histology blocks before its site codes."""
        self.assertEqual(self.by_group["brain"].sites, ("C700", "C710-C719"))

    def test_guidance_sections_stay_unscoped(self):
        """No code header means widen to all cases, never a scope scraped from a prose table."""
        for group in (
            "distinguishing_in_situ_and_localized_tumors_for_lip_oral_cavity_and_pharynx",
            "bladder_renal_pelvis_and_ureters_anatomic_structures",
            "corpus_uteri",
        ):
            with self.subTest(group=group):
                self.assertEqual(self.by_group[group].sites, ())
                self.assertEqual(self.by_group[group].histologies, ())

    def test_applicability_carries_the_effective_date_and_widens_when_empty(self):
        breast = self.by_group["breast"].applicability
        self.assertEqual(breast.dx_date_min, EFFECTIVE_DX_DATE_MIN)
        self.assertEqual(breast.sites, ["C500-C506", "C508-C509"])

        general = self.chapters[0].applicability
        self.assertIsNone(general.sites)
        self.assertIsNone(general.histologies)
        self.assertEqual(general.dx_date_min, EFFECTIVE_DX_DATE_MIN)

    def test_line_ranges_are_disjoint_and_ordered(self):
        spans = [(c.start_line, c.end_line) for c in self.chapters]
        self.assertEqual(spans, sorted(spans))
        for (_, prev_end), (next_start, _) in zip(spans, spans[1:]):
            self.assertLess(prev_end, next_start)

    def test_manifest_entry_is_dated_by_edition_not_effective_date(self):
        """Precedence ranks same-family manuals by publication_date.

        Dating this entry to the 2018 effective date would put it behind SPCSM
        2024, whose summary of Summary Stage would then outrank the dedicated
        manual and drop every unit compiled here.
        """
        self.assertGreater(MANIFEST_ENTRY["publication_date"], "2024-01-01")
        self.assertLess(EFFECTIVE_DX_DATE_MIN, MANIFEST_ENTRY["publication_date"])


if __name__ == "__main__":
    unittest.main()
