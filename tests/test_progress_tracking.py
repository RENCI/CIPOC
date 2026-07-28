import io
import os
import re
import unittest

from cipoc.agents.extractor import ExtractorAgent
from cipoc.agents.note_retriever import NoteRetrieverAgent
from cipoc.agents.note_scanner import NoteScannerAgent
from cipoc.models import ClinicalNote
from cipoc.tools import load_variable_groups
from cipoc.utils.progress_tracking import _OrchestratorProgressDisplay


class OrchestratorProgressDisplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.groups = load_variable_groups("config/variable_groups.json")
        cls.item_ids = [
            str(variable.item_id)
            for group in cls.groups
            for variable in group.variables
        ]

    def display(self, stream=None):
        return _OrchestratorProgressDisplay(
            "Orchestrator",
            {"note_corpus": {1: {}, 2: {}}},
            self.groups,
            stream=stream or io.StringIO(),
        )

    def render(self, columns, lines):
        display = self.display()
        display._terminal_size = lambda: os.terminal_size((columns, lines))
        return display, display._render(final=False)

    def test_every_item_renders_once_within_viewport(self):
        for columns, height in ((80, 24), (120, 30)):
            with self.subTest(size=(columns, height)):
                _, rendered = self.render(columns, height)
                text = "\n".join(rendered)
                self.assertLessEqual(len(rendered), height)
                for item_id in self.item_ids:
                    matches = re.findall(rf"(?<!\d){item_id}(?!\d)", text)
                    self.assertEqual(len(matches), 1, item_id)

    def test_names_expand_with_available_width(self):
        _, compact = self.render(40, 24)
        _, narrow = self.render(80, 24)
        _, wide = self.render(120, 30)
        self.assertNotIn("Date of Diagn", "\n".join(compact))
        self.assertIn("Date o", "\n".join(narrow))
        self.assertIn("Date of Diagnosis", "\n".join(wide))

    def test_group_names_precede_their_variables(self):
        _, rendered = self.render(120, 30)
        text = "\n".join(rendered)
        for group in self.groups:
            self.assertEqual(text.count(f"[{group.name}]"), 1)
            self.assertLess(text.index(f"[{group.name}]"), text.index(str(group.variables[0].item_id)))

    def test_group_running_and_durable_status_updates(self):
        display = self.display()
        group = self.groups[0]
        display.start(
            "extract-1",
            (),
            "extract_branch",
            {"requested_variables": group},
        )
        for variable in group.variables:
            self.assertEqual(
                display._display_status(display.variables[str(variable.item_id)]),
                "running",
            )

        statuses = [
            "extracted",
            "structured_data",
            "not_found",
            "not_applicable",
            "blocked",
            "error",
        ]
        results = {
            int(item_id): {
                "item_id": int(item_id),
                "status": status,
                "value": "1" if status in {"extracted", "structured_data"} else None,
                "reason": "test reason" if status in {"blocked", "error"} else None,
            }
            for item_id, status in zip(self.item_ids, statuses)
        }
        display.update_values({"variable_results": results})
        for item_id, status in zip(self.item_ids, statuses):
            self.assertEqual(display.variables[item_id].status, status)
            self.assertFalse(display.variables[item_id].running)

    def test_noninteractive_output_is_start_and_summary_only(self):
        stream = io.StringIO()
        display = self.display(stream)
        display.draw()
        display.start("initialize", (), "initialize", {})
        display.finish("initialize", (), "initialize", {})
        display.complete()
        output = stream.getvalue().splitlines()
        self.assertEqual(len(output), 2)
        self.assertIn("52 variables", output[0])
        self.assertIn("complete:", output[1])

    def test_redraw_tracks_shorter_resized_dashboard(self):
        stream = io.StringIO()
        display = self.display(stream)
        display._redraw_terminal(["x"] * 24)
        display._redraw_terminal(["x"] * 20)
        self.assertEqual(display._rendered_lines, 20)

    def test_subagents_can_run_without_creating_progress_displays(self):
        class FakeGraph:
            def __init__(self, result):
                self.result = result
                self.inputs = []

            def invoke(self, graph_input):
                self.inputs.append(graph_input)
                return self.result

        note = ClinicalNote(note_id=1, date="2026-01-01", type="test", content="")

        scanner = object.__new__(NoteScannerAgent)
        scanner._graph = FakeGraph({})
        scanned = scanner.run(note, progress=False)
        self.assertEqual(scanned.note_id, 1)

        retriever = object.__new__(NoteRetrieverAgent)
        retriever._graph = FakeGraph({"relevant_note_ids": [1]})
        self.assertEqual(retriever.run({}, progress=False), [1])

        extractor = object.__new__(ExtractorAgent)
        extractor._graph = FakeGraph({"extracted_values": None})
        self.assertIsNone(extractor.run({}, progress=False).extracted_values)


if __name__ == "__main__":
    unittest.main()
