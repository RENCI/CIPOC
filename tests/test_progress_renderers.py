import io
import os
import re
import unittest
from dataclasses import replace

from cipoc.utils.progress.model import (
    BranchSnapshot,
    GroupSnapshot,
    Snapshot,
    Stage,
    VariableSnapshot,
)
from cipoc.utils.progress.renderers import AnsiAltScreen, NotebookDisplay, PlainLog


ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def snapshots():
    variable = VariableSnapshot(390, "Date <of> Diagnosis", "initial")
    group = GroupSnapshot("initial", "Initial", 0, item_ids=(390,))
    base = Snapshot(
        description="Renderer Test",
        started_at=10.0,
        groups=(group,),
        variables={390: variable},
        notes_total=2,
    )
    active_variable = replace(variable, stage=Stage.VALIDATE, attempt=2)
    active_group = replace(
        group, active=True, stage=Stage.VALIDATE, note_count=1
    )
    active = replace(
        base,
        groups=(active_group,),
        variables={390: active_variable},
        branches=(
            BranchSnapshot(
                key="initial",
                label="Initial",
                stage=Stage.VALIDATE,
                variables=1,
                note_count=1,
                started_at=10.5,
            ),
        ),
        notes_done=1,
    )
    terminal_variable = replace(
        active_variable,
        stage=Stage.DONE,
        attempt=0,
        status="extracted",
        value="\x1b[31m20260101",
        confidence="low",
        flag="?",
    )
    terminal = replace(
        active,
        groups=(replace(active_group, active=False, stage=Stage.DONE),),
        variables={390: terminal_variable},
        branches=(),
        finished=True,
        review_flags=1,
    )
    return base, active, terminal


class PlainLogTests(unittest.TestCase):
    def test_is_append_only_deduplicated_and_contains_no_escape_bytes(self):
        base, active, terminal = snapshots()
        stream = io.StringIO()
        renderer = PlainLog(stream)

        self.assertTrue(renderer.paint(base, now=10.0))
        self.assertTrue(renderer.paint(active, now=11.0))
        self.assertFalse(renderer.paint(active, now=11.1))
        self.assertTrue(renderer.paint(terminal, now=12.0, final=True))

        output = stream.getvalue()
        self.assertEqual(
            output.splitlines(),
            [
                "CIPOC / Renderer Test",
                "notes 1/2 scanned",
                "group Initial: validate · 1 variable · 1 note",
                "390 Date <of> Diagnosis: validate·2",
                "group Initial: complete",
                "390 Date <of> Diagnosis: ✔ extracted = 20260101 ? [low]",
                "complete: 1/1 variables in 00:02 · 1 flagged for review",
            ],
        )
        self.assertNotIn("\x1b", output)

    def test_compact_completion_does_not_report_zero_variables(self):
        stream = io.StringIO()
        renderer = PlainLog(stream)
        snapshot = Snapshot(
            description="Note Scanner",
            started_at=10.0,
            finished=True,
        )

        renderer.paint(snapshot, now=12.0, final=True)

        output = stream.getvalue()
        self.assertIn("complete in 00:02", output)
        self.assertNotIn("variables", output)


class AnsiAltScreenTests(unittest.TestCase):
    def test_paints_styled_rows_throttles_and_restores_terminal(self):
        _, active, _ = snapshots()
        stream = io.StringIO()
        renderer = AnsiAltScreen(
            stream,
            color=True,
            size_provider=lambda: os.terminal_size((80, 24)),
        )

        with renderer:
            self.assertTrue(renderer.paint(active, now=11.0, tick=0))
            self.assertFalse(renderer.paint(active, now=11.05, tick=1))
            self.assertTrue(renderer.paint(active, now=11.05, tick=1, final=True))

        output = stream.getvalue()
        self.assertTrue(output.startswith("\x1b[?1049h\x1b[?25l"))
        self.assertTrue(output.endswith("\x1b[0m\x1b[?25h\x1b[?1049l"))
        self.assertIn("\x1b[1;36m", output)
        self.assertIn("validate·2", ANSI.sub("", output))
        self.assertEqual(output.count("\x1b[H"), 3)


class _Html:
    def __init__(self, data):
        self.data = data


class _Handle:
    def __init__(self):
        self.updates = []

    def update(self, value):
        self.updates.append(value)


class NotebookDisplayTests(unittest.TestCase):
    def test_updates_one_html_handle_throttles_and_always_paints_final(self):
        _, active, terminal = snapshots()
        calls = []
        handle = _Handle()

        def display(value, **kwargs):
            calls.append((value, kwargs))
            return handle

        renderer = NotebookDisplay(
            width=80,
            display_fn=display,
            html_factory=_Html,
        )
        self.assertTrue(renderer.paint(active, now=11.0))
        self.assertFalse(renderer.paint(active, now=11.1))
        self.assertTrue(renderer.paint(terminal, now=11.1, final=True))

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], {"display_id": True})
        self.assertEqual(len(handle.updates), 1)
        first = calls[0][0].data
        final = handle.updates[0].data
        self.assertIn("Date &lt;of&gt; Diagnosis", first)
        self.assertIn("validate·2", first)
        self.assertIn("202601", final)
        self.assertNotIn("[31m", final)
        self.assertNotIn("\x1b", first + final)


if __name__ == "__main__":
    unittest.main()
