import unittest
from pathlib import Path
from unittest.mock import patch

from cipoc.agents.orchestrator import OrchestratorAgent
from cipoc.models import CaseFacts
from cipoc.tools import load_variable_groups, site_applies


VARIABLE_GROUPS = Path(__file__).resolve().parents[1] / "config" / "variable_groups.json"


class SiteApplicabilityTests(unittest.TestCase):
    def test_item_832_accepts_coded_breast_primary_site(self):
        group = next(
            group
            for group in load_variable_groups(VARIABLE_GROUPS)
            if any(variable.item_id == 832 for variable in group.variables)
        )

        self.assertTrue(site_applies(group.applies_to, CaseFacts(primary_site="C50.4")))
        self.assertFalse(site_applies(group.applies_to, CaseFacts(primary_site="C34.9")))


class OrchestratorRunTests(unittest.TestCase):
    @patch("cipoc.agents.orchestrator.run_with_progress", return_value={})
    def test_max_concurrency_is_forwarded(self, run_with_progress):
        agent = object.__new__(OrchestratorAgent)
        agent._graph = object()
        agent._target_variables = []
        agent._target_group_hierarchy = []

        agent.run([], max_concurrency=48)

        self.assertEqual(
            run_with_progress.call_args.kwargs["config"],
            {"max_concurrency": 48},
        )

    def test_max_concurrency_must_be_positive(self):
        agent = object.__new__(OrchestratorAgent)

        with self.assertRaisesRegex(ValueError, "at least 1"):
            agent.run([], max_concurrency=0)


if __name__ == "__main__":
    unittest.main()
