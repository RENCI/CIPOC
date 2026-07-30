import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
