"""
# test_guardrails.py - unit tests for agent/guardrails.py (pure functions, no live infra)

- Runs every case in eval/guardrail_queries.yaml through check_input(), including must-NOT-block cases.
- Checks check_output()'s groundedness heuristic.
"""

from pathlib import Path

import pytest
import yaml
from retailgraph.agent.guardrails import check_input, check_output

CASES = yaml.safe_load(
    (Path(__file__).parent.parent.parent / "eval" / "guardrail_queries.yaml").read_text()
)


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_guardrail_case(case):
    result = check_input(case["query"])
    assert result["blocked"] == case["expect_blocked"], (
        f"{case['id']}: expected blocked={case['expect_blocked']}, "
        f"got {result['blocked']} (categories={result['categories']})"
    )
    if case["expect_blocked"]:
        assert set(case["categories"]) & set(result["categories"]), (
            f"{case['id']}: expected one of {case['categories']}, got {result['categories']}"
        )


class TestCheckOutput:

    def test_no_results_always_grounded(self):
        assert check_output("No products found matching your search.", []) is True

    def test_grounded_when_result_name_appears_in_answer(self):
        raw = [{"item_name": "SOLELY Organic Banana Fruit Jerky"}]
        answer = "You might like SOLELY Organic Banana Fruit Jerky for $1.94."
        assert check_output(answer, raw) is True

    def test_not_grounded_when_no_result_name_appears(self):
        raw = [{"item_name": "SOLELY Organic Banana Fruit Jerky"}]
        answer = "You might like Completely Different Product Name."
        assert check_output(answer, raw) is False
