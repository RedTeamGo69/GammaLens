from pathlib import Path
import re

from range_finder.forward_test.config import UNIVERSE
from ui_theme import QUICK_TICKERS

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "scheduled_snapshot.yml"
# Captured for the app but not forward-tested (see the workflow's matrix comment).
CAPTURE_EXTRAS = {"NDX"}


def test_cron_matrix_captures_exactly_the_forward_test_roster():
    # GEX/EM/weekly-setup capture must cover every forward-test ticker plus the app-only extras.
    block = re.search(r"ticker: \[(.*?)\]", WORKFLOW.read_text(encoding="utf-8"), re.S).group(1)
    matrix = [t.strip() for t in block.split(",")]
    assert len(matrix) == len(set(matrix))
    assert set(matrix) == set(UNIVERSE) | CAPTURE_EXTRAS


def test_quick_picks_are_the_forward_test_roster():
    assert QUICK_TICKERS == list(UNIVERSE)
