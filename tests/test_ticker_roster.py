from pathlib import Path
import re

from phase1.config import EXTRA_CAPTURE_INDICES
from range_finder.forward_test.config import UNIVERSE
from ui_theme import QUICK_TICKERS

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "scheduled_snapshot.yml"


def test_cron_matrix_captures_exactly_the_roster_plus_extra_indices():
    # GEX/EM/weekly-setup capture must cover every forward-test ticker, the
    # extra cash indices, and nothing else.
    block = re.search(r"ticker: \[(.*?)\]", WORKFLOW.read_text(encoding="utf-8"), re.S).group(1)
    matrix = [t.strip() for t in block.split(",")]
    assert len(matrix) == len(set(matrix))
    assert set(matrix) == set(UNIVERSE) | set(EXTRA_CAPTURE_INDICES)


def test_quick_picks_are_the_roster_plus_extra_indices():
    assert QUICK_TICKERS == [*UNIVERSE, *EXTRA_CAPTURE_INDICES]
    # Extras must stay out of the study roster (that needs a new study ID).
    assert not set(EXTRA_CAPTURE_INDICES) & set(UNIVERSE)
