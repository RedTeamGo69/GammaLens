"""saved_models is the only store: a failed save must surface, not vanish.

The old on-disk pickle fallback wrote to Streamlit Cloud's throwaway disk
while the tab reported "rows saved", so other sessions kept the stale fit.
"""
import pytest

from range_finder import model_persistence as mp


class _Conn:
    def __init__(self, row=None, fail=False):
        self.row, self.fail = row, fail

    def execute(self, sql, params=None):
        if self.fail:
            raise RuntimeError("connection reset")
        return self

    def fetchone(self):
        return self.row

    def commit(self):
        pass


def test_save_failure_raises():
    with pytest.raises(RuntimeError, match="connection reset"):
        mp.save_model(object(), ["har_d1"], "M1_baseline", {}, conn=_Conn(fail=True))


def test_missing_fit_is_file_not_found():
    with pytest.raises(FileNotFoundError):
        mp.load_model("M1_baseline", conn=_Conn(row=None), ticker="SPX")
