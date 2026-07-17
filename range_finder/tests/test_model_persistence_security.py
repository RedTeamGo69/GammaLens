"""Restricted unpickling of model blobs (audit S2).

A poisoned saved_models BYTEA row (or .pkl on disk) must not be an RCE
primitive: only the data-science stack's constructors may be referenced by
the pickle stream.
"""
import pickle

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from range_finder.model_persistence import _restricted_loads


def test_real_model_payload_round_trips():
    X = sm.add_constant(pd.DataFrame({"x": np.arange(60.0)}))
    res = sm.OLS(pd.Series(np.arange(60.0) * 3 - 1), X).fit()
    payload = {"result": res, "fitted_at": "2026-07-16",
               "feature_cols": ["x"], "schema_version": 3}
    out = _restricted_loads(pickle.dumps(payload))
    assert out["feature_cols"] == ["x"]
    assert float(out["result"].params["x"]) == pytest.approx(3.0)


def test_os_system_payload_is_blocked():
    class Evil:
        def __reduce__(self):
            import os
            return (os.system, ("echo pwned",))

    with pytest.raises(pickle.UnpicklingError, match="Blocked unpickle"):
        _restricted_loads(pickle.dumps(Evil()))


def test_builtins_eval_is_blocked():
    class Evil:
        def __reduce__(self):
            return (eval, ("1+1",))

    with pytest.raises(pickle.UnpicklingError, match="Blocked unpickle"):
        _restricted_loads(pickle.dumps(Evil()))
