import numpy as np

from asrs_eval.baselines import MajorityClass


def test_majority_class_hand_worked():
    m = MajorityClass().fit(["a"] * 4, ["X", "X", "X", "Y"])
    assert list(m.predict(["q", "r"])) == ["X", "X"]
    assert list(m.classes_) == ["X", "Y"]
    np.testing.assert_allclose(m.predict_proba(["q"]), [[0.75, 0.25]])
