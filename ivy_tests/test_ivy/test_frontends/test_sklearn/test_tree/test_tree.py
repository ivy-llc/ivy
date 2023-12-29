from ivy.functional.frontends.sklearn.tree import DecisionTreeClassifier as ivy_DTC
import numpy as np
import ivy
from hypothesis import given
import ivy_tests.test_ivy.helpers as helpers


# --- Helpers --- #
# --------------- #


# helper functions
def _get_sklearn_predict(X, y, max_depth, DecisionTreeClassifier):
    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(X, y)
    return clf.predict


# --- Main --- #
# ------------ #


# todo: integrate with already existing strats and generalize
@given(
    X=helpers.array_values(
        shape=(5, 2), dtype=helpers.get_dtypes("float", prune_function=False)
    ),
    y=helpers.array_values(
        shape=(5,), dtype=helpers.get_dtypes("signed_integer", prune_function=False)
    ),
)
def test_sklearn_tree_predict(X, y):
    try:
        from sklearn.tree import DecisionTreeClassifier as sklearn_DTC
    except ImportError:
        print("sklearn not installed, skipping test_sklearn_tree_predict")
        return
    sklearn_pred = _get_sklearn_predict(X, y, 3, sklearn_DTC)(X)
    ivy_pred = _get_sklearn_predict(ivy.array(X), ivy.array(y), 3, ivy_DTC)(X)
    assert np.allclose(ivy_pred.to_numpy(), sklearn_pred)
