import ivy.functional.frontends.sklearn as sklearn_frontend
import numpy as np
import ivy
from hypothesis import given
import ivy_tests.test_ivy.helpers as helpers


# --- Helpers --- #
# --------------- #


# helper functions
def _get_sklearn_predict(X, y, max_depth, module=None):
    clf = module.tree.DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(X, y)
    return clf.predict


# --- Main --- #
# ------------ #


# todo: integrate with already existing strats and generalize
@given(
    X=helpers.array_values(shape=(5, 2), dtype=helpers.get_dtypes("float")),
    y=helpers.array_values(shape=(5,), dtype=helpers.get_dtypes("integer")),
)
def test_sklearn_tree_predict(X, y):
    try:
        import sklearn
    except ImportError:
        print("sklearn not installed, skipping test_sklearn_tree_predict")
        return
    sklearn_pred = _get_sklearn_predict(X, y, max_depth=3, module=sklearn)(X)
    for fw in helpers.available_frameworks:
        ivy.set_backend(fw)
        ivy_pred = _get_sklearn_predict(
            ivy.array(X), ivy.array(y), max_depth=3, module=sklearn_frontend
        )(X)
        assert np.allclose(ivy_pred.to_numpy(), sklearn_pred)
        ivy.unset_backend()
