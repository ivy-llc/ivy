import pytest
import ivy
from ivy.functional.frontends.sklearn.utils.multiclass import type_of_target


# not suitable for usual frontend testing
@pytest.mark.parametrize(
    ("y", "label"),
    [
        ([1.2], "continuous"),
        ([1], "binary"),
        ([1, 2], "binary"),
        ([1, 2, 3], "multiclass"),
        ([1, 2, 3, 4], "multiclass"),
        ([1, 2, 3, 4, 5], "multiclass"),
        ([1, 2, 2], "binary"),
        ([1, 2.0, 2, 3], "multiclass"),
        ([1.0, 2.0, 2.0], "binary"),
        ([[[1, 2], [3, 4]]], "unknown"),
        ([[1, 2], [1, 1]], "multilabel-indicator"),
    ],
)
def test_sklearn_type_of_target(y, label):
    assert type_of_target(ivy.array(y)) == label
