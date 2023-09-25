import ivy

import pytest

# This test has been adapted from TensorLy
# https://github.com/tensorly/tensorly/blob/main/tensorly/tests/test_tt_matrix.py


@pytest.mark.parametrize(
    "shape, n_rows, n_cols",
    [((2, 2, 2, 3, 3, 3), 8, 27)],
)
def test_tt_matrix_manipulation(shape, n_rows, n_cols):
    tt_matrix = ivy.random_tt_matrix(shape, 2, full=False)
    rec = ivy.TTMatrix.tt_matrix_to_tensor(tt_matrix)
    assert ivy.shape(rec) == shape

    mat = ivy.TTMatrix.tt_matrix_to_matrix(tt_matrix)
    assert ivy.shape(mat) == (n_rows, n_cols)

    vec = ivy.TTMatrix.tt_matrix_to_vec(tt_matrix)
    assert ivy.shape(vec) == (n_rows * n_cols,)
