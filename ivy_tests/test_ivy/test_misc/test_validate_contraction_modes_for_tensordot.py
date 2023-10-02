import pytest

from ivy.utils.tensordot_contraction_modes import (
    _get_valid_contraction_modes_for_axes,
    _get_valid_contraction_modes_for_batches,
)


@pytest.fixture
def shapes():
    shape1 = (3, 4, 5, 2, 2, 4)
    shape2 = (3, 3, 5, 2, 3, 4)
    return shape1, shape2


@pytest.mark.parametrize(
    "modes, expected_modes1, expected_modes2",
    [
        ([(0, -2, -1), (1, -3, -1)], [0, 4, 5], [1, 3, 5]),
    ],
)
def test_negative_dims_become_positive(shapes, modes, expected_modes1, expected_modes2):
    shape1, shape2 = shapes
    modes1, modes2 = _get_valid_contraction_modes_for_axes(shape1, shape2, modes)
    assert (expected_modes1, expected_modes2) == (
        modes1,
        modes2,
    ), f"Got ({modes1}, {modes2}), but expected ({expected_modes1}, {expected_modes2})."


@pytest.mark.parametrize("modes", [[(0, 1, 4), (1, 5, 3, 3)]])
def test_not_equal_length_modes_raises_error(shapes, modes):
    shape1, shape2 = shapes
    modes = [(0, 1, 4), (1, 5, 3, 3)]
    with pytest.raises(ValueError):
        _get_valid_contraction_modes_for_axes(shape1, shape2, modes)
    with pytest.raises(ValueError):
        _get_valid_contraction_modes_for_batches(shape1, shape2, modes)


@pytest.mark.parametrize("modes", [[(0, 1, 4), (2, 5, 3)]])
def test_not_same_dimension_in_both_tensors_raises_error(shapes, modes):
    shape1, shape2 = shapes
    with pytest.raises(ValueError):
        _get_valid_contraction_modes_for_axes(shape1, shape2, modes)
    with pytest.raises(ValueError):
        _get_valid_contraction_modes_for_batches(shape1, shape2, modes)


@pytest.mark.parametrize(
    "modes, expected_modes1, expected_modes2",
    [
        (0, [], []),
        ([(0, 1, 4), (1, 5, 3)], [0, 1, 4], [1, 5, 3]),
    ],
)
def test_validate_contraction_modes_for_axes(
    shapes, modes, expected_modes1, expected_modes2
):
    shape1, shape2 = shapes
    modes1, modes2 = _get_valid_contraction_modes_for_axes(shape1, shape2, modes)
    assert (expected_modes1, expected_modes2) == (
        modes1,
        modes2,
    ), f"Got ({modes1}, {modes2}), but expected ({expected_modes1}, {expected_modes2})."


@pytest.mark.parametrize(
    "modes, expected_batched_modes1, expected_batched_modes2",
    [
        ([(0, 2), (0, 2)], [0, 2], [0, 2]),
        ((0, 2, 5), [0, 2, 5], [0, 2, 5]),
        (0, [0], [0]),
    ],
)
def test_validate_contraction_modes_for_batches(
    shapes, modes, expected_batched_modes1, expected_batched_modes2
):
    shape1, shape2 = shapes
    batched_modes1, batched_modes2 = _get_valid_contraction_modes_for_batches(
        shape1,
        shape2,
        modes,
    )
    assert (expected_batched_modes1, expected_batched_modes2) == (
        batched_modes1,
        batched_modes2,
    ), (
        f"Got ({batched_modes1}, {batched_modes2}), "
        f"but expected ({expected_batched_modes1}, {expected_batched_modes2})."
    )
