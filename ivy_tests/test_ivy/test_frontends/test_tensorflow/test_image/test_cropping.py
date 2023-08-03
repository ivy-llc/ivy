from typing import List, Type
from ivy_tests.test_ivy import helpers
from ivy_tests.test_ivy.helpers.hypothesis_helpers.array_helpers import dtype_and_values
from ivy_tests.test_ivy.helpers.hypothesis_helpers.dtype_helpers import get_dtypes
from ivy_tests.test_ivy.helpers.testing_helpers import handle_frontend_test
from ivy_tests.test_ivy.test_functional.test_experimental.test_nn.test_layers import (
    _interp_args,
)
import numpy as np
import enum
from hypothesis import strategies as st
import pytest
from hypothesis import assume


# Custom strategy for dtype_values_and_other argument
@pytest.hookimpl(trylast=True)
def pytest_hypothesis_register_custom_strategies():
    pass

    # Register the custom strategy for dtype_values_and_other argument
    @st.composite
    def dtype_values_and_other(draw):
        return draw(dtype_and_values(draw, []))

    return {
        "dtype_values_and_other": dtype_values_and_other,
    }


# Helper function to sample from a non-empty list
def sampled_from_nonempty_list(elements: List) -> Type[enum.Enum]:
    if not elements:
        raise ValueError("Cannot sample from an empty list.")
    return st.sampled_from(elements)


# Helper function for _max_pool_helper
@st.composite
def _max_pool_helper(draw, get_dtypes):
    sizes = [draw(st.integers(min_value=1, max_value=5)) for _ in range(3)]
    rates = [draw(st.integers(min_value=1, max_value=3)) for _ in range(3)]
    strides = [draw(st.integers(min_value=1, max_value=3)) for _ in range(3)]

    if len(sizes) < 3 or len(rates) < 3 or len(strides) < 3:
        raise ValueError("Sizes, rates, and strides must have at least 3 elements.")

    x_dim = []
    for i in range(1, 3):
        x_dim.append(draw(st.integers(min_value=1, max_value=5)))
    x_shape = [
        draw(st.integers(min_value=1, max_value=5)),
        *x_dim,
        draw(st.integers(min_value=1, max_value=5)),
    ]
    dtype_x = draw(dtype_and_values(get_dtypes, x_shape))
    kernel_size = [
        1,
        *draw(st.lists(st.integers(min_value=1, max_value=3), min_size=2, max_size=2)),
        1,
    ]
    strides = [1, *strides]
    padding = draw(st.sampled_from(["VALID", "SAME"]))

    return dtype_x, kernel_size, strides, padding


@st.composite
def extract_patches_helper(draw):
    sizes = draw(
        st.lists(st.integers(min_value=1, max_value=5), min_size=2, max_size=2)
    )
    rates = draw(
        st.lists(st.integers(min_value=1, max_value=3), min_size=2, max_size=2)
    )
    strides = draw(
        st.lists(st.integers(min_value=1, max_value=3), min_size=2, max_size=2)
    )

    num_elements = len(sizes)
    if num_elements < 3:
        # Raise a ValueError exception if the sizes list has less than 3 elements.
        raise ValueError("Sizes must have at least 3 elements.")

    for size in sizes:
        if size < 3:
            # Raise a ValueError exception if any of the sizes are less than 3.
            raise ValueError("Sizes must have at least 3 elements.")

    # Generate random values for the images tensor
    x_dim = [draw(st.integers(min_value=1, max_value=5)) for _ in range(3)]
    x_shape = [
        draw(st.integers(min_value=1, max_value=5)),
        *x_dim,
        draw(st.integers(min_value=1, max_value=5)),
    ]
    dtype_x = draw(dtype_and_values(get_dtypes, x_shape))
    x_values = np.random.randn(*x_shape).astype(dtype_x[0])

    # Generate random values for other arguments
    kernel_size = [
        1,
        *draw(st.lists(st.integers(min_value=1, max_value=3), min_size=2, max_size=2)),
        1,
    ]
    strides = [1, *strides]
    padding = draw(st.sampled_from(["VALID", "SAME"]))

    return (dtype_x, x_values), kernel_size, strides, padding, rates


# Test function for extract_patches
@handle_frontend_test(
    fn_tree="tensorflow.image.extract_patches",  # Update the function tree
    dtype_values_and_other=extract_patches_helper,
    test_with_out=st.just(True),
)
def test_tensorflow_extract_patches(
    dtype_values_and_other,
    frontend,
    test_flags,
    fn_tree,
):
    (x_dtype, x), sizes, strides, padding = dtype_values_and_other


# Test function for resize
@handle_frontend_test(
    fn_tree="tensorflow.image.resize",
    dtype_x_mode=_interp_args(
        mode_list=[
            "bilinear",
            "nearest",
            "area",
            "bicubic",
            "lanczos3",
            "lanczos5",
            "mitchellcubic",
            "gaussian",
        ]
    ),
    antialias=st.booleans(),
    test_with_out=st.just(False),
)
def test_tensorflow_resize(
    dtype_x_mode,
    antialias,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x, mode, size, _, _, preserve = dtype_x_mode
    try:
        helpers.test_frontend_function(
            input_dtypes=input_dtype,
            frontend=frontend,
            test_flags=test_flags,
            fn_tree=fn_tree,
            on_device=on_device,
            rtol=1e-01,
            atol=1e-01,
            image=x[0],
            size=size,
            method=mode,
            antialias=antialias,
            preserve_aspect_ratio=preserve,
        )
    except Exception as e:
        if hasattr(e, "message") and (
            "output dimensions must be positive" in e.message
            or "Input and output sizes should be greater than 0" in e.message
        ):
            assume(False)
        raise e
