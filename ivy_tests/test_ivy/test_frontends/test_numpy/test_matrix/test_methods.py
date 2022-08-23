# global
import inspect

import pytest

import ivy
import numpy as np
from hypothesis import given, strategies as st, settings

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@handle_cmd_line_args
@given(
    dtypes_and_xs=helpers.dtype_and_values(
        available_dtypes=ivy.valid_numeric_dtypes,
        num_arrays=1,
        shared_dtype=True,
    ),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.min"
    ),
)
def test_numpy_min(
        dtypes_and_xs,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        fw,

):
    input_dtypes, xs = dtypes_and_xs
    xs = [np.asarray(x, dtype=dt) for x, dt in zip(xs, input_dtypes)]
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="min",
        arrays=xs,
        axis=None,
        out=None,
        keepdims=False,
    )
