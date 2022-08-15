import random
from math import prod
import numpy as np

from hypothesis import given, strategies as st
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np


def _products(n, min_divisor=2):

    if n == 1:
        yield []

    for divisor in range(min_divisor, n + 1):
        if n % divisor == 0:
            for product in _products(n // divisor, divisor):
                yield product + [divisor]


@st.composite
def _a_newshape_and_dtype(draw):

    array_dtype = draw(
        st.shared(
            helpers.array_dtypes(
                num_arrays=1,
                available_dtypes=ivy_np.valid_float_dtypes
            ), key="array_dtype"
        )
    )
    array_shape = draw(
        st.shared(
            helpers.get_shape(
                min_num_dims=2,
                max_num_dims=5,
                min_dim_size=2,
                max_dim_size=10,
            ), key='array_shape')
    )
    array = draw(
        st.shared(helpers.array_values(
            dtype=array_dtype[0],
            shape=array_shape,
            min_value=0,
            max_value=1,
        ), key='array')
    )

    # Create all possible reshape candidates and select one of them randomly
    divisor_combinations = list(_products(prod(array_shape)))
    newshape = random.choice(divisor_combinations)

    return array, newshape, array_dtype


@given(
    _a_newshape_and_dtype=_a_newshape_and_dtype(),
    as_variable=helpers.list_of_length(x=st.booleans(), length=1),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.reshape"
    ),
    native_array=helpers.list_of_length(x=st.booleans(), length=1),
)
def test_numpy_reshape(
    _a_newshape_and_dtype,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):

    a, new_shape, dtype = _a_newshape_and_dtype

    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="reshape",
        a=np.asarray(a),
        newshape=-1,
    )
