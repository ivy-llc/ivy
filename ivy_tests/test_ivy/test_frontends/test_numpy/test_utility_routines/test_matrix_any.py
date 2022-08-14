# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers


@st.composite
def _array_with_dtype_axis_keepdims_and_where(draw):
    dtypes = draw(
                helpers.array_dtypes(
                    num_arrays=1
        )
    )
    shape = draw(
                helpers.get_shape(
                    min_num_dims=1,
                    max_num_dims=5,
        )
    )
    axis = draw(
               helpers.integers(min_value=-1, max_value=len(shape)-1)
    )
    if axis == -1:
        axis = None
    x = draw(
            helpers.array_values(
                shape=shape,
                dtype=dtypes[0]
        )
    )
    where_shape_length = draw(helpers.integers(min_value=0,max_value=len(shape)))
    if where_shape_length != 0:
        where_nb_dims_to_change = draw(helpers.integers(min_value=0, max_value=where_shape_length-1))
        where_dims_to_change = [
                                   draw(
                                       helpers.integers(min_value=0, max_value=where_shape_length-1)
            ) for i in range(where_nb_dims_to_change)
        ]
        where_dims_list = [1]*where_shape_length
        for dim in where_dims_to_change:
            where_dims_list[dim] = shape[::-1][dim]
        where_dims_list = where_dims_list[::-1]
        where = draw(
                    helpers.array_values(
                        shape=where_dims_list,
                        dtype="bool"
            )
        )
    else:
        where = draw(
        st.booleans()
        ) 
    keepdims = draw(
                   st.booleans()
    )
    return x, dtypes[0], axis, keepdims, where

@given(
    x_dtype_axis_keepdims_where=_array_with_dtype_axis_keepdims_and_where(),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    native_array=st.booleans()
)
def test_numpy_any(
    x_dtype_axis_keepdims_where,
    as_variable,
    with_out,
    native_array,
    fw
):
    x, input_dtype, axis, keepdims, where = x_dtype_axis_keepdims_where
    x = np.asarray(x, dtype=input_dtype)
    num_positional_args = 1
    input_dtypes = [input_dtype]
    as_variable = [as_variable]
    native_array = [native_array]
    where = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        as_variable=as_variable,
        native_array=native_array
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_name="any",
        x=x,
        axis=axis,
        out=None,
        keepdims=keepdims,
        where=where
    )


