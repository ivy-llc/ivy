# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
import ivy.functional.backends.torch as ivy_torch
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# noinspection DuplicatedCode
@st.composite
def _arrays_idx_n_dtypes(draw):
    num_dims = draw(st.shared(helpers.ints(min_value=1, max_value=4), key="num_dims"))
    num_arrays = draw(
        st.shared(helpers.ints(min_value=2, max_value=4), key="num_arrays")
    )
    common_shape = draw(
        helpers.lists(
            arg=helpers.ints(min_value=2, max_value=3),
            min_size=num_dims - 1,
            max_size=num_dims - 1,
        )
    )
    unique_idx = draw(helpers.ints(min_value=0, max_value=num_dims - 1))
    unique_dims = draw(
        helpers.lists(
            arg=helpers.ints(min_value=2, max_value=3),
            min_size=num_arrays,
            max_size=num_arrays,
        )
    )
    xs = list()
    available_dtypes = set(ivy_torch.valid_float_dtypes).intersection(
        ivy_torch.valid_float_dtypes
    )
    input_dtypes = draw(helpers.array_dtypes(available_dtypes=available_dtypes))
    for ud, dt in zip(unique_dims, input_dtypes):
        x = draw(
            helpers.array_values(
                shape=common_shape[:unique_idx] + [ud] + common_shape[unique_idx:],
                dtype=dt,
            )
        )
        xs.append(x)
    return xs, input_dtypes, unique_idx


# cat
@handle_cmd_line_args
@given(
    xs_n_input_dtypes_n_unique_idx=_arrays_idx_n_dtypes(),
    as_variable=helpers.array_bools(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.cat"
    ),
    native_array=helpers.array_bools(),
    with_out=st.booleans(),
)
def test_torch_cat(
    xs_n_input_dtypes_n_unique_idx,
    as_variable,
    num_positional_args,
    native_array,
    with_out,
    fw,
):
    xs, input_dtypes, unique_idx = xs_n_input_dtypes_n_unique_idx
    xs = [np.asarray(x, dtype=dt) for x, dt in zip(xs, input_dtypes)]
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="cat",
        tensors=xs,
        dim=unique_idx,
        out=None,
    )


# concat
@given(
    xs_n_input_dtypes_n_unique_idx=_arrays_idx_n_dtypes(),
    as_variable=helpers.array_bools(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.concat"
    ),
    native_array=helpers.array_bools(),
    with_out=st.booleans(),
)
def test_torch_concat(
    xs_n_input_dtypes_n_unique_idx,
    as_variable,
    num_positional_args,
    native_array,
    with_out,
    fw,
):
    xs, input_dtypes, unique_idx = xs_n_input_dtypes_n_unique_idx
    xs = [np.asarray(x, dtype=dt) for x, dt in zip(xs, input_dtypes)]
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="concat",
        tensors=xs,
        dim=unique_idx,
        out=None,
    )


# permute
@given(
    dtype_values_axis=helpers.dtype_values_axis(
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(
                set(ivy_torch.valid_float_dtypes)
            ),
        ),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
    axis=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
        ret_tuple=True,
    ),
    as_variable=helpers.array_bools(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.permute"
    ),
    native_array=helpers.array_bools(),
)
def test_torch_permute(
    dtype_values_axis,
    axis,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    dtype, value, axis = dtype_values_axis
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="permute",
        input=np.asarray(value, dtype=dtype),
        dims=axis,
    )


# swapdims
@given(
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(
                set(ivy_torch.valid_float_dtypes)
            )
        ),
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"),
    ),
    dim0=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"),
    ).filter(lambda axis: isinstance(axis, int)),
    dim1=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"),
    ).filter(lambda axis: isinstance(axis, int)),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.swapdims"
    ),
    native_array=st.booleans(),
)
def test_torch_swapdims(
    dtype_and_values,
    dim0,
    dim1,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, value = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="swapdims",
        input=np.asarray(value, dtype=input_dtype),
        dim0=dim0,
        dim1=dim1,
    )


# reshape
@given(
    dtype_value_shape=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(
                set(ivy_torch.valid_float_dtypes)
            ),
        ),
        ret_shape=True,
    ),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.reshape"
    ),
    native_array=st.booleans(),
)
def test_torch_reshape(
    dtype_value_shape,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, value, shape = dtype_value_shape
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="reshape",
        input=np.asarray(value, dtype=input_dtype),
        shape=shape,
    )
    
    
# stack
@given(
    dtype_value_shape=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(
                set(ivy_torch.valid_float_dtypes)),
        ),
        num_arrays=st.shared(helpers.ints(min_value=1, max_value=4), key="num_arrays"),
        shape=helpers.get_shape(min_num_dims=1),
        ret_shape=True,
    ),
    as_variable=helpers.array_bools(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.stack"
    ),
    native_array=helpers.array_bools(),
    with_out=st.booleans(),
)
def test_torch_stack(
    dtype_value_shape,
    as_variable,
    num_positional_args,
    native_array,
    with_out,
    fw,
):
    input_dtype, value, shape = dtype_value_shape
    tensors = [np.asarray(x, dtype=dtype) for x, dtype in zip(value, input_dtype)]
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="stack",
        tensors=tensors,
        dim=shape,
        out=None,
    )
