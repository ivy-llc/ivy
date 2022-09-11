# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
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
    input_dtypes = draw(
        helpers.array_dtypes(available_dtypes=draw(helpers.get_dtypes("float")))
    )
    for ud, dt in zip(unique_dims, input_dtypes):
        x = draw(
            helpers.array_values(
                shape=common_shape[:unique_idx] + [ud] + common_shape[unique_idx:],
                dtype=dt,
            )
        )
        xs.append(x)
    return xs, input_dtypes, unique_idx


# noinspection DuplicatedCode
@st.composite
def _array_idxes_n_dtype(draw, **kwargs):
    num_dims = draw(helpers.ints(min_value=1, max_value=4))
    dtype, x = draw(
        helpers.dtype_and_values(
            **kwargs, min_num_dims=num_dims, max_num_dims=num_dims, shared_dtype=True
        )
    )
    idxes = draw(
        st.lists(
            helpers.ints(min_value=0, max_value=num_dims - 1),
            min_size=num_dims,
            max_size=num_dims,
            unique=True,
        )
    )
    return x, idxes, dtype


# cat
@handle_cmd_line_args
@given(
    xs_n_input_dtypes_n_unique_idx=_arrays_idx_n_dtypes(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.cat"
    ),
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
@handle_cmd_line_args
@given(
    xs_n_input_dtypes_n_unique_idx=_arrays_idx_n_dtypes(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.concat"
    ),
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


# nonzero
@handle_cmd_line_args
@given(
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
    ),
    as_tuple=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.nonzero"
    ),
)
def test_torch_nonzero(
    *,
    dtype_and_values,
    as_tuple,
    as_variable,
    with_out,
    native_array,
    num_positional_args,
    fw,
):
    dtype, input = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="nonzero",
        input=np.asarray(input, dtype=dtype),
        as_tuple=as_tuple,
    )


# permute
@handle_cmd_line_args
@given(
    dtype_values_axis=_array_idxes_n_dtype(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.permute"
    ),
)
def test_torch_permute(
    dtype_values_axis,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    x, idxes, dtype = dtype_values_axis
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="permute",
        input=np.asarray(x, dtype=dtype),
        dims=tuple(idxes),
    )


# swapdims
@handle_cmd_line_args
@given(
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"),
    ),
    dim0=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"),
    ).filter(lambda axis: isinstance(axis, int)),
    dim1=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"),
    ).filter(lambda axis: isinstance(axis, int)),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.swapdims"
    ),
)
def test_torch_swapdims(
    dtype_and_values,
    dim0,
    dim1,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, value = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
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
@st.composite
def dtypes_x_reshape(draw):
    dtypes, x = draw(
        helpers.dtype_and_values(
            shape=helpers.get_shape(
                allow_none=False,
                min_num_dims=1,
                max_num_dims=5,
                min_dim_size=1,
                max_dim_size=10,
            )
        )
    )
    shape = draw(helpers.reshape_shapes(shape=np.array(x).shape))
    return dtypes, x, shape


@handle_cmd_line_args
@given(
    dtypes_x_reshape=dtypes_x_reshape(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.reshape"
    ),
)
def test_torch_reshape(
    dtypes_x_reshape,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x, shape = dtypes_x_reshape
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="reshape",
        input=np.asarray(x, dtype=input_dtype),
        shape=shape,
    )


# stack
@handle_cmd_line_args
@given(
    dtype_value_shape=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=st.shared(helpers.ints(min_value=2, max_value=4), key="num_arrays"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
    dim=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ).filter(lambda axis: isinstance(axis, int)),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.stack"
    ),
)
def test_torch_stack(
    dtype_value_shape,
    dim,
    as_variable,
    num_positional_args,
    native_array,
    with_out,
    fw,
):
    input_dtype, value = dtype_value_shape
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
        dim=dim,
    )


# transpose
@handle_cmd_line_args
@given(
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"),
    ),
    dim0=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"),
    ).filter(lambda axis: isinstance(axis, int)),
    dim1=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"),
    ).filter(lambda axis: isinstance(axis, int)),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.transpose"
    ),
)
def test_torch_transpose(
    dtype_and_values,
    dim0,
    dim1,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, value = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="transpose",
        input=np.asarray(value, dtype=input_dtype),
        dim0=dim0,
        dim1=dim1,
    )


# squeeze
@handle_cmd_line_args
@given(
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
    dim=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
        max_size=1,
    ).filter(lambda axis: isinstance(axis, int)),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.squeeze"
    ),
)
def test_torch_squeeze(
    dtype_and_values,
    dim,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, value = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="squeeze",
        input=np.asarray(value, dtype=input_dtype),
        dim=dim,
    )


# swapaxes
@handle_cmd_line_args
@given(
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"),
    ),
    axis0=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"),
    ).filter(lambda axis: isinstance(axis, int)),
    axis1=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"),
    ).filter(lambda axis: isinstance(axis, int)),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.swapaxes"
    ),
)
def test_torch_swapaxes(
    dtype_and_values,
    axis0,
    axis1,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, value = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="swapaxes",
        input=np.asarray(value, dtype=input_dtype),
        axis0=axis0,
        axis1=axis1,
    )


# chunk
@handle_cmd_line_args
@given(
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=2,
        max_dim_size=5,
    ),
    chunks=helpers.ints(min_value=2, max_value=5),
    dim=helpers.ints(min_value=0, max_value=1),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.chunk"
    ),
)
def test_torch_chunk(
    dtype_value,
    chunks,
    dim,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, value = dtype_value
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_tree="chunk",
        input=np.asarray(value, dtype=input_dtype),
        chunks=chunks,
        dim=dim,
    )
