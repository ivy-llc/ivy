# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


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
@handle_frontend_test(
    fn_tree="torch.cat",
    xs_n_input_dtypes_n_unique_idx=_arrays_idx_n_dtypes(),
)
def test_torch_cat(
    *,
    xs_n_input_dtypes_n_unique_idx,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    xs, input_dtypes, unique_idx = xs_n_input_dtypes_n_unique_idx
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        tensors=xs,
        dim=unique_idx,
    )


# concat
@handle_frontend_test(
    fn_tree="torch.concat",
    xs_n_input_dtypes_n_unique_idx=_arrays_idx_n_dtypes(),
)
def test_torch_concat(
    *,
    xs_n_input_dtypes_n_unique_idx,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    xs, input_dtypes, unique_idx = xs_n_input_dtypes_n_unique_idx
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        tensors=xs,
        dim=unique_idx,
    )


# gather
@handle_frontend_test(
    fn_tree="torch.gather",
    params_indices_others=helpers.array_indices_axis(
        array_dtypes=helpers.get_dtypes("valid"),
        indices_dtypes=["int64"],
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
        indices_same_dims=True,
    ),
)
def test_torch_gather(
    *,
    params_indices_others,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtypes, input, indices, axis, batch_dims = params_indices_others
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input,
        dim=axis,
        index=indices,
    )


# nonzero
@handle_frontend_test(
    fn_tree="torch.nonzero",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
    as_tuple=st.booleans(),
)
def test_torch_nonzero(
    *,
    dtype_and_values,
    as_tuple,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtype, input = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input[0],
        as_tuple=as_tuple,
    )


# permute
@handle_frontend_test(
    fn_tree="torch.permute",
    dtype_values_axis=_array_idxes_n_dtype(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_permute(
    *,
    dtype_values_axis,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    x, idxes, dtype = dtype_values_axis
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        dims=tuple(idxes),
    )


# swapdims
@handle_frontend_test(
    fn_tree="torch.swapdims",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"),
    ),
    dim0=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"),
        force_int=True,
    ),
    dim1=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"),
        force_int=True,
    ),
)
def test_torch_swapdims(
    *,
    dtype_and_values,
    dim0,
    dim1,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, value = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=value[0],
        dim0=dim0,
        dim1=dim1,
    )


# reshape
@st.composite
def dtypes_x_reshape(draw):
    shape = draw(
        helpers.get_shape(
            allow_none=False,
            min_num_dims=1,
            max_num_dims=3,
            min_dim_size=1,
            max_dim_size=3,
        )
    )
    dtypes, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
            shape=shape,
        )
    )
    shape = draw(helpers.reshape_shapes(shape=shape))
    return dtypes, x, shape


@handle_frontend_test(
    fn_tree="torch.reshape",
    dtypes_x_reshape=dtypes_x_reshape(),
)
def test_torch_reshape(
    *,
    dtypes_x_reshape,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x, shape = dtypes_x_reshape
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        shape=shape,
    )


# stack
@handle_frontend_test(
    fn_tree="torch.stack",
    dtype_value_shape=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=st.shared(helpers.ints(min_value=2, max_value=4), key="num_arrays"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
    dim=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ).filter(lambda axis: isinstance(axis, int)),
)
def test_torch_stack(
    *,
    dtype_value_shape,
    dim,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, value = dtype_value_shape
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        tensors=value,
        dim=dim,
    )


# transpose
@handle_frontend_test(
    fn_tree="torch.transpose",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"),
    ),
    dim0=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"),
        force_int=True,
    ),
    dim1=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"),
        force_int=True,
    ),
)
def test_torch_transpose(
    *,
    dtype_and_values,
    dim0,
    dim1,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, value = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=value[0],
        dim0=dim0,
        dim1=dim1,
    )


# squeeze
@handle_frontend_test(
    fn_tree="torch.squeeze",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
    dim=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
        max_size=1,
    ).filter(lambda axis: isinstance(axis, int)),
)
def test_torch_squeeze(
    *,
    dtype_and_values,
    dim,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, value = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=value[0],
        dim=dim,
    )


# swapaxes
@handle_frontend_test(
    fn_tree="torch.swapaxes",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"),
    ),
    axis0=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"),
        force_int=True,
    ),
    axis1=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"),
        force_int=True,
    ),
)
def test_torch_swapaxes(
    *,
    dtype_and_values,
    axis0,
    axis1,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, value = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=value[0],
        axis0=axis0,
        axis1=axis1,
    )


# chunk
@handle_frontend_test(
    fn_tree="torch.chunk",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=2,
        max_num_dims=4,
        min_dim_size=2,
        max_dim_size=4,
    ),
    chunks=helpers.ints(min_value=1, max_value=3),
    dim=helpers.ints(min_value=0, max_value=1),
)
def test_torch_chunk(
    *,
    dtype_value,
    chunks,
    dim,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, value = dtype_value
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=value[0],
        chunks=chunks,
        dim=dim,
    )


# tile
@handle_frontend_test(
    fn_tree="torch.tile",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(), key="shape"),
    ),
    dim=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="shape"),
        allow_neg=False,
        force_tuple=True,
    ),
)
def test_torch_tile(
    *,
    dtype_value,
    dim,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, value = dtype_value
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=value[0],
        dims=dim,
    )


# unsqueeze
@handle_frontend_test(
    fn_tree="torch.unsqueeze",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(), key="shape"),
    ),
    dim=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="shape"),
        allow_neg=True,
        force_int=True,
    ),
)
def test_torch_unsqueeze(
    *,
    dtype_value,
    dim,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, value = dtype_value
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=value[0],
        dim=dim,
    )


# movedim
@handle_frontend_test(
    fn_tree="torch.movedim",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-100,
        max_value=100,
        shape=st.shared(
            helpers.get_shape(
                min_num_dims=1,
                max_num_dims=3,
                min_dim_size=1,
                max_dim_size=3,
            ),
            key="a_s_d",
        ),
    ),
    source=helpers.get_axis(
        allow_none=False,
        unique=True,
        shape=st.shared(
            helpers.get_shape(
                min_num_dims=1,
                max_num_dims=3,
                min_dim_size=1,
                max_dim_size=3,
            ),
            key="a_s_d",
        ),
        min_size=1,
        force_int=True,
    ),
    destination=helpers.get_axis(
        allow_none=False,
        unique=True,
        shape=st.shared(
            helpers.get_shape(
                min_num_dims=1,
                max_num_dims=3,
                min_dim_size=1,
                max_dim_size=3,
            ),
            key="a_s_d",
        ),
        min_size=1,
        force_int=True,
    ),
)
def test_torch_movedim(
    *,
    dtype_and_input,
    source,
    destination,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, value = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=value[0],
        source=source,
        destination=destination,
    )


# hstack
@handle_frontend_test(
    fn_tree="torch.hstack",
    dtype_value_shape=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_hstack(
    *,
    dtype_value_shape,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, value = dtype_value_shape
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        tensors=value,
    )


# dstack
@handle_frontend_test(
    fn_tree="torch.dstack",
    dtype_value_shape=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_dstack(
    *,
    dtype_value_shape,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, value = dtype_value_shape
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        tensors=value,
    )


# index_select
@handle_frontend_test(
    fn_tree="torch.index_select",
    params_indices_others=helpers.array_indices_axis(
        array_dtypes=helpers.get_dtypes("valid"),
        indices_dtypes=["int64"],
        max_num_dims=1,
        indices_same_dims=True,
    ),
)
def test_torch_index_select(
    *,
    params_indices_others,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtypes, input, indices, axis, batch_dims = params_indices_others
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input,
        dim=axis,
        index=indices,
    )


# take_along_dim
@handle_frontend_test(
    fn_tree="torch.take_along_dim",
    dtype_indices_axis=helpers.array_indices_axis(
        array_dtypes=helpers.get_dtypes("numeric"),
        indices_dtypes=["int64"],
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
        indices_same_dims=True,
    ),
)
def test_torch_take_along_dim(
    *,
    dtype_indices_axis,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtypes, value, indices, axis, _ = dtype_indices_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=value,
        indices=indices,
        dim=axis,
    )
