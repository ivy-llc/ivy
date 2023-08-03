# global
from hypothesis import strategies as st
import math


# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_functional.test_core.test_manipulation import _get_splits
from ivy_tests.test_ivy.test_functional.test_core.test_searching import (
    _broadcastable_trio,
)
from ivy_tests.test_ivy.test_functional.test_core.test_manipulation import (  # noqa
    _get_splits,
)


# noinspection DuplicatedCode
@st.composite
def _arrays_idx_n_dtypes(draw):
    num_dims = draw(st.shared(helpers.ints(min_value=1, max_value=4), key="num_dims"))
    num_arrays = draw(
        st.shared(helpers.ints(min_value=2, max_value=4), key="num_arrays")
    )
    common_shape = draw(
        helpers.list_of_size(
            x=helpers.ints(min_value=2, max_value=3),
            size=num_dims - 1,
        )
    )
    unique_idx = draw(helpers.ints(min_value=0, max_value=num_dims - 1))
    unique_dims = draw(
        helpers.list_of_size(
            x=helpers.ints(min_value=2, max_value=3),
            size=num_arrays,
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


# adjoint
@handle_frontend_test(
    fn_tree="torch.adjoint",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("real_and_complex"),
        min_num_dims=2,
        min_dim_size=2,
    ),
)
def test_torch_adjoint(
    *,
    dtype_and_values,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, value = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=value[0],
    )


# cat
@handle_frontend_test(
    fn_tree="torch.cat",
    xs_n_input_dtypes_n_unique_idx=_arrays_idx_n_dtypes(),
)
def test_torch_cat(
    *,
    xs_n_input_dtypes_n_unique_idx,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    xs, input_dtypes, unique_idx = xs_n_input_dtypes_n_unique_idx
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
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
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    xs, input_dtypes, unique_idx = xs_n_input_dtypes_n_unique_idx
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
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
        indices_same_dims=True,
    ),
)
def test_torch_gather(
    *,
    params_indices_others,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtypes, input, indices, axis, batch_dims = params_indices_others
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
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
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    dtype, input = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
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
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    x, idxes, dtype = dtype_values_axis
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
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
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, value = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=value[0],
        dim0=dim0,
        dim1=dim1,
    )


# reshape
@st.composite
def dtypes_x_reshape(draw):
    shape = draw(helpers.get_shape(min_num_dims=1))
    dtypes, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
            shape=shape,
        )
    )
    shape = draw(
        helpers.get_shape(min_num_dims=1).filter(
            lambda s: math.prod(s) == math.prod(shape)
        )
    )
    return dtypes, x, shape


@handle_frontend_test(
    fn_tree="torch.reshape",
    dtypes_x_reshape=dtypes_x_reshape(),
)
def test_torch_reshape(
    *,
    dtypes_x_reshape,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x, shape = dtypes_x_reshape
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
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
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, value = dtype_value_shape
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
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
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, value = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=value[0],
        dim0=dim0,
        dim1=dim1,
    )


# t
@handle_frontend_test(
    fn_tree="torch.t",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=st.shared(helpers.get_shape(max_num_dims=2), key="shape"),
    ),
)
def test_torch_t(
    *,
    dtype_and_values,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, value = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=value[0],
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
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, value = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
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
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, value = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=value[0],
        axis0=axis0,
        axis1=axis1,
    )


@st.composite
def _chunk_helper(draw):
    dtype, x, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
            min_num_dims=1,
            ret_shape=True,
        )
    )
    axis = draw(helpers.get_axis(shape=shape, force_int=True))
    if shape[axis] == 0:
        chunks = 0
    else:
        factors = []
        for i in range(1, shape[axis] + 1):
            if shape[axis] % i == 0:
                factors.append(i)
        chunks = draw(st.sampled_from(factors))
    return dtype, x, axis, chunks


# chunk
@handle_frontend_test(
    fn_tree="torch.chunk",
    x_dim_chunks=_chunk_helper(),
    test_with_out=st.just(False),
)
def test_torch_chunk(
    *,
    x_dim_chunks,
    fn_tree,
    on_device,
    frontend,
    test_flags,
):
    dtype, x, axis, chunks = x_dim_chunks
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        chunks=chunks,
        dim=axis,
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
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, value = dtype_value
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
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
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, value = dtype_value
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=value[0],
        dim=dim,
    )


@handle_frontend_test(
    fn_tree="torch.argwhere",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_torch_argwhere(
    *,
    dtype_and_values,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    dtype, input = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input[0],
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
    test_with_out=st.just(False),
)
def test_torch_movedim(
    *,
    dtype_and_input,
    source,
    destination,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, value = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=value[0],
        source=source,
        destination=destination,
    )


# moveaxis
@handle_frontend_test(
    fn_tree="torch.moveaxis",
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
    test_with_out=st.just(False),
)
def test_torch_moveaxis(
    *,
    dtype_and_input,
    source,
    destination,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, value = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
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
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, value = dtype_value_shape
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
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
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, value = dtype_value_shape
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
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
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtypes, input, indices, axis, batch_dims = params_indices_others
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
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
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtypes, value, indices, axis, _ = dtype_indices_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=value,
        indices=indices,
        dim=axis,
    )


# vstack
@handle_frontend_test(
    fn_tree="torch.vstack",
    dtype_value_shape=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_vstack(
    *,
    dtype_value_shape,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, value = dtype_value_shape
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        tensors=value,
    )


# split
@handle_frontend_test(
    fn_tree="torch.split",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
    ),
    split_size_or_sections=_get_splits(
        allow_none=False, min_num_dims=1, allow_array_indices=False
    ),
    dim=st.shared(
        helpers.get_axis(
            shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
            force_int=True,
        ),
        key="target_axis",
    ),
)
def test_torch_split(
    *,
    dtype_value,
    split_size_or_sections,
    dim,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, value = dtype_value
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        tensor=value[0],
        split_size_or_sections=split_size_or_sections,
        dim=dim,
    )


# tensor_split
@handle_frontend_test(
    fn_tree="torch.tensor_split",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
    ),
    indices_or_sections=_get_splits(
        min_num_dims=1, allow_none=False, allow_array_indices=False
    ),
    axis=st.shared(
        helpers.get_axis(
            shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
            force_int=True,
        ),
        key="target_axis",
    ),
    number_positional_args=st.just(2),
    test_with_out=st.just(False),
)
def test_torch_tensor_split(
    *,
    dtype_value,
    indices_or_sections,
    axis,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, value = dtype_value
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=value[0],
        indices_or_sections=indices_or_sections,
        dim=axis,
    )


# unbind
@handle_frontend_test(
    fn_tree="torch.unbind",
    dtype_value_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
    ),
)
def test_torch_unbind(
    *,
    dtype_value_axis,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtypes, value, axis = dtype_value_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=value[0],
        dim=axis,
    )


# dsplit
@handle_frontend_test(
    fn_tree="torch.dsplit",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=3), key="value_shape"),
    ),
    indices_or_sections=_get_splits(
        min_num_dims=3,
        axis=2,
        allow_none=False,
        allow_array_indices=False,
        is_mod_split=True,
    ),
)
def test_torch_dsplit(
    *,
    dtype_value,
    indices_or_sections,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, value = dtype_value
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=value[0],
        indices_or_sections=indices_or_sections,
    )


# hsplit
@handle_frontend_test(
    fn_tree="torch.hsplit",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="value_shape"),
    ),
    indices_or_sections=_get_splits(
        min_num_dims=1,
        axis=1,
        allow_none=False,
        allow_array_indices=False,
        is_mod_split=True,
    ),
)
def test_torch_hsplit(
    *,
    dtype_value,
    indices_or_sections,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, value = dtype_value
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=value[0],
        indices_or_sections=indices_or_sections,
    )


# vsplit
@handle_frontend_test(
    fn_tree="torch.vsplit",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="value_shape"),
    ),
    indices_or_sections=_get_splits(
        min_num_dims=2,
        axis=0,
        allow_none=False,
        allow_array_indices=False,
        is_mod_split=True,
    ),
)
def test_torch_vsplit(
    *,
    dtype_value,
    indices_or_sections,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, value = dtype_value
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=value[0],
        indices_or_sections=indices_or_sections,
    )


# row_stack
@handle_frontend_test(
    fn_tree="torch.row_stack",
    dtype_value_shape=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=st.integers(1, 5),
    ),
)
def test_torch_row_stack(
    *,
    dtype_value_shape,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, value = dtype_value_shape
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        tensors=value,
    )


@handle_frontend_test(
    fn_tree="torch.where",
    broadcastables=_broadcastable_trio(),
    only_cond=st.booleans(),
)
def test_torch_where(
    *,
    broadcastables,
    only_cond,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    cond, xs, dtypes = broadcastables

    if only_cond:
        helpers.test_frontend_function(
            input_dtypes=[dtypes[0]],
            frontend=frontend,
            test_flags=test_flags,
            fn_tree=fn_tree,
            on_device=on_device,
            condition=xs[0],
        )

    else:
        helpers.test_frontend_function(
            input_dtypes=["bool"] + dtypes,
            frontend=frontend,
            test_flags=test_flags,
            fn_tree=fn_tree,
            on_device=on_device,
            condition=cond,
            input=xs[0],
            other=xs[1],
        )


@handle_frontend_test(
    fn_tree="torch.conj",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_complex"),
    ),
)
def test_torch_conj(
    on_device,
    frontend,
    *,
    dtype_and_x,
    fn_tree,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
    )


@st.composite
def _arrays_dim_idx_n_dtypes(draw):
    num_dims = draw(st.shared(helpers.ints(min_value=1, max_value=4), key="num_dims"))
    num_arrays = 2
    common_shape = draw(
        helpers.lists(
            x=helpers.ints(min_value=2, max_value=3),
            min_size=num_dims - 1,
            max_size=num_dims - 1,
        )
    )
    _dim = draw(helpers.ints(min_value=0, max_value=num_dims - 1))
    unique_dims = draw(
        helpers.lists(
            x=helpers.ints(min_value=2, max_value=3),
            min_size=num_arrays,
            max_size=num_arrays,
        )
    )

    min_dim = min(unique_dims)
    max_dim = max(unique_dims)
    _idx = draw(
        helpers.array_values(
            shape=min_dim,
            dtype="int64",
            min_value=0,
            max_value=max_dim,
            exclude_min=False,
        )
    )

    xs = list()
    available_input_types = draw(helpers.get_dtypes("numeric"))
    available_input_types.remove("float16")  # half summation unstable in backends
    input_dtypes = draw(
        helpers.array_dtypes(
            available_dtypes=available_input_types,
            num_arrays=num_arrays,
            shared_dtype=True,
        )
    )
    for ud, dt in zip(unique_dims, input_dtypes):
        x = draw(
            helpers.array_values(
                shape=common_shape[:_dim] + [ud] + common_shape[_dim:],
                dtype=dt,
                large_abs_safety_factor=2.5,
                small_abs_safety_factor=2.5,
                safety_factor_scale="log",
            )
        )
        xs.append(x)
    return xs, input_dtypes, _dim, _idx


# index_add
@handle_frontend_test(
    fn_tree="torch.index_add",
    xs_dtypes_dim_idx=_arrays_dim_idx_n_dtypes(),
    alpha=st.integers(min_value=1, max_value=2),
)
def test_torch_index_add(
    *,
    xs_dtypes_dim_idx,
    alpha,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    xs, input_dtypes, axis, indices = xs_dtypes_dim_idx
    if xs[0].shape[axis] < xs[1].shape[axis]:
        source, input = xs
    else:
        input, source = xs
    helpers.test_frontend_function(
        input_dtypes=[input_dtypes[0], "int64", input_dtypes[1]],
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-03,
        input=input,
        dim=axis,
        index=indices,
        source=source,
        alpha=alpha,
    )


# index_copy
@handle_frontend_test(
    fn_tree="torch.index_copy",
    xs_dtypes_dim_idx=_arrays_dim_idx_n_dtypes(),
)
def test_torch_index_copy(
    *,
    xs_dtypes_dim_idx,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    xs, input_dtypes, axis, indices = xs_dtypes_dim_idx
    if xs[0].shape[axis] < xs[1].shape[axis]:
        source, input = xs
    else:
        input, source = xs
    helpers.test_frontend_function(
        input_dtypes=[input_dtypes[0], "int64", input_dtypes[1]],
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input,
        dim=axis,
        index=indices,
        source=source,
    )


@st.composite
def _dtypes_input_mask(draw):
    _shape = draw(helpers.get_shape(min_num_dims=1, min_dim_size=1))
    _mask = draw(helpers.array_values(dtype=helpers.get_dtypes("bool"), shape=_shape))
    _dtype, _x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
            num_arrays=1,
            shape=_shape,
        )
    )

    return _dtype, _x, _mask


@handle_frontend_test(
    fn_tree="torch.masked_select",
    dtype_input_mask=_dtypes_input_mask(),
)
def test_torch_masked_select(
    *,
    dtype_input_mask,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    (
        input_dtype,
        x,
        mask,
    ) = dtype_input_mask

    helpers.test_frontend_function(
        input_dtypes=input_dtype + ["bool"],
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        mask=mask,
    )


@handle_frontend_test(
    fn_tree="torch.take",
    dtype_and_x=helpers.array_indices_axis(
        array_dtypes=helpers.get_dtypes(), indices_dtypes=["int64"]
    ),
)
def test_torch_take(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    dtypes, xs, indices, _, _ = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=xs,
        index=indices,
    )


@st.composite
def _dtype_input_dim_start_length(draw):
    _shape = draw(helpers.get_shape(min_num_dims=1, min_dim_size=1))
    _dtype, _x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
            num_arrays=1,
            shape=_shape,
        )
    )
    _dim = draw(
        helpers.get_axis(
            shape=_shape,
            force_int=True,
        ),
    )
    _start = draw(helpers.ints(min_value=1, max_value=_shape[_dim]))

    _length = draw(helpers.ints(min_value=0, max_value=_shape[_dim] - _start))

    return _dtype, _x, _dim, _start, _length


@handle_frontend_test(
    fn_tree="torch.narrow",
    dtype_input_dim_start_length=_dtype_input_dim_start_length(),
)
def test_torch_narrow(
    *,
    dtype_input_dim_start_length,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    (input_dtype, x, dim, start, length) = dtype_input_dim_start_length

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        dim=dim,
        start=start,
        length=length,
    )


@st.composite
def _dtype_input_idx_axis(draw):
    dtype_x_axis_shape = draw(
        helpers.dtype_values_axis(
            available_dtypes=helpers.get_dtypes("valid"),
            force_int_axis=True,
            ret_shape=True,
            valid_axis=True,
            min_num_dims=2,
        )
    )

    input_dtype, x, axis, shape = dtype_x_axis_shape
    max_idx = 0
    if shape:
        max_idx = shape[axis] - 1
    idx = draw(helpers.ints(min_value=0, max_value=max_idx))
    x = x[0]

    return input_dtype, x, idx, axis


@handle_frontend_test(
    fn_tree="torch.select",
    dtype_x_idx_axis=_dtype_input_idx_axis(),
)
def test_torch_select(
    *,
    dtype_x_idx_axis,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x, idx, axis = dtype_x_idx_axis

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x,
        dim=axis,
        index=idx,
    )
