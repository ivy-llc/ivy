# global
import math

import numpy as np
from hypothesis import assume, strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# helpers
@st.composite
def _get_repeat_interleaves_args(
    draw, *, available_dtypes, valid_axis, max_num_dims, max_dim_size
):
    values_dtype, values, axis, shape = draw(
        helpers.dtype_values_axis(
            available_dtypes=available_dtypes,
            valid_axis=valid_axis,
            force_int_axis=True,
            shape=draw(
                helpers.get_shape(
                    allow_none=False,
                    min_num_dims=0,
                    max_num_dims=max_num_dims,
                    min_dim_size=1,
                    max_dim_size=max_dim_size,
                )
            ),
            ret_shape=True,
        )
    )

    if axis is None:
        generate_repeats_as_integer = draw(st.booleans())
        num_repeats = 1 if generate_repeats_as_integer else math.prod(tuple(shape))
    else:
        num_repeats = shape[axis]

    repeats_dtype, repeats = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("integer"),
            min_value=0,
            max_value=10,
            shape=[num_repeats],
        )
    )

    # Output size is an optional parameter accepted by Torch for optimisation
    use_output_size = draw(st.booleans())
    output_size = np.sum(repeats) if use_output_size else None

    return [values_dtype, repeats_dtype], values, repeats, axis, output_size


# flip
@handle_frontend_test(
    fn_tree="torch.flip",
    dtype_and_values=helpers.dtype_and_values(
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
        available_dtypes=helpers.get_dtypes("float"),
    ),
    axis=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
        force_tuple=True,
    ),
)
def test_torch_flip(
    *,
    dtype_and_values,
    axis,
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
        dims=axis,
    )


# roll
@handle_frontend_test(
    fn_tree="torch.roll",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
    shift=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
        force_tuple=True,
    ),
    axis=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
        force_tuple=True,
    ),
)
def test_torch_roll(
    *,
    dtype_and_values,
    shift,
    axis,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, value = dtype_and_values
    if isinstance(shift, int) and isinstance(axis, tuple):
        axis = axis[0]
    if isinstance(shift, tuple) and isinstance(axis, tuple):
        if len(shift) != len(axis):
            mn = min(len(shift), len(axis))
            shift = shift[:mn]
            axis = axis[:mn]
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
        shifts=shift,
        dims=axis,
    )


# fliplr
@handle_frontend_test(
    fn_tree="torch.fliplr",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=helpers.get_shape(min_num_dims=2),
    ),
)
def test_torch_fliplr(
    *,
    dtype_and_values,
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
    )


# cumsum
@handle_frontend_test(
    fn_tree="torch.cumsum",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=5,
        valid_axis=True,
        allow_neg_axes=False,
        max_axes_size=1,
        force_int_axis=True,
    ),
    dtype=helpers.get_dtypes("numeric", none=True, full=False),
)
def test_torch_cumsum(
    *,
    dtype_x_axis,
    dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x, axis = dtype_x_axis
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
        dim=axis,
        dtype=dtype[0],
    )


@st.composite
def dims_and_offset(draw, shape):
    shape_actual = draw(shape)
    dim1 = draw(helpers.get_axis(shape=shape, force_int=True))
    dim2 = draw(helpers.get_axis(shape=shape, force_int=True))
    offset = draw(
        st.integers(min_value=-shape_actual[dim1], max_value=shape_actual[dim1])
    )
    return dim1, dim2, offset


@handle_frontend_test(
    fn_tree="torch.diagonal",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"),
    ),
    dims_and_offset=dims_and_offset(
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape")
    ),
)
def test_torch_diagonal(
    *,
    dtype_and_values,
    dims_and_offset,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, value = dtype_and_values
    dim1, dim2, offset = dims_and_offset
    input = value[0]
    num_dims = len(np.shape(input))
    assume(dim1 != dim2)
    if dim1 < 0:
        assume(dim1 + num_dims != dim2)
    if dim2 < 0:
        assume(dim1 != dim2 + num_dims)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        all_aliases=["diagonal"],
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input,
        offset=offset,
        dim1=dim1,
        dim2=dim2,
    )


@handle_frontend_test(
    fn_tree="torch.cartesian_prod",
    dtype_and_tensors=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=st.integers(min_value=1, max_value=5),
        min_num_dims=1,
        max_num_dims=1,
        max_dim_size=5,
        shared_dtype=True,
    ),
)
def test_torch_cartesian_prod(
    *,
    dtype_and_tensors,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtypes, tensors = dtype_and_tensors
    if isinstance(dtypes, list):  # If more than one value was generated
        args = {
            f"x{i}": np.array(tensor, dtype=dtypes[i])
            for i, tensor in enumerate(tensors)
        }
    else:  # If exactly one value was generated
        args = {"x0": np.array(tensors, dtype=dtypes)}
    num_positional_args = len(tensors)
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        **args,
    )


@handle_frontend_test(
    fn_tree="torch.triu",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=2,  # Torch requires this.
    ),
    diagonal=st.integers(min_value=-100, max_value=100),
)
def test_torch_triu(
    *,
    dtype_and_values,
    diagonal,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtype, values = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=values[0],
        diagonal=diagonal,
    )


# cumprod
@handle_frontend_test(
    fn_tree="torch.cumprod",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=5,
        min_value=-100,
        max_value=100,
        valid_axis=True,
        allow_neg_axes=False,
        max_axes_size=1,
        force_int_axis=True,
    ),
    dtype=helpers.get_dtypes("numeric", none=True, full=False),
)
def test_torch_cumprod(
    *,
    dtype_x_axis,
    dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x, axis = dtype_x_axis
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
        dim=axis,
        dtype=dtype[0],
    )


# trace
@handle_frontend_test(
    fn_tree="torch.trace",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        shape=st.shared(helpers.get_shape(min_num_dims=2, max_num_dims=2), key="shape"),
    ),
)
def test_torch_trace(
    *,
    dtype_and_values,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtype, value = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=value[0],
    )


# tril_indices
@handle_frontend_test(
    fn_tree="torch.tril_indices",
    row=st.integers(min_value=0, max_value=10),
    col=st.integers(min_value=0, max_value=10),
    offset=st.integers(),
    dtype=helpers.get_dtypes("valid", none=True, full=False),
)
def test_torch_tril_indices(
    *,
    row,
    col,
    offset,
    dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    helpers.test_frontend_function(
        input_dtypes=[ivy.int32],
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        row=row,
        col=col,
        offset=offset,
        dtype=dtype[0],
    )


@handle_frontend_test(
    fn_tree="torch.triu_indices",
    row=st.integers(min_value=0, max_value=100),
    col=st.integers(min_value=0, max_value=100),
    offset=st.integers(),
)
def test_torch_triu_indices(
    *,
    row,
    col,
    offset,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    helpers.test_frontend_function(
        input_dtypes=["int32"],
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        row=row,
        col=col,
        offset=offset,
    )


# tril
@handle_frontend_test(
    fn_tree="torch.tril",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,  # Torch requires this.
    ),
    diagonal=st.integers(min_value=-100, max_value=100),
)
def test_torch_tril(
    *,
    dtype_and_values,
    diagonal,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtype, values = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=values[0],
        diagonal=diagonal,
    )


@st.composite
def _get_dtype_and_arrays_and_start_end_dim(
    draw,
    *,
    available_dtypes,
    min_num_dims=1,
    max_num_dims=5,
    min_dim_size=1,
    max_dim_size=5,
):
    """Samples a dtype, array, and start and end dimension which are within the array,
    with the caveat that the end dimension can be `-1`. This is to match the API
    for PyTorch's flatten.

    Parameters
    ----------
    available_dtypes
        The dtypes that are permitted for the array, expected to be
        `helpers.get_dtypes("valid") or similar.

    min_num_dims
        The minimum number of dimensions the array can have. Defaults to 1

    max_num_dims
        The maximum number of dimensions the array can have. Defaults to 5

    min_dim_size
        The minimum size of any dimension in the array. Defaults to 1

    max_dim_size
        The maximum size of any dimension in the array. Defaults to 5

    Returns
    -------
    ret
        A 4-tuple (dtype, array, start_dim, end_dim) where dtype is
        one of the available dtypes, the array is an array of values
        and start_dim and end_dim are legal dimensions contained
        within the array, with either start_dim <= end_dim or
        end_dim = 1.

    """
    num_dims = draw(st.integers(min_value=min_num_dims, max_value=max_num_dims))
    shape = tuple(
        draw(st.integers(min_value=min_dim_size, max_value=max_dim_size))
        for _ in range(num_dims)
    )

    dtype, array = draw(
        helpers.dtype_and_values(
            available_dtypes=available_dtypes,
            shape=shape,
        )
    )

    start_dim = draw(st.integers(min_value=0, max_value=num_dims - 1))

    # End_dim must be either -1 or in [start_dim, num_dims)
    # If end_dim is -1, then it's going to flatten to a 1-D array.
    is_full_flatten = draw(st.booleans())
    if is_full_flatten:
        end_dim = -1
    else:
        end_dim = draw(st.integers(min_value=start_dim, max_value=num_dims - 1))

    return dtype, array, start_dim, end_dim


@handle_frontend_test(
    fn_tree="torch.flatten",
    dtype_and_input_and_start_end_dim=_get_dtype_and_arrays_and_start_end_dim(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_torch_flatten(
    *,
    dtype_and_input_and_start_end_dim,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtype, input, start_dim, end_dim = dtype_and_input_and_start_end_dim
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
        start_dim=start_dim,
        end_dim=end_dim,
    )


# renorm
@handle_frontend_test(
    fn_tree="torch.renorm",
    dtype_and_values=helpers.dtype_and_values(
        # Min_num_dims is 2 to prevent a Torch crash.
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"),
        # Setting available types to valid allows Bool and integer types
        # which causes a Torch crash.
        available_dtypes=helpers.get_dtypes("float"),
        max_value=1e4,
        min_value=-1e4,
    ),
    dim=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="shape"),
        force_int=True,
    ),
    p=st.floats(
        min_value=0.5,
        exclude_min=True,
        max_value=5,
    ),  # Non-positive norms aren't supported in backends.
    # Small positive norms cause issues due to finite-precision.
    maxnorm=st.floats(min_value=0),  # Norms are positive semi-definite
)
def test_torch_renorm(
    *,
    dtype_and_values,
    p,
    dim,
    maxnorm,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtype, values = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        atol=1e-02,
        input=values[0],
        p=p,
        dim=dim,
        maxnorm=maxnorm,
    )


# logcumsumexp
@handle_frontend_test(
    fn_tree="torch.logcumsumexp",
    dtype_and_input=helpers.dtype_and_values(
        # Torch version is not implemented for Integer or Bool types
        available_dtypes=helpers.get_dtypes("float"),
        shape=st.shared(helpers.get_shape(), key="shape"),
        max_value=100,
        min_value=-100,
    ),
    dim=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="shape"), force_int=True
    ),
)
def test_torch_logcumsumexp(
    *,
    dtype_and_input,
    dim,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-2,
        atol=1e-2,
        input=input[0],
        dim=dim,
    )


# repeat_interleave
@handle_frontend_test(
    fn_tree="torch.repeat_interleave",
    dtype_values_repeats_axis_output_size=_get_repeat_interleaves_args(
        available_dtypes=helpers.get_dtypes("valid"),
        valid_axis=True,
        max_num_dims=4,
        max_dim_size=4,
    ),
)
def test_torch_repeat_interleave(
    *,
    dtype_values_repeats_axis_output_size,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtype, values, repeats, axis, output_size = dtype_values_repeats_axis_output_size

    helpers.test_frontend_function(
        input_dtypes=dtype[0],
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=values[0],
        repeats=repeats[0],
        dim=axis,
        output_size=output_size,
    )


# ravel
@handle_frontend_test(
    fn_tree="torch.ravel",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=1,
    ),
)
def test_torch_ravel(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=np.asarray(x[0], dtype=input_dtype[0]),
    )


# rot90
@handle_frontend_test(
    fn_tree="torch.rot90",
    dtype_and_x=helpers.dtype_and_values(
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"),
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
    dims=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"),
        min_size=2,
        max_size=2,
        unique=True,
        allow_neg=False,
        force_tuple=True,
    ),
    k=st.integers(min_value=-10, max_value=10),
)
def test_torch_rot90(
    *,
    dtype_and_x,
    dims,
    k,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
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
        k=k,
        dims=dims,
    )


# vander
@handle_frontend_test(
    fn_tree="torch.vander",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=st.tuples(
            st.integers(min_value=1, max_value=5),
        ),
    ),
    N=st.integers(min_value=0, max_value=5),
    increasing=st.booleans(),
)
def test_torch_vander(
    *,
    dtype_and_x,
    N,
    increasing,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=np.asarray(x[0], dtype=input_dtype[0]),
        N=N,
        increasing=increasing,
    )


# lcm
@handle_frontend_test(
    fn_tree="torch.lcm",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=True,
        min_num_dims=1,
        max_num_dims=3,
        min_value=-100,
        max_value=100,
        allow_nan=False,
    ),
)
def test_torch_lcm(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
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
        other=x[1],
    )


# einsum
@handle_frontend_test(
    fn_tree="torch.einsum",
    eq_n_op_n_shp=st.sampled_from(
        [
            ("ii", (np.arange(25).reshape(5, 5),), ()),
            ("ii->i", (np.arange(25).reshape(5, 5),), (5,)),
            ("ij,j", (np.arange(25).reshape(5, 5), np.arange(5)), (5,)),
        ]
    ),
    dtype=helpers.get_dtypes("float", full=False),
)
def test_torch_einsum(
    *,
    eq_n_op_n_shp,
    dtype,
    as_variable,
    with_out,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    eq, operands, _ = eq_n_op_n_shp
    kw = {}
    i = 0
    for x_ in operands:
        kw["x{}".format(i)] = x_
        i += 1
    # len(operands) + 1 because of the equation
    num_positional_args = len(operands) + 1
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        equation=eq,
        **kw,
    )


# cross
@st.composite
def dtype_value1_value2_axis(
    draw,
    available_dtypes,
    abs_smallest_val=None,
    min_value=None,
    max_value=None,
    allow_inf=False,
    exclude_min=False,
    exclude_max=False,
    min_num_dims=1,
    max_num_dims=10,
    min_dim_size=1,
    max_dim_size=10,
    specific_dim_size=3,
    large_abs_safety_factor=4,
    small_abs_safety_factor=4,
    safety_factor_scale="log",
):
    # For cross product, a dim with size 3 is required
    shape = draw(
        helpers.get_shape(
            allow_none=False,
            min_num_dims=min_num_dims,
            max_num_dims=max_num_dims,
            min_dim_size=min_dim_size,
            max_dim_size=max_dim_size,
        )
    )
    axis = draw(helpers.ints(min_value=0, max_value=len(shape)))
    # make sure there is a dim with specific dim size
    shape = list(shape)
    shape = shape[:axis] + [specific_dim_size] + shape[axis:]
    shape = tuple(shape)

    dtype = draw(st.sampled_from(draw(available_dtypes)))

    values = []
    for i in range(2):
        values.append(
            draw(
                helpers.array_values(
                    dtype=dtype,
                    shape=shape,
                    abs_smallest_val=abs_smallest_val,
                    min_value=min_value,
                    max_value=max_value,
                    allow_inf=allow_inf,
                    exclude_min=exclude_min,
                    exclude_max=exclude_max,
                    large_abs_safety_factor=large_abs_safety_factor,
                    small_abs_safety_factor=small_abs_safety_factor,
                    safety_factor_scale=safety_factor_scale,
                )
            )
        )

    value1, value2 = values[0], values[1]
    return [dtype], value1, value2, axis


@handle_frontend_test(
    fn_tree="torch.cross",
    dtype_input_other_dim=dtype_value1_value2_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=10,
        min_dim_size=3,
        max_dim_size=3,
        min_value=-1e10,
        max_value=1e10,
        abs_smallest_val=0.01,
        large_abs_safety_factor=2,
        safety_factor_scale="log",
    ),
)
def test_torch_cross(
    dtype_input_other_dim,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
):
    dtype, input, other, dim = dtype_input_other_dim
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        rtol=1e-1,
        atol=1e-2,
        input=input,
        other=other,
        dim=dim,
    )
