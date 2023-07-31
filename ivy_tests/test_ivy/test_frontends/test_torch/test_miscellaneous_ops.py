# global
import math

import numpy as np
from hypothesis import assume, strategies as st
import hypothesis.extra.numpy as nph

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_functional.test_core.test_linalg import (
    _get_dtype_value1_value2_axis_for_tensordot,
)


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


# atleast_1d
@handle_frontend_test(
    fn_tree="torch.atleast_1d",
    dtype_and_tensors=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=st.integers(min_value=1, max_value=5),
    ),
    test_with_out=st.just(False),
)
def test_torch_atleast_1d(
    *,
    dtype_and_tensors,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtypes, tensors = dtype_and_tensors
    if isinstance(dtypes, list):  # If more than one value was generated
        args = {
            f"x{i}": np.array(tensor, dtype=dtypes[i])
            for i, tensor in enumerate(tensors)
        }
    else:  # If exactly one value was generated
        args = {"x0": np.array(tensors, dtype=dtypes)}
    test_flags.num_positional_args = len(tensors)
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        **args,
    )


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
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, value = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
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
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
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
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=value[0],
        shifts=shift,
        dims=axis,
    )


# meshgrid
@handle_frontend_test(
    fn_tree="torch.meshgrid",
    dtypes_and_tensors=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=st.integers(min_value=2, max_value=5),
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
        max_dim_size=5,
        shared_dtype=True,
    ),
    indexing=st.sampled_from(["ij", "xy"]),
)
def test_torch_meshgrid(
    *,
    dtypes_and_tensors,
    indexing,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtypes, tensors = dtypes_and_tensors
    kwargs = {
        f"tensor{i}": np.array(tensor, dtype=dtypes[i])
        for i, tensor in enumerate(tensors)
    }
    kwargs["indexing"] = indexing
    test_flags.num_positional_args = len(tensors)
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        **kwargs,
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
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, value = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=value[0],
    )


# flipud
@handle_frontend_test(
    fn_tree="torch.flipud",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=helpers.get_shape(min_num_dims=1),
    ),
)
def test_torch_flipud(
    *,
    dtype_and_values,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, value = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=value[0],
    )


# cumsum
@handle_frontend_test(
    fn_tree="torch.cumsum",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_value=-100,
        max_value=100,
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
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x, axis = dtype_x_axis
    # ToDo: set as_variable_flags as the parameter generated by test_torch_cumsum once
    # this issue is marked as completed https://github.com/pytorch/pytorch/issues/75733
    if ivy.current_backend_str() == "torch":
        test_flags.as_variable = [False]
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
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
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
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
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
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
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtypes, tensors = dtype_and_tensors
    if isinstance(dtypes, list):  # If more than one value was generated
        args = {
            f"x{i}": np.array(tensor, dtype=dtypes[i])
            for i, tensor in enumerate(tensors)
        }
    else:  # If exactly one value was generated
        args = {"x0": np.array(tensors, dtype=dtypes)}
    test_flags.num_positional_args = len(tensors)
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
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
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, values = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=values[0],
        diagonal=diagonal,
    )


# cummax
@handle_frontend_test(
    fn_tree="torch.cummax",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        max_num_dims=2,
        min_value=-100,
        max_value=100,
        valid_axis=True,
        allow_neg_axes=False,
        max_axes_size=1,
        force_int_axis=True,
    ),
    dtype=helpers.get_dtypes("float", none=True, full=False),
)
def test_torch_cummax(
    *,
    dtype_x_axis,
    dtype,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x, axis = dtype_x_axis
    if ivy.current_backend_str() == "torch":
        test_flags.as_variable = [False]
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        dim=axis,
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
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x, axis = dtype_x_axis
    # ToDo: set as_variable_flags as the parameter generated by test_torch_cumsum once
    # this issue is marked as completed https://github.com/pytorch/pytorch/issues/75733
    if ivy.current_backend_str() == "torch":
        test_flags.as_variable = [False]
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
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
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, value = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=value[0],
    )


# tril_indices
@handle_frontend_test(
    fn_tree="torch.tril_indices",
    row=st.integers(min_value=1, max_value=10),
    col=st.integers(min_value=1, max_value=10),
    offset=st.integers(min_value=-8, max_value=8),
    dtype=helpers.get_dtypes("integer", full=False),
)
def test_torch_tril_indices(
    *,
    row,
    col,
    offset,
    dtype,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    helpers.test_frontend_function(
        input_dtypes=[ivy.int32],
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        row=row,
        col=col,
        offset=offset,
        dtype=dtype[0],
    )


@handle_frontend_test(
    fn_tree="torch.triu_indices",
    row=st.integers(min_value=1, max_value=100),
    col=st.integers(min_value=1, max_value=100),
    offset=st.integers(min_value=-10, max_value=10),
)
def test_torch_triu_indices(
    *,
    row,
    col,
    offset,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    helpers.test_frontend_function(
        input_dtypes=["int32"],
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
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
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, values = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=values[0],
        diagonal=diagonal,
    )


@handle_frontend_test(
    fn_tree="torch.flatten",
    dtype_input_axes=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        valid_axis=True,
        min_num_dims=1,
        min_axes_size=2,
        max_axes_size=2,
    ),
)
def test_torch_flatten(
    *,
    dtype_input_axes,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, input, axes = dtype_input_axes
    if isinstance(axes, int):
        start_dim = axes
        end_dim = -1
    else:
        start_dim = axes[0]
        end_dim = axes[1]
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
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
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"),
        available_dtypes=helpers.get_dtypes("numeric"),
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
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, values = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
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
        available_dtypes=helpers.get_dtypes("numeric"),
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
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
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
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, values, repeats, axis, output_size = dtype_values_repeats_axis_output_size

    helpers.test_frontend_function(
        input_dtypes=dtype[0],
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
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
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
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
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
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
        min_num_dims=0,
        max_num_dims=5,
    ),
    N=st.integers(min_value=1, max_value=10) | st.none(),
    increasing=st.booleans(),
)
def test_torch_vander(
    *,
    dtype_and_x,
    N,
    increasing,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
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
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
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
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    eq, operands, _ = eq_n_op_n_shp
    kw = {}
    i = 0
    for x_ in operands:
        kw["x{}".format(i)] = x_
        i += 1
    # len(operands) + 1 because of the equation
    test_flags.num_positional_args = len(operands) + 1
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
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
        min_value=-1e5,
        max_value=1e5,
        abs_smallest_val=0.01,
        large_abs_safety_factor=2,
        safety_factor_scale="log",
    ),
)
def test_torch_cross(
    dtype_input_other_dim,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
):
    dtype, input, other, dim = dtype_input_other_dim
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        rtol=1e-2,
        atol=1e-2,
        input=input,
        other=other,
        dim=dim,
    )


# gcd
@handle_frontend_test(
    fn_tree="torch.gcd",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        min_value=-100,
        max_value=100,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_torch_gcd(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        other=x[1],
    )


@handle_frontend_test(
    fn_tree="torch.tensordot",
    dtype_values_and_axes=_get_dtype_value1_value2_axis_for_tensordot(
        helpers.get_dtypes(kind="float"),
        min_value=-10,
        max_value=10,
    ),
)
def test_torch_tensordot(
    dtype_values_and_axes,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
):
    dtype, a, b, dims = dtype_values_and_axes
    if ivy.current_backend_str() == "paddle":
        # Paddle only supports ndim from 0 to 9
        assume(a.shape[0] < 10)
        assume(b.shape[0] < 10)

    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        a=a,
        b=b,
        rtol=1e-2,
        atol=1e-2,
        dims=dims,
    )


# diff
@handle_frontend_test(
    fn_tree="torch.diff",
    dtype_n_x_n_axis=helpers.dtype_values_axis(
        available_dtypes=st.shared(helpers.get_dtypes("valid"), key="dtype"),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
    ),
    n=st.integers(min_value=0, max_value=5),
    dtype_prepend=helpers.dtype_and_values(
        available_dtypes=st.shared(helpers.get_dtypes("valid"), key="dtype"),
        min_num_dims=1,
        max_num_dims=1,
    ),
    dtype_append=helpers.dtype_and_values(
        available_dtypes=st.shared(helpers.get_dtypes("valid"), key="dtype"),
        min_num_dims=1,
        max_num_dims=1,
    ),
)
def test_torch_diff(
    *,
    dtype_n_x_n_axis,
    n,
    dtype_prepend,
    dtype_append,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
):
    input_dtype, x, axis = dtype_n_x_n_axis
    _, prepend = dtype_prepend
    _, append = dtype_append
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        input=x[0],
        n=n,
        dim=axis,
        prepend=prepend[0],
        append=append[0],
    )


@handle_frontend_test(
    fn_tree="torch.broadcast_shapes",
    shapes=nph.mutually_broadcastable_shapes(
        num_shapes=4, min_dims=1, max_dims=5, min_side=1, max_side=5
    ),
    test_with_out=st.just(False),
)
def test_torch_broadcast_shapes(
    *,
    shapes,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    shape, _ = shapes
    shapes = {f"shape{i}": shape[i] for i in range(len(shape))}
    test_flags.num_positional_args = len(shapes)
    ret, frontend_ret = helpers.test_frontend_function(
        input_dtypes=["int64"],
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        **shapes,
        test_values=False,
    )
    assert ret == frontend_ret


# atleast_2d
@handle_frontend_test(
    fn_tree="torch.atleast_2d",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=helpers.ints(min_value=1, max_value=10),
    ),
    test_with_out=st.just(False),
)
def test_torch_atleast_2d(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, arrays = dtype_and_x
    arys = {}
    for i, (array, idtype) in enumerate(zip(arrays, input_dtype)):
        arys["arrs{}".format(i)] = array
    test_flags.num_positional_args = len(arys)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        **arys,
    )


@handle_frontend_test(
    fn_tree="torch.searchsorted",
    dtype_x_v=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shared_dtype=True,
        min_num_dims=1,
        max_num_dims=1,
        num_arrays=2,
    ),
    side=st.sampled_from(["left", "right"]),
    out_int32=st.booleans(),
    right=st.just(False),
    test_with_out=st.just(False),
)
def test_torch_searchsorted(
    dtype_x_v,
    side,
    out_int32,
    right,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtypes, xs = dtype_x_v
    use_sorter = st.booleans()
    if use_sorter:
        sorter = np.argsort(xs[0])
        sorter = np.array(sorter, dtype=np.int64)
    else:
        xs[0] = np.sort(xs[0])
        sorter = None
    helpers.test_frontend_function(
        input_dtypes=input_dtypes + ["int64"],
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        sorted_sequence=xs[0],
        values=xs[1],
        side=side,
        out_int32=out_int32,
        right=right,
        sorter=sorter,
    )


# atleast_3d
@handle_frontend_test(
    fn_tree="torch.atleast_3d",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=helpers.ints(min_value=1, max_value=10),
    ),
    test_with_out=st.just(False),
)
def test_torch_atleast_3d(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, arrays = dtype_and_x
    arys = {}
    for i, array in enumerate(arrays):
        arys["arrs{}".format(i)] = array
    test_flags.num_positional_args = len(arys)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        **arys,
    )


# diag
@handle_frontend_test(
    fn_tree="torch.diag",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=st.shared(helpers.get_shape(min_num_dims=1, max_num_dims=2), key="shape"),
    ),
    diagonal=st.integers(min_value=-100, max_value=100),
)
def test_torch_diag(
    *,
    dtype_and_values,
    diagonal,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, values = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=values[0],
        diagonal=diagonal,
    )


# clone
@handle_frontend_test(
    fn_tree="torch.clone",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_torch_clone(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
    )


@st.composite
def _get_dtype_value1_value2_cov(
    draw,
    available_dtypes,
    min_num_dims,
    max_num_dims,
    min_dim_size,
    max_dim_size,
    abs_smallest_val=None,
    min_value=None,
    max_value=None,
    allow_inf=False,
    exclude_min=False,
    exclude_max=False,
    large_abs_safety_factor=4,
    small_abs_safety_factor=4,
    safety_factor_scale="log",
):
    shape = draw(
        helpers.get_shape(
            allow_none=False,
            min_num_dims=min_num_dims,
            max_num_dims=max_num_dims,
            min_dim_size=min_dim_size,
            max_dim_size=max_dim_size,
        )
    )

    dtype = draw(st.sampled_from(draw(available_dtypes)))

    values = []
    for i in range(1):
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

    value1 = values[0]

    correction = draw(helpers.ints(min_value=0, max_value=1))

    fweights = draw(
        helpers.array_values(
            dtype="int64",
            shape=shape[1],
            abs_smallest_val=1,
            min_value=1,
            max_value=10,
            allow_inf=False,
        )
    )

    aweights = draw(
        helpers.array_values(
            dtype="float64",
            shape=shape[1],
            abs_smallest_val=1,
            min_value=1,
            max_value=10,
            allow_inf=False,
            small_abs_safety_factor=1,
        )
    )

    return [dtype], value1, correction, fweights, aweights


# cov
@handle_frontend_test(
    fn_tree="torch.cov",
    dtype_x1_corr_cov=_get_dtype_value1_value2_cov(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=2,
        max_dim_size=5,
        min_value=1,
        max_value=1e10,
        abs_smallest_val=0.01,
        large_abs_safety_factor=2,
        safety_factor_scale="log",
    ),
    test_with_out=st.just(False),
)
def test_torch_cov(
    dtype_x1_corr_cov,
    test_flags,
    frontend,
    fn_tree,
    on_device,
    backend_fw,
):
    dtype, x1, correction, fweights, aweights = dtype_x1_corr_cov
    helpers.test_frontend_function(
        input_dtypes=["float64", "int64", "float64"],
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-2,
        atol=1e-2,
        input=x1,
        correction=correction,
        fweights=fweights,
        aweights=aweights,
    )


# view_as_real
@handle_frontend_test(
    fn_tree="torch.view_as_real",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=1,
    ),
)
def test_torch_view_as_real(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=np.asarray(x[0], dtype=input_dtype[0]),
    )


@st.composite
def complex_strategy(
    draw, min_num_dims=0, max_num_dims=5, min_dim_size=1, max_dim_size=10
):
    shape = draw(
        st.lists(
            helpers.ints(min_value=min_dim_size, max_value=max_dim_size),
            min_size=min_num_dims,
            max_size=max_num_dims,
        )
    )
    shape = list(shape)
    shape.append(2)
    return tuple(shape)


# view_as_complex
@handle_frontend_test(
    fn_tree="torch.view_as_complex",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=st.shared(complex_strategy()),
    ),
)
def test_torch_view_as_complex(
    *,
    dtype_and_values,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    dtype, value = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=value[0],
    )
