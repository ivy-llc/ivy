# global
from hypothesis import strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_functional.test_core.test_statistical import (
    _statistical_dtype_values,
    _get_castable_dtype,
)
from ivy_tests.test_ivy.test_functional.test_experimental.test_core.test_statistical import (  # noqa
    _quantile_helper, _nanquantile_helper
)


@handle_frontend_test(
    fn_tree="torch.dist",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
    p=helpers.floats(min_value=1.0, max_value=10.0),
)
def test_torch_dist(
    *,
    dtype_and_input,
    p,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input[0],
        other=input[1],
        p=p,
    )


@handle_frontend_test(
    fn_tree="torch.argmax",
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        force_int_axis=True,
        min_num_dims=1,
        min_axis=-1,
        max_axis=0,
    ),
    keepdims=st.booleans(),
)
def test_torch_argmax(
    *,
    dtype_input_axis,
    keepdims,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x, axis = dtype_input_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        dim=axis,
        keepdim=keepdims,
    )


@handle_frontend_test(
    fn_tree="torch.argmin",
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        force_int_axis=True,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
        min_value=1,
        max_value=5,
        valid_axis=True,
        allow_neg_axes=True,
    ),
    keepdims=st.booleans(),
)
def test_torch_argmin(
    *,
    dtype_input_axis,
    keepdims,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x, axis = dtype_input_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        dim=axis,
        keepdim=keepdims,
    )


@handle_frontend_test(
    fn_tree="torch.amax",
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        min_axis=-1,
        max_axis=0,
    ),
    keepdims=st.booleans(),
)
def test_torch_amax(
    *,
    dtype_input_axis,
    keepdims,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x, axis = dtype_input_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        dim=axis,
        keepdim=keepdims,
    )


@handle_frontend_test(
    fn_tree="torch.amin",
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        min_axis=-1,
        max_axis=0,
    ),
    keepdims=st.booleans(),
)
def test_torch_amin(
    *,
    dtype_input_axis,
    keepdims,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x, axis = dtype_input_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        dim=axis,
        keepdim=keepdims,
    )


@handle_frontend_test(
    fn_tree="torch.all",
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        allow_inf=False,
    ),
    keepdims=st.booleans(),
)
def test_torch_all(
    *,
    dtype_input_axis,
    keepdims,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x, axis = dtype_input_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        dim=axis,
        keepdim=keepdims,
    )


@handle_frontend_test(
    fn_tree="torch.any",
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        allow_inf=False,
    ),
    keepdims=st.booleans(),
)
def test_torch_any(
    *,
    dtype_input_axis,
    keepdims,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x, axis = dtype_input_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        dim=axis,
        keepdim=keepdims,
    )


@handle_frontend_test(
    fn_tree="torch.sum",
    dtype_and_x=_get_castable_dtype(
        min_value=-1e04,
        max_value=1e04,
    ),
    keepdims=st.booleans(),
)
def test_torch_sum(
    *,
    dtype_and_x,
    keepdims,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x, axis, castable_dtype = dtype_and_x
    if test_flags.as_variable:
        castable_dtype = input_dtype
    input_dtype = [input_dtype]
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        dim=axis,
        keepdim=keepdims,
        dtype=castable_dtype,
    )


@handle_frontend_test(
    fn_tree="torch.mean",
    dtype_and_x=_statistical_dtype_values(
        function="mean",
        min_value=-1e04,
        max_value=1e04,
    ),
    keepdims=st.booleans(),
    dtypes=helpers.get_dtypes("float_and_complex", none=True, full=False),
)
def test_torch_mean(
    *,
    dtype_and_x,
    keepdims,
    dtypes,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x, axis = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        dim=axis,
        keepdim=keepdims,
        dtype=dtypes[0],
        atol=1e-2,
    )


@handle_frontend_test(
    fn_tree="torch.nanmean",
    dtype_and_x=_statistical_dtype_values(
        function="nanmean",
        min_value=-1e04,
        max_value=1e04,
    ),
    keepdims=st.booleans(),
)
def test_torch_nanmean(
    *,
    dtype_and_x,
    keepdims,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x, axis = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        dim=axis,
        keepdim=keepdims,
    )


@handle_frontend_test(
    fn_tree="torch.nanquantile",
    dtype_and_x=_statistical_dtype_values(
        function="nanquantile",
    ),
    keepdims=st.booleans(),
)
def test_torch_nanquantile(
    *,
    dtype_and_x,
    keepdims,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x, axis = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        dim=axis,
        keepdim=keepdims,
    )


@handle_frontend_test(
    fn_tree="torch.median",
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
    ),
    keepdim=st.booleans(),
)
def test_torch_median(
    *,
    dtype_input_axis,
    keepdim,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, input, dim = dtype_input_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input[0],
        dim=dim,
        keepdim=keepdim,
    )


@handle_frontend_test(
    fn_tree="torch.std",
    dtype_and_x=_statistical_dtype_values(function="std"),
    keepdims=st.booleans(),
)
def test_torch_std(
    *,
    dtype_and_x,
    keepdims,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x, axis, correction = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        dim=axis,
        unbiased=bool(correction),
        keepdim=keepdims,
    )


# prod
@handle_frontend_test(
    fn_tree="torch.prod",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=5,
        valid_axis=True,
        allow_neg_axes=False,
        max_axes_size=1,
        force_int_axis=True,
        large_abs_safety_factor=10,
        small_abs_safety_factor=10,
        safety_factor_scale="log",
    ),
    dtype=helpers.get_dtypes("numeric", none=True, full=False),
    keepdims=st.booleans(),
)
def test_torch_prod(
    *,
    dtype_x_axis,
    dtype,
    keepdims,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x, axis = dtype_x_axis
    # ToDo: set as_variable_flags as the parameter generated by test_torch_prod once
    #  this issue is marked as completed https://github.com/pytorch/pytorch/issues/75733
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
        keepdim=keepdims,
        dtype=dtype[0],
    )


@handle_frontend_test(
    fn_tree="torch.var",
    dtype_and_x=_statistical_dtype_values(
        function="var",
        min_value=-1e04,
        max_value=1e04,
    ),
    keepdims=st.booleans(),
)
def test_torch_var(
    *,
    dtype_and_x,
    keepdims,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x, axis, correction = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        dim=axis,
        unbiased=bool(correction),
        keepdim=keepdims,
    )


# min
@handle_frontend_test(
    fn_tree="torch.min",
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
        num_arrays=st.integers(min_value=1, max_value=2),
    ),
    keepdim=st.booleans(),
)
def test_torch_min(
    *,
    dtype_input_axis,
    keepdim,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, input, axis = dtype_input_axis
    inputs = {f"input{i}": input[i] for i in range(len(input))}
    kwargs = {"dim": axis, "keepdim": keepdim} if len(inputs) == 1 else {}
    test_flags.num_positional_args = len(input)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        **inputs,
        **kwargs,
    )


# moveaxis
@handle_frontend_test(
    fn_tree="torch.moveaxis",
    dtype_and_a=helpers.dtype_and_values(
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
def test_torch_moveaxis(
    *,
    dtype_and_a,
    source,
    destination,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, a = dtype_and_a
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=a[0],
        source=source,
        destination=destination,
    )


@handle_frontend_test(
    fn_tree="torch.max",
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
        num_arrays=st.integers(min_value=1, max_value=2),
    ),
    keepdim=st.booleans(),
)
def test_torch_max(
    *,
    dtype_input_axis,
    keepdim,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, input, axis = dtype_input_axis
    inputs = {f"input{i}": input[i] for i in range(len(input))}
    kwargs = {"dim": axis, "keepdim": keepdim} if len(inputs) == 1 else {}
    test_flags.num_positional_args = len(input)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        **inputs,
        **kwargs,
    )


@handle_frontend_test(
    fn_tree="torch.std_mean",
    dtype_and_x=_statistical_dtype_values(
        function="std_mean",
        min_value=-1e04,
        max_value=1e04,
    ),
    keepdims=st.booleans(),
)
def test_torch_std_mean(
    *,
    dtype_and_x,
    keepdims,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x, axis, correction = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        dim=axis,
        unbiased=bool(correction),
        keepdim=keepdims,
    )


@handle_frontend_test(
    fn_tree="torch.var_mean",
    dtype_and_x=_statistical_dtype_values(
        function="var_mean",
        min_value=-1e04,
        max_value=1e04,
    ),
    keepdims=st.booleans(),
)
def test_torch_var_mean(
    *,
    dtype_and_x,
    keepdims,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x, axis, correction = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        dim=axis,
        unbiased=bool(correction),
        keepdim=keepdims,
    )


@handle_frontend_test(
    fn_tree="torch.aminmax",
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        min_axis=-1,
        max_axis=0,
    ),
    keepdims=st.booleans(),
)
def test_torch_aminmax(
    *,
    dtype_input_axis,
    keepdims,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    input_dtype, x, axis = dtype_input_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        dim=axis,
        keepdim=keepdims,
    )


@handle_frontend_test(
    fn_tree="torch.quantile",
    dtype_and_x=_quantile_helper(),
    keepdims=st.booleans(),
)
def test_torch_quantile(
    *,
    dtype_and_x,
    keepdims,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x, axis, interpolation, q = dtype_and_x
    if type(axis) is tuple:
        axis = axis[0]
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        q=q,
        dim=axis,
        keepdim=keepdims,
        interpolation=interpolation[0],
    )


@handle_frontend_test(
    fn_tree="torch.count_nonzero",
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        force_int_axis=True,
        min_num_dims=1,
        min_axis=-1,
        max_axis=0,
    ),
)
def test_torch_count_nonzero(
    *,
    dtype_input_axis,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x, axis = dtype_input_axis
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


@handle_frontend_test(
    fn_tree="torch.logsumexp",
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_value=-50,
        max_value=50,
        min_num_dims=1,
        max_num_dims=5,
        valid_axis=True,
        force_int_axis=True,
    ),
    keepdims=st.booleans(),
)
def test_torch_logsumexp(
    *,
    dtype_input_axis,
    keepdims,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x, axis = dtype_input_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        dim=axis,
        keepdim=keepdims,
    )


@handle_frontend_test(
    fn_tree="torch.unique",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        force_int_axis=True,
        valid_axis=True,
    ),
    return_inverse=st.booleans(),
    return_counts=st.booleans(),
    sorted=st.booleans(),
)
def test_torch_unique(
    *,
    dtype_x_axis,
    return_inverse,
    return_counts,
    sorted,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtypes, x, axis = dtype_x_axis

    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        sorted=sorted,
        return_inverse=return_inverse,
        return_counts=return_counts,
        dim=axis,
    )


@st.composite
def _get_axis_and_p(draw):
    p = draw(st.sampled_from(["fro", "nuc", 1, 2, -1, -2, float("inf"), -float("inf")]))
    if p == "fro" or p == "nuc":
        max_axes_size = 2
        min_axes_size = 2
    else:
        min_axes_size = 1
        max_axes_size = 5
    x_dtype, values, axis = draw(
        helpers.dtype_values_axis(
            available_dtypes=helpers.get_dtypes("valid"),
            min_num_dims=2,
            valid_axis=True,
            min_value=-1e04,
            max_value=1e04,
            min_axes_size=min_axes_size,
            max_axes_size=max_axes_size,
            large_abs_safety_factor=2,
            safety_factor_scale="log",
        )
    )
    axis = axis[0] if isinstance(axis, tuple) and len(axis) == 1 else axis
    # ToDo: fix the castable dtype helper. Right now using `dtype` causes errors
    #  dtype should be real for real inputs, but got ComplexDouble
    x_dtype, values, dtype = draw(
        helpers.get_castable_dtype(
            draw(helpers.get_dtypes("valid")), x_dtype[0], values[0]
        )
    )
    return p, x_dtype, values, axis, x_dtype


# norm
@handle_frontend_test(
    fn_tree="torch.norm",
    p_dtype_x_axis=_get_axis_and_p(),
    keepdim=st.booleans(),
)
def test_torch_norm(
    *,
    p_dtype_x_axis,
    keepdim,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    p, x_dtype, x, axis, dtype = p_dtype_x_axis
    helpers.test_frontend_function(
        backend_to_test=backend_fw,
        input_dtypes=[x_dtype],
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-01,
        atol=1e-08,
        input=x,
        p=p,
        dim=axis,
        keepdim=keepdim,
        out=None,
        dtype=dtype,
    )


# known bug of returning empty tensors when ret_inv or ret_counts is passed positionally
# https://github.com/pytorch/pytorch/issues/68610
# ToDo: activate test_values when this is resolved
@handle_frontend_test(
    fn_tree="torch.unique_consecutive",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        min_dim_size=2,
        force_int_axis=True,
        valid_axis=True,
    ),
    ret_inv=st.booleans(),
    ret_counts=st.booleans(),
)
def test_torch_unique_consecutive(
    *,
    dtype_x_axis,
    ret_inv,
    ret_counts,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        return_inverse=ret_inv,
        return_counts=ret_counts,
        dim=axis,
        test_values=False,
    )
