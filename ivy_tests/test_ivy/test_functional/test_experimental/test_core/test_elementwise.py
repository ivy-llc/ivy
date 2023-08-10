# global
from hypothesis import strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


# Helpers #
# ------- #
# lgamma
@handle_test(
    fn_tree="functional.ivy.experimental.lgamma",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        safety_factor_scale="log",
    ),
    test_gradients=st.just(False),
)
def test_lgamma(
    *,
    dtype_and_x,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        on_device=on_device,
        fn_name=fn_name,
        x=x[0],
    )


# sinc
@handle_test(
    fn_tree="functional.ivy.experimental.sinc",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=4,
        small_abs_safety_factor=4,
    ),
    test_gradients=st.just(False),
)
def test_sinc(*, dtype_and_x, test_flags, backend_fw, fn_name, on_device):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        atol_=1e-02,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x=x[0],
    )


# fmax
@handle_test(
    fn_tree="functional.ivy.experimental.fmax",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        large_abs_safety_factor=4,
        small_abs_safety_factor=4,
        safety_factor_scale="log",
        num_arrays=2,
    ),
    test_gradients=st.just(False),
)
def test_fmax(dtype_and_x, test_flags, backend_fw, fn_name, on_device):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x1=x[0],
        x2=x[1],
    )


# float_power_helper
@st.composite
def _float_power_helper(draw, *, available_dtypes=None):
    if available_dtypes is None:
        available_dtypes = helpers.get_dtypes("numeric")
    dtype1, x1 = draw(
        helpers.dtype_and_values(
            available_dtypes=available_dtypes,
            small_abs_safety_factor=16,
            large_abs_safety_factor=16,
            safety_factor_scale="log",
        )
    )
    dtype2 = draw(helpers.get_dtypes("numeric"))
    if ivy.is_int_dtype(dtype2[0]):
        min_value = 0
    else:
        min_value = -10
    dtype2, x2 = draw(
        helpers.dtype_and_values(
            min_value=min_value,
            max_value=10,
            dtype=dtype2,
        )
    )
    return (dtype1[0], dtype2[0]), (x1[0], x2[0])


# float_power
@handle_test(
    fn_tree="functional.ivy.experimental.float_power",
    dtype_and_x=_float_power_helper(),
    test_gradients=st.just(False),
)
def test_float_power(dtype_and_x, test_flags, backend_fw, fn_name, on_device):
    input_dtypes, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtypes,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x1=x[0],
        x2=x[1],
        rtol_=1e-1,
        atol_=1e-1,
    )


# copysign
@handle_test(
    fn_tree="functional.ivy.experimental.copysign",
    dtype_x1_x2=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_num_dims=0,
        allow_nan=False,
        shared_dtype=False,
    ),
    test_gradients=st.just(False),
)
def test_copysign(dtype_x1_x2, test_flags, backend_fw, fn_name, on_device):
    (x1_dtype, x2_dtype), (x1, x2) = dtype_x1_x2
    helpers.test_function(
        input_dtypes=[x1_dtype, x2_dtype],
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x1=x1,
        x2=x2,
    )


@st.composite
def _get_dtype_values_axis_for_count_nonzero(
    draw,
    in_available_dtypes,
    out_available_dtypes,
    min_num_dims=1,
    max_num_dims=10,
    min_dim_size=1,
    max_dim_size=10,
):
    input_dtype, values, axis = draw(
        helpers.dtype_values_axis(
            available_dtypes=helpers.get_dtypes(in_available_dtypes),
            min_num_dims=min_num_dims,
            max_num_dims=max_num_dims,
            min_dim_size=min_dim_size,
            max_dim_size=max_dim_size,
            valid_axis=True,
        )
    )
    axis = draw(st.one_of(st.just(axis), st.none()))
    output_dtype = draw(st.one_of(helpers.get_dtypes(out_available_dtypes)))
    return [input_dtype, output_dtype], values, axis


# count_nonzero
@handle_test(
    fn_tree="functional.ivy.experimental.count_nonzero",
    dtype_values_axis=_get_dtype_values_axis_for_count_nonzero(
        in_available_dtypes="integer",
        out_available_dtypes="integer",
        min_num_dims=1,
        max_num_dims=10,
        min_dim_size=1,
        max_dim_size=10,
    ),
    keepdims=st.booleans(),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
)
def test_count_nonzero(
    *, dtype_values_axis, keepdims, test_flags, on_device, fn_name, backend_fw
):
    i_o_dtype, a, axis = dtype_values_axis
    helpers.test_function(
        input_dtypes=i_o_dtype[0],
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        a=a[0],
        axis=axis,
        keepdims=keepdims,
        dtype=i_o_dtype[1][0],
    )


# nansum
@st.composite
def _get_castable_dtypes_values(draw, *, allow_nan=False):
    available_dtypes = helpers.get_dtypes("numeric")
    shape = draw(helpers.get_shape(min_num_dims=1, max_num_dims=4, max_dim_size=6))
    dtype, values = draw(
        helpers.dtype_and_values(
            available_dtypes=available_dtypes,
            num_arrays=1,
            large_abs_safety_factor=24,
            small_abs_safety_factor=24,
            safety_factor_scale="log",
            shape=shape,
            allow_nan=allow_nan,
        )
    )
    axis = draw(helpers.get_axis(shape=shape, force_int=True))
    dtype1, values, dtype2 = draw(
        helpers.get_castable_dtype(
            draw(helpers.get_dtypes("float")), dtype[0], values[0]
        )
    )
    return [dtype1], [values], axis, dtype2


# nansum
@handle_test(
    fn_tree="functional.ivy.experimental.nansum",
    dtype_x_axis_dtype=_get_castable_dtypes_values(allow_nan=True),
    keep_dims=st.booleans(),
    test_gradients=st.just(False),
)
def test_nansum(
    *, dtype_x_axis_dtype, keep_dims, test_flags, on_device, fn_name, backend_fw
):
    input_dtype, x, axis, dtype = dtype_x_axis_dtype
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        on_device=on_device,
        fn_name=fn_name,
        x=x[0],
        axis=axis,
        keepdims=keep_dims,
        dtype=dtype,
    )


# isclose
@handle_test(
    fn_tree="functional.ivy.experimental.isclose",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=4,
        small_abs_safety_factor=4,
        safety_factor_scale="log",
        num_arrays=2,
        shared_dtype=True,
        allow_nan=True,
    ),
    rtol=st.floats(
        min_value=1e-05, max_value=1e-01, exclude_min=True, exclude_max=True
    ),
    atol=st.floats(
        min_value=1e-08, max_value=1e-01, exclude_min=True, exclude_max=True
    ),
    equal_nan=st.booleans(),
    test_gradients=st.just(False),
)
def test_isclose(
    *, dtype_and_x, rtol, atol, equal_nan, test_flags, backend_fw, fn_name, on_device
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        a=x[0],
        b=x[1],
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
    )


# allclose
@handle_test(
    fn_tree="functional.ivy.experimental.allclose",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=4,
        small_abs_safety_factor=4,
        safety_factor_scale="log",
        num_arrays=2,
        shared_dtype=True,
        allow_nan=True,
    ),
    rtol=st.floats(
        min_value=1e-05, max_value=1e-01, exclude_min=True, exclude_max=True
    ),
    atol=st.floats(
        min_value=1e-08, max_value=1e-01, exclude_min=True, exclude_max=True
    ),
    equal_nan=st.booleans(),
    test_gradients=st.just(False),
    test_with_out=st.just(False),
)
def test_allclose(
    dtype_and_x, rtol, atol, equal_nan, test_flags, backend_fw, fn_name, on_device
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x1=x[0],
        x2=x[1],
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
    )


# fix
@handle_test(
    fn_tree="functional.ivy.experimental.fix",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
    ),
    test_gradients=st.just(False),
)
def test_fix(dtype_and_x, test_flags, backend_fw, fn_name, on_device):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
    )


# nextafter
@handle_test(
    fn_tree="functional.ivy.experimental.nextafter",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=["float32", "float64"],
        num_arrays=2,
        shared_dtype=True,
        min_value=-10,
        max_value=10,
        min_num_dims=1,
        max_num_dims=3,
    ),
    test_gradients=st.just(False),
)
def test_nextafter(*, dtype_and_x, test_flags, backend_fw, fn_name, on_device):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x1=x[0],
        x2=x[1],
    )


# diff
@handle_test(
    fn_tree="functional.ivy.experimental.diff",
    dtype_n_x_n_axis=helpers.dtype_values_axis(
        available_dtypes=st.shared(helpers.get_dtypes("valid"), key="dtype"),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
    ),
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
    n=st.integers(min_value=0, max_value=5),
    test_gradients=st.just(False),
)
def test_diff(
    *,
    dtype_n_x_n_axis,
    n,
    dtype_prepend,
    dtype_append,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    input_dtype, x, axis = dtype_n_x_n_axis
    _, prepend = dtype_prepend
    _, append = dtype_append
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
        n=n,
        axis=axis,
        prepend=prepend[0],
        append=append[0],
    )


# zeta
@handle_test(
    fn_tree="functional.ivy.experimental.zeta",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
        min_value=-10,
        max_value=10,
        min_num_dims=1,
        max_num_dims=3,
    ),
    test_gradients=st.just(False),
)
def test_zeta(
    dtype_and_x,
    test_flags,
    fn_name,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        on_device=on_device,
        fn_name=fn_name,
        rtol_=1e-02,
        atol_=1e-02,
        x=x[0],
        q=x[1],
    )


# gradient
@handle_test(
    fn_tree="functional.ivy.experimental.gradient",
    dtype_n_x_n_axis=helpers.dtype_values_axis(
        available_dtypes=("float32", "float16", "float64"),
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=2,
        max_dim_size=4,
        valid_axis=True,
        force_int_axis=True,
    ),
    spacing=helpers.ints(
        min_value=-3,
        max_value=3,
    ),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
)
def test_gradient(
    *, dtype_n_x_n_axis, spacing, test_flags, backend_fw, fn_name, on_device
):
    input_dtype, x, axis = dtype_n_x_n_axis
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        on_device=on_device,
        fn_name=fn_name,
        x=x[0],
        spacing=spacing,
        axis=axis,
    )


# xlogy
@handle_test(
    fn_tree="functional.ivy.experimental.xlogy",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_complex"),
        num_arrays=2,
        min_value=-10,
        max_value=10,
        min_num_dims=1,
        max_num_dims=3,
    ),
    test_gradients=st.just(False),
)
def test_xlogy(*, dtype_and_x, test_flags, backend_fw, fn_name, on_device):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
        y=x[1],
    )


# hypot
@handle_test(
    fn_tree="functional.ivy.experimental.hypot",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
        min_value=-100,
        max_value=100,
        min_num_dims=1,
        max_num_dims=3,
    ),
    test_gradients=st.just(False),
)
def test_hypot(dtype_and_x, test_flags, backend_fw, fn_name, on_device):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        atol_=1e-2,
        x1=x[0],
        x2=x[1],
    )


@handle_test(
    fn_tree="functional.ivy.experimental.binarizer",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric")
    ),
    threshold=helpers.floats(),
    container_flags=st.just([False]),
)
def test_binarizer(
    *, dtype_and_x, threshold, test_flags, backend_fw, fn_name, on_device
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
        threshold=threshold,
    )


# conj
@handle_test(
    fn_tree="conj",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("real_and_complex")
    ),
    test_with_out=st.just(False),
)
def test_conj(*, dtype_and_x, test_flags, backend_fw, fn_name, on_device):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
    )


# ldexp
@st.composite
def ldexp_args(draw):
    dtype1, x1 = draw(
        helpers.dtype_and_values(
            available_dtypes=["float32", "float64"],
            num_arrays=1,
            shared_dtype=True,
            min_value=-100,
            max_value=100,
            min_num_dims=1,
            max_num_dims=3,
        )
    )
    dtype2, x2 = draw(
        helpers.dtype_and_values(
            available_dtypes=["int32", "int64"],
            num_arrays=1,
            shared_dtype=True,
            min_value=-100,
            max_value=100,
            min_num_dims=1,
            max_num_dims=3,
        )
    )
    return (dtype1[0], dtype2[0]), (x1[0], x2[0])


@handle_test(
    fn_tree="functional.ivy.experimental.ldexp",
    dtype_and_x=ldexp_args(),
    test_gradients=st.just(False),
)
def test_ldexp(*, dtype_and_x, test_flags, backend_fw, fn_name, on_device):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x1=x[0],
        x2=x[1],
    )


@st.composite
def _lerp_data_helper(draw):
    mixed_fn_compos = draw(st.booleans())
    is_torch_backend = ivy.current_backend_str() == "torch"

    kwargs = {
        "shared_dtype": True,
        "large_abs_safety_factor": 2.5,
        "small_abs_safety_factor": 2.5,
        "safety_factor_scale": "log",
        "allow_nan": False,
        "allow_inf": False,
    }

    if is_torch_backend and not mixed_fn_compos:
        dtype1, start_end = draw(
            helpers.dtype_and_values(
                available_dtypes=(
                    helpers.get_dtypes("numeric", mixed_fn_compos=mixed_fn_compos)
                ),
                num_arrays=2,
                **kwargs,
            )
        )
        dtype2, weight = draw(
            helpers.dtype_and_values(
                available_dtypes=helpers.get_dtypes(
                    "integer", mixed_fn_compos=mixed_fn_compos
                ),
                num_arrays=1,
                **kwargs,
            )
        )
        input_dtypes = dtype1 + dtype2
        inputs = start_end + weight
    else:
        input_dtypes, inputs = draw(
            helpers.dtype_and_values(
                available_dtypes=helpers.get_dtypes(
                    "valid", mixed_fn_compos=mixed_fn_compos
                ),
                num_arrays=3,
                **kwargs,
            )
        )

    return input_dtypes, inputs[0], inputs[1], inputs[2]


# lerp
@handle_test(
    fn_tree="functional.ivy.experimental.lerp",
    data=_lerp_data_helper(),
    test_gradients=st.just(False),
)
def test_lerp(
    *,
    data,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    input_dtypes, start, end, weight = data
    helpers.test_function(
        input_dtypes=input_dtypes,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        atol_=1e-01,
        rtol_=1e-01,
        on_device=on_device,
        input=start,
        end=end,
        weight=weight,
    )


# frexp
@handle_test(
    fn_tree="functional.ivy.experimental.frexp",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=["float32", "float64"],
        num_arrays=1,
        shared_dtype=True,
        min_value=-100,
        max_value=100,
        min_num_dims=1,
        max_num_dims=3,
    ),
    test_gradients=st.just(False),
)
def test_frexp(*, dtype_and_x, test_flags, backend_fw, fn_name, on_device):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
    )


# modf
@handle_test(
    fn_tree="functional.ivy.experimental.modf",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=1,
        min_value=0,
        exclude_min=True,
    ),
    test_with_out=st.just(False),
)
def test_modf(
    *,
    dtype_and_x,
    backend_fw,
    test_flags,
    fn_name,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
    )


# digamma
@handle_test(
    fn_tree="functional.ivy.experimental.digamma",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-10,
        max_value=10,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
    ).filter(lambda x: "bfloat16" not in x[0] and "float16" not in x[0]),
    ground_truth_backend="tensorflow",
)
def test_digamma(
    dtype_and_x,
    backend_fw,
    test_flags,
    fn_name,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_name=fn_name,
        on_device=on_device,
        x=x[0],
    )


#amin
@handle_test(
    fn_tree="functional.ivy.experimental.amin",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=1,
        min_value=0,
        exclude_min=True,
    ),
    test_with_out=st.just(False),
)
def test_amin(
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )
