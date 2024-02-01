# global
from hypothesis import strategies as st, assume
import numpy as np
import ivy

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_functional.test_core.test_linalg import (
    _get_first_matrix_and_dtype,
    _get_second_matrix_and_dtype,
    _get_dtype_value1_value2_axis_for_tensordot,
)
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.test_functional.test_experimental.test_core.test_elementwise import (  # noqa
    ldexp_args,
)


# --- Helpers --- #
# --------------- #


# trapz
@st.composite
def _either_x_dx(draw):
    dtype_values_axis = draw(
        helpers.dtype_values_axis(
            available_dtypes=st.shared(helpers.get_dtypes("float"), key="trapz_dtype"),
            min_value=-100,
            max_value=100,
            min_num_dims=1,
            max_num_dims=3,
            min_dim_size=1,
            max_dim_size=3,
            allow_neg_axes=True,
            valid_axis=True,
            force_int_axis=True,
        ),
    )
    rand = (draw(st.integers(min_value=0, max_value=1)),)
    if rand == 0:
        either_x_dx = draw(
            helpers.dtype_and_x(
                avaliable_dtypes=st.shared(
                    helpers.get_dtypes("float"), key="trapz_dtype"
                ),
                min_value=-100,
                max_value=100,
                min_num_dims=1,
                max_num_dims=3,
                min_dim_size=1,
                max_dim_size=3,
            )
        )
        return dtype_values_axis, rand, either_x_dx
    else:
        either_x_dx = draw(
            st.floats(min_value=-10, max_value=10),
        )
        return dtype_values_axis, rand, either_x_dx


# polyint
@st.composite
def _get_array_values_m_and_k(draw):
    dtype_and_x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
            num_arrays=1,
            min_num_dims=1,
            max_num_dims=1,
            min_dim_size=1,
        )
    )
    dtype, x = dtype_and_x
    m = draw(st.integers(min_value=0, max_value=10))
    max_bound = m - 1
    if max_bound <= m:
        k = None
    else:
        k = draw(st.integers(min_value=0, max_value=max_bound))
    return dtype, x, m, k


@st.composite
def _get_castable_dtypes_values(draw, *, allow_nan=False, use_where=False):
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
        helpers.get_castable_dtype(draw(available_dtypes), dtype[0], values[0])
    )
    if use_where:
        where = draw(np_frontend_helpers.where(shape=shape))
        return [dtype1], [values], axis, dtype2, where
    return [dtype1], [values], axis, dtype2


# diff
@st.composite
def _get_dtype_input_and_vector(draw):
    size1 = draw(helpers.ints(min_value=1, max_value=5))
    size2 = draw(helpers.ints(min_value=1, max_value=5))
    dtype = draw(helpers.get_dtypes("integer"))
    vec1 = draw(helpers.array_values(dtype=dtype[0], shape=(size1, size2)))
    return dtype, vec1


# dot
@st.composite
def _get_dtype_input_and_vectors(draw):
    dim_size = draw(helpers.ints(min_value=1, max_value=5))
    dtype = draw(helpers.get_dtypes("float", index=1, full=False))
    if dim_size == 1:
        vec1 = draw(
            helpers.array_values(
                dtype=dtype[0], shape=(dim_size,), min_value=2, max_value=5
            )
        )
        vec2 = draw(
            helpers.array_values(
                dtype=dtype[0], shape=(dim_size,), min_value=2, max_value=5
            )
        )
    else:
        vec1 = draw(
            helpers.array_values(
                dtype=dtype[0], shape=(dim_size, dim_size), min_value=2, max_value=5
            )
        )
        vec2 = draw(
            helpers.array_values(
                dtype=dtype[0], shape=(dim_size, dim_size), min_value=2, max_value=5
            )
        )
    return dtype, vec1, vec2


# --- Main --- #
# ------------ #


# absolute
@handle_frontend_test(
    fn_tree="jax.numpy.absolute",
    aliases=["jax.numpy.abs"],
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("signed_integer"),
    ),
    test_with_out=st.just(False),
)
def test_jax_absolute(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# add
@handle_frontend_test(
    fn_tree="jax.numpy.add",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_jax_add(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=x[0],
        x2=x[0],
    )


# angle
@handle_frontend_test(
    fn_tree="jax.numpy.angle",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=["float64"],
        min_value=-5,
        max_value=5,
        max_dim_size=5,
        max_num_dims=5,
        min_dim_size=1,
        min_num_dims=1,
        allow_inf=False,
        allow_nan=False,
    ),
    deg=st.booleans(),
    test_with_out=st.just(False),
)
def test_jax_angle(
    *,
    dtype_and_x,
    deg,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, z = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        z=z[0],
        deg=deg,
    )


# arccos
@handle_frontend_test(
    fn_tree="jax.numpy.arccos",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    test_with_out=st.just(False),
)
def test_jax_arccos(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# arccosh
@handle_frontend_test(
    fn_tree="jax.numpy.arccosh",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    test_with_out=st.just(False),
)
def test_jax_arccosh(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# arcsin
@handle_frontend_test(
    fn_tree="jax.numpy.arcsin",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=4,
        small_abs_safety_factor=4,
    ),
)
def test_jax_arcsin(
    dtype_and_x,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        rtol=1e-2,
        atol=1e-2,
    )


# arcsinh
@handle_frontend_test(
    fn_tree="jax.numpy.arcsinh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=4,
        small_abs_safety_factor=4,
    ),
    test_with_out=st.just(False),
)
def test_jax_arcsinh(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# arctan
@handle_frontend_test(
    fn_tree="jax.numpy.arctan",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_jax_arctan(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# arctan2
@handle_frontend_test(
    fn_tree="jax.numpy.arctan2",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
    ),
)
def test_jax_arctan2(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=x[0],
        x2=x[1],
    )


# arctanh
@handle_frontend_test(
    fn_tree="jax.numpy.arctanh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=0,
    ),
    test_with_out=st.just(False),
)
def test_jax_arctanh(
    dtype_and_x,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        x=x[0],
    )


# around
@handle_frontend_test(
    fn_tree="jax.numpy.around",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
    decimals=st.integers(min_value=0, max_value=5),
)
def test_jax_around(
    *,
    dtype_and_x,
    decimals,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        decimals=decimals,
    )


# cbrt
@handle_frontend_test(
    fn_tree="jax.numpy.cbrt",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_with_out=st.just(False),
)
def test_jax_cbrt(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-2,
        atol=1e-2,
        x=x[0],
    )


# ceil
@handle_frontend_test(
    fn_tree="jax.numpy.ceil",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    test_with_out=st.just(False),
)
def test_jax_ceil(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# clip
@handle_frontend_test(
    fn_tree="jax.numpy.clip",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_integer"),
        min_value=-1e3,
        max_value=1e3,
        max_dim_size=10,
        max_num_dims=4,
        min_dim_size=1,
        min_num_dims=1,
    ),
    a_min=st.integers(min_value=0, max_value=5),
    a_max=st.integers(min_value=5, max_value=10),
)
def test_jax_clip(
    *,
    dtype_and_x,
    a_min,
    a_max,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        a_min=a_min,
        a_max=a_max,
    )


# conj
@handle_frontend_test(
    fn_tree="jax.numpy.conj",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_jax_conj(
    *,
    dtype_and_x,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# TODO: uncomment with multiversion pipeline (deprecated since 0.4.12)
# @handle_frontend_test(
#     fn_tree="jax.numpy.product",
#     dtype_x_axis_dtype_where=_get_castable_dtypes_values(use_where=True),
#     keepdims=st.booleans(),
#     initial=st.one_of(st.floats(min_value=-100, max_value=100)),
#     promote_integers=st.booleans(),
# )
# def test_jax_product(
#     dtype_x_axis_dtype_where,
#     keepdims,
#     initial,
#     promote_integers,
#     frontend,
#     test_flags,
#     fn_tree,
#     on_device,
# ):
#     input_dtypes, x, axis, dtype, where = dtype_x_axis_dtype_where
#     if ivy.current_backend_str() == "torch":
#         assume(not test_flags.as_variable[0])
#     where, input_dtypes, test_flags = np_frontend_helpers.
#      handle_where_and_array_bools(
#         where=where,
#         input_dtype=input_dtypes,
#         test_flags=test_flags,
#     )
#     helpers.test_frontend_function(
#         input_dtypes=input_dtypes,
#         frontend=frontend,
#         test_flags=test_flags,
#         fn_tree=fn_tree,
#         on_device=on_device,
#         a=x[0],
#         axis=axis,
#         dtype=dtype,
#         keepdims=keepdims,
#         initial=initial,
#         where=where,
#         promote_integers=promote_integers,
#     )


# conjugate
@handle_frontend_test(
    fn_tree="jax.numpy.conjugate",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
)
def test_jax_conjugate(
    *,
    dtype_and_x,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# convolve
@handle_frontend_test(
    fn_tree="jax.numpy.convolve",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_num_dims=1,
        max_num_dims=1,
        min_value=-1e04,
        max_value=1e04,
        shared_dtype=True,
    ),
    mode=st.sampled_from(["valid", "same", "full"]),
)
def test_jax_convolve(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
    mode,
):
    input_dtype, x = dtype_and_x
    assume("float16" not in input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        rtol=1e-2,
        atol=1e-2,
        on_device=on_device,
        a=x[0],
        v=x[1],
        mode=mode,
        precision=None,
    )


# copysign
@handle_frontend_test(
    fn_tree="jax.numpy.copysign",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-100,
        max_value=100,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_jax_copysign(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=x[0],
        x2=x[0],
    )


@handle_frontend_test(
    fn_tree="jax.numpy.cos",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_jax_cos(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# cosh
@handle_frontend_test(
    fn_tree="jax.numpy.cosh",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    test_with_out=st.just(False),
)
def test_jax_cosh(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# deg2rad
@handle_frontend_test(
    fn_tree="jax.numpy.deg2rad",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_with_out=st.just(False),
)
def test_jax_deg2rad(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# degrees
@handle_frontend_test(
    fn_tree="jax.numpy.degrees",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_with_out=st.just(False),
)
def test_jax_degrees(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="jax.numpy.diff",
    dtype_and_x=_get_dtype_input_and_vector(),
    n=helpers.ints(
        min_value=0,
        max_value=10,
    ),
    axis=helpers.ints(
        min_value=-1,
        max_value=10,
    ),
)
def test_jax_diff(
    *,
    dtype_and_x,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    n,
    axis,
):
    input_dtype, x = dtype_and_x
    axis = min(axis, x[0].ndim - 1)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        n=n,
        axis=axis,
        prepend=None,
        append=None,
    )


# divide
@handle_frontend_test(
    fn_tree="jax.numpy.divide",
    aliases=["jax.numpy.true_divide"],
    dtype_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        allow_inf=False,
        large_abs_safety_factor=2,
        small_abs_safety_factor=2,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_jax_divide(
    *,
    dtype_values,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
):
    input_dtype, x = dtype_values
    assume(not np.any(np.isclose(x[1], 0)))
    if ivy.current_backend_str() == "paddle":
        atol, rtol = 1e-2, 1e-2
    else:
        atol, rtol = 1e-5, 1e-5
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        a=x[0],
        b=x[1],
        atol=atol,
        rtol=rtol,
    )


# divmod
@handle_frontend_test(
    fn_tree="jax.numpy.divmod",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        allow_inf=False,
        large_abs_safety_factor=2,
        safety_factor_scale="linear",
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_jax_divmod(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x[1], 0)) and "bfloat16" not in input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=x[0],
        x2=x[1],
        atol=1,
    )


@handle_frontend_test(
    fn_tree="jax.numpy.dot",
    dtype_x_y=_get_dtype_input_and_vectors(),
    test_with_out=st.just(False),
)
def test_jax_dot(
    *,
    dtype_x_y,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x, y = dtype_x_y
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        rtol=1e-01,
        atol=1e-01,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x,
        b=y,
        precision=None,
    )


# ediff1d
@handle_frontend_test(
    fn_tree="jax.numpy.ediff1d",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"), min_num_dims=1, max_num_dims=1
    ),
    to_end=helpers.ints(
        min_value=-1,
        max_value=10,
    ),
    to_begin=helpers.ints(
        min_value=-1,
        max_value=10,
    ),
)
def test_jax_ediff1d(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
    to_end,
    to_begin,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        test_flags=test_flags,
        ary=x[0],
        to_end=to_end,
        to_begin=to_begin,
    )


# einsum_path
# For the optimize parameter boolean values are not added to the samples for testing
# as it seems that Jax einsum_path function currently fails when True or False is passed
# as optimize values. Jax einsum_path function calls opt_einsum.contract_path function,
# and it seems that there is an open bug on their repository for boolean values.
# Please see link to the bug https://github.com/dgasmith/opt_einsum/issues/219
@handle_frontend_test(
    fn_tree="jax.numpy.einsum_path",
    eq_n_op_n_shp=helpers.einsum_helper(),
    dtype=helpers.get_dtypes("numeric", full=False),
    test_with_out=st.just(False),
    optimize=st.sampled_from(["greedy", "optimal"]),
)
def test_jax_einsum_path(
    *,
    eq_n_op_n_shp,
    dtype,
    on_device,
    fn_tree,
    backend_fw,
    frontend,
    test_flags,
    optimize,
):
    eq, operands, dtypes = eq_n_op_n_shp
    kw = {}
    for i, x_ in enumerate(operands):
        dtype = dtypes[i][0]
        kw[f"x{i}"] = np.array(x_).astype(dtype)
    test_flags.num_positional_args = len(operands) + 1
    ret, ret_gt = helpers.test_frontend_function(
        input_dtypes=dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        subscripts=eq,
        **kw,
        optimize=optimize,
    )
    assert len(ret[0]) == len(ret_gt[0])
    assert all(x == y for x, y in zip(ret[0], ret_gt[0]))
    assert ret[1] == str(ret_gt[1])


# exp
@handle_frontend_test(
    fn_tree="jax.numpy.exp",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    test_with_out=st.just(False),
)
def test_jax_exp(
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# exp2
@handle_frontend_test(
    fn_tree="jax.numpy.exp2",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-10,
        max_value=10,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
    ),
    test_with_out=st.just(False),
)
def test_jax_exp2(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        rtol=1e-01,
        atol=1e-02,
    )


# expm1
@handle_frontend_test(
    fn_tree="jax.numpy.expm1",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_with_out=st.just(False),
)
def test_jax_expm1(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# fabs
@handle_frontend_test(
    fn_tree="jax.numpy.fabs",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_jax_fabs(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# fix
@handle_frontend_test(
    fn_tree="jax.numpy.fix",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", index=2),
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
    ),
)
def test_jax_fix(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        x=x[0],
    )


# float_power
@handle_frontend_test(
    fn_tree="jax.numpy.float_power",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-10,
        max_value=10,
        num_arrays=2,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
    ),
    test_with_out=st.just(False),
)
def test_jax_float_power(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=x[0],
        x2=x[1],
    )


# floor
@handle_frontend_test(
    fn_tree="jax.numpy.floor",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    test_with_out=st.just(False),
)
def test_jax_floor(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# floor_divide
@handle_frontend_test(
    fn_tree="jax.numpy.floor_divide",
    dtype_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
        min_value=-10.0,
        max_value=10.0,
        large_abs_safety_factor=2,
        small_abs_safety_factor=2,
        safety_factor_scale="linear",
    ),
)
def test_jax_floor_divide(
    *,
    dtype_values,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
    test_flags,
):
    input_dtype, x = dtype_values
    # Making sure division by zero doesn't occur
    assume(not np.any(np.isclose(x[1], 0)))
    # Absolute tolerance is 1,
    # due to flooring can cause absolute error of 1 due to precision
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        on_device=on_device,
        test_flags=test_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        x1=x[0],
        x2=x[1],
        atol=1,
    )


# fmax
@handle_frontend_test(
    fn_tree="jax.numpy.fmax",
    dtype_and_inputs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        min_value=-np.inf,
        max_value=np.inf,
    ),
    test_with_out=st.just(False),
)
def test_jax_fmax(
    *,
    dtype_and_inputs,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, inputs = dtype_and_inputs
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=inputs[0],
        x2=inputs[1],
    )


# fmin
@handle_frontend_test(
    fn_tree="jax.numpy.fmin",
    dtype_and_inputs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        min_value=-np.inf,
        max_value=np.inf,
    ),
)
def test_jax_fmin(
    *,
    dtype_and_inputs,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, inputs = dtype_and_inputs
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=inputs[0],
        x2=inputs[1],
    )


# fmod
@handle_frontend_test(
    fn_tree="jax.numpy.fmod",
    dtype_and_inputs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=1.5,
        safety_factor_scale="log",
    ),
    test_with_out=st.just(False),
)
def test_jax_fmod(
    *,
    dtype_and_inputs,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_inputs
    assume(not np.any(np.isclose(x[1], 0)))
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=x[0],
        x2=x[1],
    )


# frexp
@handle_frontend_test(
    fn_tree="jax.numpy.frexp",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=1,
        max_value=100,
    ),
)
def test_jax_frexp(
    *,
    dtype_and_x,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# gcd
@handle_frontend_test(
    fn_tree="jax.numpy.gcd",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        min_value=-100,
        max_value=100,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
        num_arrays=2,
    ).filter(lambda x: all(dtype != "uint64" for dtype in x[0])),
    test_with_out=st.just(False),
)
def test_jax_gcd(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=x[0],
        x2=x[1],
    )


# gradient
@handle_frontend_test(
    fn_tree="jax.numpy.gradient",
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=("float32", "float16", "float64"),
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=2,
        max_dim_size=4,
        valid_axis=True,
        force_int_axis=True,
    ),
    varargs=helpers.ints(
        min_value=-3,
        max_value=3,
    ),
)
def test_jax_gradient(
    dtype_input_axis,
    varargs,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x, axis = dtype_input_axis
    test_flags.num_positional_args = 2
    kw = {}
    kw["varargs"] = varargs
    kw["axis"] = axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        f=x[0],
        **kw,
    )


# heaviside
@handle_frontend_test(
    fn_tree="jax.numpy.heaviside",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-100,
        max_value=100,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
        num_arrays=2,
    ),
    test_with_out=st.just(False),
)
def test_jax_heaviside(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=x[0],
        x2=x[0],
    )


# hypot
@handle_frontend_test(
    fn_tree="jax.numpy.hypot",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
        min_value=-100,
        max_value=100,
        min_num_dims=1,
        max_num_dims=3,
    ),
)
def test_jax_hypot(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        atol=1e-2,
        x1=x[0],
        x2=x[1],
        backend_to_test=backend_fw,
    )


# i0
@handle_frontend_test(
    fn_tree="jax.numpy.i0",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-10,
        max_value=10,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
    ),
    test_with_out=st.just(False),
)
def test_jax_i0(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# imag
@handle_frontend_test(
    fn_tree="jax.numpy.imag",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_complex"),
        min_value=-20,
        max_value=20,
    ),
    test_with_out=st.just(False),
)
def test_jax_imag(
    *,
    dtype_and_x,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-5,
        atol=1e-5,
        val=x[0],
    )


# inner
@handle_frontend_test(
    fn_tree="jax.numpy.inner",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_value=-10,
        max_value=10,
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_jax_inner(
    *,
    dtype_and_x,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    input_dtypes, xs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        test_flags=test_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        a=xs[0],
        b=xs[1],
    )


@handle_frontend_test(
    fn_tree="jax.numpy.interp",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        max_num_dims=1,
    ),
    dtype_and_xp_fp=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_num_dims=1,
        max_num_dims=1,
    ),
    left=st.one_of(st.floats(min_value=-1e04, max_value=1e04), st.just(np.nan)),
    right=st.one_of(st.floats(min_value=-1e04, max_value=1e04), st.just(np.nan)),
    test_with_out=st.just(False),
)
def test_jax_interp(
    *,
    dtype_and_x,
    dtype_and_xp_fp,
    left,
    right,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    input_dtype2, xp_fp = dtype_and_xp_fp
    xp = xp_fp[0]
    fp = xp_fp[1]
    helpers.test_frontend_function(
        input_dtypes=[input_dtype, input_dtype2],
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        xp=xp,
        fp=fp,
        left=left,
        right=right,
    )


# kron
@handle_frontend_test(
    fn_tree="jax.numpy.kron",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=1,
        max_dim_size=3,
        num_arrays=2,
    ),
    test_with_out=st.just(False),
)
def test_jax_kron(
    *,
    dtype_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        b=x[1],
    )


# lcm
@handle_frontend_test(
    fn_tree="jax.numpy.lcm",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        min_num_dims=1,
        max_num_dims=3,
        min_value=-100,
        max_value=100,
        allow_nan=False,
        small_abs_safety_factor=2,
        large_abs_safety_factor=2,
        safety_factor_scale="log",
    ),
    test_with_out=st.just(False),
)
def test_jax_lcm(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    value_test = True
    # Skip Tensorflow backend value test for lcm
    # https://github.com/tensorflow/tensorflow/issues/58955
    if ivy.current_backend_str() == "tensorflow":
        value_test = False
    if ivy.current_backend_str() in ("jax", "numpy"):
        assume(input_dtype[0] != "uint64" and input_dtype[1] != "uint64")
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=x[0],
        x2=x[1],
        test_values=value_test,
    )


# ldexp
@handle_frontend_test(
    fn_tree="jax.numpy.ldexp",
    dtype_and_x=ldexp_args(),
)
def test_jax_ldexp(
    *,
    dtype_and_x,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=x[0],
        x2=x[1],
    )


# log
@handle_frontend_test(
    fn_tree="jax.numpy.log",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        max_num_dims=3,
        min_value=-100,
        max_value=100,
    ),
    test_with_out=st.just(False),
)
def test_jax_log(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-01,
        atol=1e-02,
        x=x[0],
    )


# log10
@handle_frontend_test(
    fn_tree="jax.numpy.log10",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        max_num_dims=3,
        min_value=-100,
        max_value=100,
    ),
    test_with_out=st.just(False),
)
def test_jax_log10(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-01,
        atol=1e-02,
        x=x[0],
    )


# log1p
@handle_frontend_test(
    fn_tree="jax.numpy.log1p",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=2,
        small_abs_safety_factor=2,
        safety_factor_scale="log",
    ),
)
def test_jax_log1p(
    dtype_and_x,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# log2
@handle_frontend_test(
    fn_tree="jax.numpy.log2",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_with_out=st.just(False),
)
def test_jax_log2(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x[0], 0)))
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-2,
        x=x[0],
    )


# logaddexp
@handle_frontend_test(
    fn_tree="jax.numpy.logaddexp",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_num_dims=1,
        max_num_dims=3,
        min_value=-100,
        max_value=100,
        allow_nan=False,
        large_abs_safety_factor=2,
        small_abs_safety_factor=2,
        safety_factor_scale="log",
    ),
    test_with_out=st.just(False),
)
def test_jax_logaddexp(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-01,
        atol=1e-02,
        x1=x[0],
        x2=x[1],
    )


# logaddexp2
@handle_frontend_test(
    fn_tree="jax.numpy.logaddexp2",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
        min_num_dims=1,
        max_num_dims=3,
        min_value=-100,
        max_value=100,
        allow_nan=False,
        large_abs_safety_factor=2,
        small_abs_safety_factor=2,
        safety_factor_scale="log",
    ),
    test_with_out=st.just(False),
)
def test_jax_logaddexp2(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-01,
        atol=1e-02,
        x1=x[0],
        x2=x[1],
    )


# matmul
@handle_frontend_test(
    fn_tree="jax.numpy.matmul",
    dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(
        arr_func=[_get_first_matrix_and_dtype, _get_second_matrix_and_dtype],
    ),
)
def test_jax_matmul(
    dtypes_values_casting,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
):
    dtypes, x, casting, dtype = dtypes_values_casting
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        b=x[1],
        precision=None,
    )


# maximum
@handle_frontend_test(
    fn_tree="jax.numpy.maximum",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
    test_with_out=st.just(False),
)
def test_jax_maximum(
    dtype_and_x,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        x1=x[0],
        x2=x[1],
    )


# minimum
@handle_frontend_test(
    fn_tree="jax.numpy.minimum",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
    test_with_out=st.just(False),
)
def test_jax_minimum(
    dtype_and_x,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        x1=x[0],
        x2=x[1],
    )


# mod
@handle_frontend_test(
    fn_tree="jax.numpy.mod",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2,
        small_abs_safety_factor=2,
        safety_factor_scale="log",
    ),
    test_with_out=st.just(False),
)
def test_jax_mod(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x[1], 0)) and "bfloat16" not in input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=x[0],
        x2=x[1],
    )


# modf
@handle_frontend_test(
    fn_tree="jax.numpy.modf",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_integer"),
        large_abs_safety_factor=2,
        small_abs_safety_factor=2,
        safety_factor_scale="log",
    ),
    test_with_out=st.just(False),
)
def test_jax_modf(
    dtype_and_x,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# multiply
@handle_frontend_test(
    fn_tree="jax.numpy.multiply",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
    test_with_out=st.just(False),
)
def test_jax_multiply(
    dtype_and_x,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        x1=x[0],
        x2=x[1],
    )


# nan_to_num
@handle_frontend_test(
    fn_tree="jax.numpy.nan_to_num",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        max_num_dims=3,
        min_value=-100,
        max_value=100,
        allow_nan=True,
        allow_inf=True,
    ),
    copy=st.booleans(),
    nan=st.floats(min_value=0.0, max_value=100),
    posinf=st.floats(min_value=5e100, max_value=5e100),
    neginf=st.floats(min_value=-5e100, max_value=-5e100),
    test_with_out=st.just(False),
    test_with_copy=st.just(True),
)
def test_jax_nan_to_num(
    *,
    dtype_and_x,
    copy,
    nan,
    posinf,
    neginf,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        copy=copy,
        nan=nan,
        posinf=posinf,
        neginf=neginf,
    )


# negative
@handle_frontend_test(
    fn_tree="jax.numpy.negative",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"), min_num_dims=1
    ),
    test_with_out=st.just(False),
)
def test_jax_negative(
    dtype_and_x,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# nextafter
@handle_frontend_test(
    fn_tree="jax.numpy.nextafter",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-100,
        max_value=100,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_jax_nextafter(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=x[0],
        x2=x[0],
    )


# outer
@handle_frontend_test(
    fn_tree="jax.numpy.outer",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        min_value=-10,
        max_value=10,
        min_num_dims=1,
        max_num_dims=1,
        shared_dtype=True,
    ),
)
def test_jax_outer(
    *,
    dtype_and_x,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    input_dtypes, xs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        test_flags=test_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        a=xs[0],
        b=xs[1],
    )


# poly
@handle_frontend_test(
    fn_tree="jax.numpy.poly",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=1,
        min_num_dims=1,
        max_num_dims=1,
        min_value=-1e04,
        max_value=1e04,
    ),
)
def test_jax_poly(
    *,
    dtype_and_x,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    assume("float16" not in input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        seq_of_zeros=x[0],
        atol=1e-05,
        rtol=1e-03,
    )


# polyadd
@handle_frontend_test(
    fn_tree="jax.numpy.polyadd",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
    ),
)
def test_jax_polyadd(
    *,
    dtype_and_x,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    assume("float16" not in input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        a1=x[0],
        a2=x[1],
    )


# polyder
@handle_frontend_test(
    fn_tree="jax.numpy.polyder",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=1,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=1,
    ),
    m=st.integers(min_value=0, max_value=10),
)
def test_jax_polyder(
    *,
    dtype_and_x,
    m,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        p=x[0],
        m=m,
    )


# polydiv
@handle_frontend_test(
    fn_tree="jax.numpy.polydiv",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_num_dims=1,
        min_dim_size=1,
        max_num_dims=1,
        min_value=-1e04,
        max_value=1e04,
    ),
)
def test_jax_polydiv(
    *,
    dtype_and_x,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    assume("float16" not in input_dtype)
    # TODO: remove asumme when the decorator works
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        u=x[0],
        v=x[1],
        rtol=1e-01,
        atol=1e-02,
    )


@handle_frontend_test(
    fn_tree="jax.numpy.polyint",
    dtype_and_x_and_k=_get_array_values_m_and_k(),
)
def test_jax_polyint(
    *,
    dtype_and_x_and_k,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    input_dtype, x, m, k = dtype_and_x_and_k
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        p=x[0],
        m=m,
        k=k,
    )


# polymul
@handle_frontend_test(
    fn_tree="jax.numpy.polymul",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
        large_abs_safety_factor=2,
        small_abs_safety_factor=2,
        safety_factor_scale="log",
    ),
    trim=st.booleans(),
)
def test_jax_polymul(
    *,
    dtype_and_x,
    trim,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    assume("float16" not in input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        a1=x[0],
        a2=x[1],
        trim_leading_zeros=trim,
        atol=1e-01,
        rtol=1e-01,
    )


# polysub
@handle_frontend_test(
    fn_tree="jax.numpy.polysub",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
        min_value=-1e04,
        max_value=1e04,
    ),
)
def test_jax_polysub(
    *,
    dtype_and_x,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    assume("float16" not in input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        a1=x[0],
        a2=x[1],
    )


# positive
@handle_frontend_test(
    fn_tree="jax.numpy.positive",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"), min_num_dims=1
    ),
    test_with_out=st.just(False),
)
def test_jax_positive(
    dtype_and_x,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# power
@handle_frontend_test(
    fn_tree="jax.numpy.power",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
    ),
    test_with_out=st.just(False),
)
def test_jax_power(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=x[0],
        x2=x[1],
    )


# rad2deg
@handle_frontend_test(
    fn_tree="jax.numpy.rad2deg",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"), min_num_dims=1
    ),
    test_with_out=st.just(False),
)
def test_jax_rad2deg(
    dtype_and_x,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# radians
@handle_frontend_test(
    fn_tree="jax.numpy.radians",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_with_out=st.just(False),
)
def test_jax_radians(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# real
@handle_frontend_test(
    fn_tree="jax.numpy.real",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("real_and_complex"),
    ),
)
def test_jax_real(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=True,
        val=x[0],
    )


# reciprocal
@handle_frontend_test(
    fn_tree="jax.numpy.reciprocal",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        small_abs_safety_factor=4,
        large_abs_safety_factor=4,
        safety_factor_scale="log",
        num_arrays=1,
    ),
)
def test_jax_reciprocal(
    *,
    dtype_and_x,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# remainder
@handle_frontend_test(
    fn_tree="jax.numpy.remainder",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=6,
        small_abs_safety_factor=6,
        safety_factor_scale="log",
    ),
    test_with_out=st.just(False),
)
def test_jax_remainder(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x

    assume(not np.any(np.isclose(x[1], 0)))

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=x[0],
        x2=x[1],
        rtol=1e-2,
        atol=1e-2,
    )


# round
@handle_frontend_test(
    fn_tree="jax.numpy.round",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    decimals=st.integers(min_value=0, max_value=5),
)
def test_jax_round(
    *,
    dtype_and_x,
    decimals,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        decimals=decimals,
    )


# sign
@handle_frontend_test(
    fn_tree="jax.numpy.sign",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"), min_num_dims=1
    ),
    test_with_out=st.just(False),
)
def test_jax_sign(
    *,
    dtype_and_x,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
    )


# signbit
@handle_frontend_test(
    fn_tree="jax.numpy.signbit",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_jax_signbit(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="jax.numpy.sin",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    test_with_out=st.just(False),
)
def test_jax_sin(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# sinc
@handle_frontend_test(
    fn_tree="jax.numpy.sinc",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        max_num_dims=3,
        min_value=-100,
        max_value=100,
    ),
)
def test_jax_sinc(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-01,
        atol=1e-02,
        x=x[0],
    )


# sinh
@handle_frontend_test(
    fn_tree="jax.numpy.sinh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=4,
        small_abs_safety_factor=4,
    ),
    test_with_out=st.just(False),
)
def test_jax_sinh(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# sqrt
@handle_frontend_test(
    fn_tree="jax.numpy.sqrt",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    test_with_out=st.just(False),
)
def test_jax_sqrt(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# square
@handle_frontend_test(
    fn_tree="jax.numpy.square",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    test_with_out=st.just(False),
)
def test_jax_square(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# subtract
@handle_frontend_test(
    fn_tree="jax.numpy.subtract",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_jax_subtract(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=x[0],
        x2=x[0],
    )


# tan
@handle_frontend_test(
    fn_tree="jax.numpy.tan",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    test_with_out=st.just(False),
)
def test_jax_tan(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# tanh
@handle_frontend_test(
    fn_tree="jax.numpy.tanh",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    test_with_out=st.just(False),
)
def test_jax_tanh(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# tensordot
@handle_frontend_test(
    fn_tree="jax.numpy.tensordot",
    dtype_values_and_axes=_get_dtype_value1_value2_axis_for_tensordot(
        helpers.get_dtypes(kind="numeric")
    ),
    test_with_out=st.just(False),
)
def test_jax_tensordot(
    dtype_values_and_axes,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
):
    dtype, a, b, axes = dtype_values_and_axes
    if ivy.current_backend_str() == "torch":
        atol = 1e-3
    else:
        atol = 1e-6
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        a=a,
        b=b,
        atol=atol,
        axes=axes,
    )


# trace
@handle_frontend_test(
    fn_tree="jax.numpy.trace",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=1,
        max_dim_size=10,
        large_abs_safety_factor=24,
        small_abs_safety_factor=24,
        safety_factor_scale="log",
    ),
    offset=st.integers(min_value=0, max_value=0),
    axis1=st.integers(min_value=0, max_value=0),
    axis2=st.integers(min_value=1, max_value=1),
    test_with_out=st.just(False),
)
def test_jax_trace(
    *,
    dtype_and_x,
    offset,
    axis1,
    axis2,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-1,
        atol=1e-1,
        a=x[0],
        offset=offset,
        axis1=axis1,
        axis2=axis2,
    )


@handle_frontend_test(
    fn_tree="jax.numpy.trapz",
    dtype_x_axis_rand_either=_either_x_dx(),
    test_with_out=st.just(False),
)
def test_jax_trapz(
    *,
    dtype_x_axis_rand_either,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    dtype_values_axis, rand, either_x_dx = dtype_x_axis_rand_either
    input_dtype, y, axis = dtype_values_axis
    if rand == 0:
        dtype_x, x = either_x_dx
        x = np.asarray(x, dtype=dtype_x)
        dx = None
    else:
        x = None
        dx = either_x_dx
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-2,
        atol=1e-2,
        y=y[0],
        x=x,
        dx=dx,
        axis=axis,
    )


# trunc
@handle_frontend_test(
    fn_tree="jax.numpy.trunc",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    test_with_out=st.just(False),
)
def test_jax_trunc(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# vdot
@handle_frontend_test(
    fn_tree="jax.numpy.vdot",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
    test_with_out=st.just(False),
)
def test_jax_vdot(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        a=x[0],
        b=x[1],
    )
