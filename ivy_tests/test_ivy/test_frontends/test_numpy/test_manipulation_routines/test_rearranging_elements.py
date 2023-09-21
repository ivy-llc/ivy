from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# --- Helpers --- #
# --------------- #


@st.composite
def _dtype_x_bounded_axis(draw, **kwargs):
    dtype, x, shape = draw(helpers.dtype_and_values(**kwargs, ret_shape=True))
    axis = draw(helpers.ints(min_value=0, max_value=len(shape) - 1))
    return dtype, x, axis


@st.composite
def _get_dtype_values_k_axes_for_rot90(
    draw,
    available_dtypes,
    min_value=None,
    max_value=None,
    allow_inf=False,
    exclude_min=False,
    exclude_max=False,
    min_num_dims=1,
    max_num_dims=10,
    min_dim_size=1,
    max_dim_size=10,
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
    k = draw(helpers.ints(min_value=-4, max_value=4))
    axes = tuple(
        draw(
            st.lists(
                helpers.ints(min_value=-(len(shape) - 1), max_value=len(shape) - 2),
                min_size=2,
                max_size=2,
                unique=True,
            ).filter(lambda axes: abs(axes[0] - axes[1]) != len(shape) - 1)
        )
    )
    dtype = draw(st.sampled_from(draw(available_dtypes)))
    values = draw(
        helpers.array_values(
            dtype=dtype,
            shape=shape,
            min_value=min_value,
            max_value=max_value,
            allow_inf=allow_inf,
            exclude_min=exclude_min,
            exclude_max=exclude_max,
            large_abs_safety_factor=72,
            small_abs_safety_factor=72,
            safety_factor_scale="log",
        )
    )
    return [dtype], values, k, axes


# --- Main --- #
# ------------ #


@handle_frontend_test(
    fn_tree="numpy.flip",
    dtype_x_axis=_dtype_x_bounded_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        min_dim_size=1,
    ),
    test_with_out=st.just(False),
)
def test_numpy_flip(
    *,
    dtype_x_axis,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        m=x[0],
        axis=axis,
    )


# fliplr
@handle_frontend_test(
    fn_tree="numpy.fliplr",
    dtype_and_m=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=2,
    ),
    test_with_out=st.just(False),
)
def test_numpy_fliplr(
    *,
    dtype_and_m,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, m = dtype_and_m
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        m=m[0],
    )


# flipud
@handle_frontend_test(
    fn_tree="numpy.flipud",
    dtype_and_m=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-100,
        max_value=100,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
    ),
    test_with_out=st.just(False),
)
def test_numpy_flipud(
    *,
    dtype_and_m,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, m = dtype_and_m
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        m=m[0],
    )


# roll
@handle_frontend_test(
    fn_tree="numpy.roll",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
        min_dim_size=2,
    ),
    shift=helpers.ints(min_value=1, max_value=10),
    axis=helpers.ints(min_value=-1, max_value=1),
    test_with_out=st.just(False),
)
def test_numpy_roll(
    *,
    dtype_and_x,
    shift,
    axis,
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
        a=x[0],
        shift=shift,
        axis=axis,
    )


# rot90
@handle_frontend_test(
    fn_tree="numpy.rot90",
    dtype_m_k_axes=_get_dtype_values_k_axes_for_rot90(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=3,
        max_num_dims=6,
        min_dim_size=1,
        max_dim_size=10,
    ),
    test_with_out=st.just(False),
)
def test_numpy_rot90(
    *,
    dtype_m_k_axes,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, m, k, axes = dtype_m_k_axes
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        m=m[0],
        k=k,
        axes=axes,
    )
