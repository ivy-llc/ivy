# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# --- Helpers --- #
# --------------- #


@st.composite
def _valid_dct(draw):
    dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
            max_value=65280,
            min_value=-65280,
            min_num_dims=1,
            max_num_dims=5,
            min_dim_size=2,
            max_dim_size=10,
            shared_dtype=True,
        )
    )
    dims_len = len(x[0].shape)
    n = draw(st.sampled_from([None, "int"]))
    axis = draw(helpers.ints(min_value=-dims_len, max_value=dims_len))
    norm = draw(st.sampled_from([None, "ortho"]))
    type = draw(helpers.ints(min_value=1, max_value=4))
    if n == "int":
        n = draw(helpers.ints(min_value=1, max_value=20))
        if n <= 1 and type == 1:
            n = 2
    if norm == "ortho" and type == 1:
        norm = None
    return dtype, x, type, n, axis, norm


@st.composite
def _valid_idct(draw):
    dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=["float32", "float64"],
            max_value=65280,
            min_value=-65280,
            min_num_dims=1,
            max_num_dims=5,
            min_dim_size=2,
            max_dim_size=10,
            shared_dtype=True,
        )
    )
    n = None
    axis = -1
    norm = draw(st.sampled_from([None, "ortho"]))
    type = draw(helpers.ints(min_value=1, max_value=4))
    if norm == "ortho" and type == 1:
        norm = None
    return dtype, x, type, n, axis, norm


# Helpers


@st.composite
def _x_and_fft(draw, dtypes):
    min_fft_points = 2
    dtype = draw(dtypes)
    x_dim = draw(
        helpers.get_shape(
            min_dim_size=2, max_dim_size=100, min_num_dims=1, max_num_dims=4
        )
    )
    x = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=tuple(x_dim),
        )
    )
    dim = draw(
        helpers.get_axis(shape=x_dim, allow_neg=True, allow_none=False, max_size=1)
    )
    norm = draw(st.sampled_from(["backward", "forward", "ortho"]))
    n = draw(st.integers(min_fft_points, 256))
    return dtype, x, dim, norm, n


@st.composite
def _x_and_fft2(draw):
    min_fft2_points = 2
    dtype = draw(helpers.get_dtypes("float_and_complex", full=False))
    x, dim = draw(
        helpers.arrays_and_axes(
            available_dtypes=dtype[0],
            min_dim_size=2,
            max_dim_size=100,
            min_num_dims=2,
            max_num_dims=4,
        ),
    )
    s = (
        draw(st.integers(min_fft2_points, 256)),
        draw(st.integers(min_fft2_points, 256)),
    )
    norm = draw(st.sampled_from(["backward", "forward", "ortho"]))
    return dtype, x, s, dim, norm


@st.composite
def _x_and_ifft(draw):
    min_fft_points = 2
    dtype = draw(helpers.get_dtypes("complex"))
    x_dim = draw(
        helpers.get_shape(
            min_dim_size=2, max_dim_size=100, min_num_dims=1, max_num_dims=4
        )
    )
    x = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=tuple(x_dim),
            min_value=-1e-10,
            max_value=1e10,
        )
    )
    dim = draw(st.integers(1 - len(list(x_dim)), len(list(x_dim)) - 1))
    norm = draw(st.sampled_from(["backward", "forward", "ortho"]))
    n = draw(st.integers(min_fft_points, 256))
    return dtype, x, dim, norm, n


@st.composite
def _x_and_ifftn(draw):
    _x_and_ifftn = draw(_x_and_fft2())
    workers = draw(st.integers(1, 4))
    return _x_and_ifftn + (workers,)


@st.composite
def _x_and_rfftn(draw):
    min_rfftn_points = 2
    dtype = draw(helpers.get_dtypes("float"))
    x_dim = draw(
        helpers.get_shape(
            min_dim_size=2, max_dim_size=100, min_num_dims=1, max_num_dims=3
        )
    )
    x = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=tuple(x_dim),
            min_value=-1e10,
            max_value=1e10,
            large_abs_safety_factor=2.5,
            small_abs_safety_factor=2.5,
            safety_factor_scale="log",
        )
    )
    axes = draw(
        st.lists(
            st.integers(0, len(x_dim) - 1), min_size=1, max_size=len(x_dim), unique=True
        )
    )
    s = draw(
        st.lists(
            st.integers(min_rfftn_points, 256), min_size=len(axes), max_size=len(axes)
        )
    )
    norm = draw(st.sampled_from(["backward", "forward", "ortho"]))
    return dtype, x, s, axes, norm


# --- Main --- #
# ------------ #


# dct
@handle_frontend_test(
    fn_tree="scipy.fft.dct",
    dtype_x_and_args=_valid_dct(),
    test_with_out=st.just(False),
)
def test_scipy_dct(
    dtype_x_and_args,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    backend_fw,
):
    input_dtype, x, _type, n, axis, norm = dtype_x_and_args
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        type=_type,
        n=n,
        axis=axis,
        norm=norm,
        rtol_=1e-3,
        atol_=1e-1,
    )


# Tests


# fft
@handle_frontend_test(
    fn_tree="scipy.fft.fft",
    d_x_d_n_n=_x_and_fft(helpers.get_dtypes("complex")),
    test_with_out=st.just(False),
)
def test_scipy_fft(
    d_x_d_n_n,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    backend_fw,
):
    dtype, x, dim, norm, n = d_x_d_n_n
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x,
        dim=dim,
        norm=norm,
        n=n,
    )


# fft2
@handle_frontend_test(
    fn_tree="scipy.fft.fft2",
    d_x_d_s_n=_x_and_fft2(),
    test_with_out=st.just(False),
)
def test_scipy_fft2(
    d_x_d_s_n,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    backend_fw,
):
    dtype, x, s, ax, norm = d_x_d_s_n
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        s=s,
        axes=ax,
        norm=norm,
    )


# idct
@handle_frontend_test(
    fn_tree="scipy.fft.idct",
    dtype_x_and_args=_valid_idct(),
    test_with_out=st.just(False),
)
def test_scipy_idct(
    dtype_x_and_args,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    backend_fw,
):
    input_dtype, x, _type, n, axis, norm = dtype_x_and_args
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        type=_type,
        n=n,
        axis=axis,
        norm=norm,
        rtol_=1e-3,
        atol_=1e-1,
    )


# ifft
@handle_frontend_test(
    fn_tree="scipy.fft.ifft",
    d_x_d_n_n=_x_and_ifft(),
    test_with_out=st.just(False),
)
def test_scipy_ifft(
    d_x_d_n_n,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    backend_fw,
):
    dtype, x, dim, norm, n = d_x_d_n_n
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x,
        axis=dim,
        norm=norm,
        n=n,
    )


# ifftn
@handle_frontend_test(
    fn_tree="scipy.fft.ifftn",
    d_x_d_s_n_workers=_x_and_ifftn(),
    test_with_out=st.just(False),
)
def test_scipy_ifftn(
    d_x_d_s_n_workers,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    backend_fw,
):
    dtype, x, s, ax, norm, workers = d_x_d_s_n_workers
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        s=s,
        axes=ax,
        norm=norm,
        workers=workers,
    )


# rfftn
@handle_frontend_test(
    fn_tree="scipy.fft.rfftn",
    dtype_and_x=_x_and_rfftn(),
)
def test_scipy_rfftn(dtype_and_x, frontend, backend_fw, test_flags, fn_tree, on_device):
    dtype, x, s, axes, norm = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=True,
        x=x,
        s=s,
        axes=axes,
        norm=norm,
    )
