from hypothesis import strategies as st
from ivy_tests.test_ivy.test_frontends.test_numpy.test_creation_routines.test_from_shape_or_value import (  # noqa : E501
    _input_fill_and_dtype,
)

from ivy_tests.test_ivy.test_functional.test_core.test_creation import (
    _get_dtype_buffer_count_offset,
)

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# --- Helpers --- #
# --------------- #


@st.composite
def _get_dtype_and_range(draw):
    dim = draw(helpers.ints(min_value=2, max_value=5))
    dtype = draw(helpers.get_dtypes("float", index=1, full=False))
    start = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=(dim,),
            min_value=-50,
            max_value=0,
            large_abs_safety_factor=4,
            small_abs_safety_factor=4,
            safety_factor_scale="log",
        )
    )
    stop = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=(dim,),
            min_value=1,
            max_value=50,
            large_abs_safety_factor=4,
            small_abs_safety_factor=4,
            safety_factor_scale="log",
        )
    )
    return dtype * 2, start, stop


# --- Main --- #
# ------------ #


# arange
@handle_frontend_test(
    fn_tree="jax.numpy.arange",
    start=st.integers(min_value=-100, max_value=100),
    stop=st.integers(min_value=-100, max_value=100) | st.none(),
    step=st.integers(min_value=-100, max_value=100).filter(lambda x: x != 0),
    dtype=helpers.get_dtypes("numeric", full=False),
    test_with_out=st.just(False),
)
def test_jax_arange(
    *,
    start,
    stop,
    step,
    dtype,
    on_device,
    fn_tree,
    test_flags,
    frontend,
    backend_fw,
):
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        start=start,
        stop=stop,
        step=step,
        dtype=dtype[0],
    )


@handle_frontend_test(
    fn_tree="jax.numpy.array",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=st.integers(min_value=1, max_value=10),
        min_num_dims=0,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
        shared_dtype=True,
    ),
    as_list=st.booleans(),
    copy=st.booleans(),
    ndmin=helpers.ints(min_value=0, max_value=9),
    test_with_out=st.just(True),
    test_with_copy=st.just(True),
)
def test_jax_array(
    *,
    dtype_and_x,
    as_list,
    copy,
    ndmin,
    on_device,
    fn_tree,
    test_flags,
    frontend,
    backend_fw,
):
    input_dtype, x = dtype_and_x

    if as_list:
        if isinstance(x, list) and "complex" not in input_dtype[0]:
            x = [list(i) if len(i.shape) > 0 else [float(i)] for i in x]
        else:
            x = list(x)
    else:
        if len(x) == 1:
            x = x[0]

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        object=x,
        dtype=input_dtype[0],
        copy=copy,
        order="K",
        ndmin=ndmin,
    )


# asarray
@handle_frontend_test(
    fn_tree="jax.numpy.asarray",
    dtype_and_a=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=1,
        min_num_dims=0,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    test_with_out=st.just(False),
)
def test_jax_asarray(
    dtype_and_a,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
):
    dtype, a = dtype_and_a
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=a,
        dtype=dtype[0],
    )


# bool_
@handle_frontend_test(
    fn_tree="jax.numpy.bool_",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("bool")),
)
def test_jax_bool_(
    dtype_and_x,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="jax.numpy.cdouble",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("complex")
    ),
)
def test_jax_cdouble(
    dtype_and_x,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="jax.numpy.compress",
    dtype_arr_ax=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=10,
        max_dim_size=100,
        valid_axis=True,
        force_int_axis=True,
    ),
    condition=helpers.array_values(
        dtype=helpers.get_dtypes("bool"),
        shape=helpers.get_shape(
            min_num_dims=1, max_num_dims=1, min_dim_size=1, max_dim_size=5
        ),
    ),
)
def test_jax_compress(
    dtype_arr_ax,
    condition,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    dtype, arr, ax = dtype_arr_ax
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        condition=condition,
        a=arr[0],
        axis=ax,
    )


# copy
@handle_frontend_test(
    fn_tree="jax.numpy.copy",
    dtype_and_a=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=1,
        min_num_dims=0,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    test_with_out=st.just(False),
    test_with_copy=st.just(True),
)
def test_jax_copy(
    dtype_and_a,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    dtype, a = dtype_and_a
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=a[0],
    )


@handle_frontend_test(
    fn_tree="jax.numpy.csingle",
    aliases=["jax.numpy.complex64"],
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric")
    ),
)
def test_jax_csingle(
    dtype_and_x,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# double
@handle_frontend_test(
    fn_tree="jax.numpy.double",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_jax_double(
    dtype_and_x,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# empty
@handle_frontend_test(
    fn_tree="jax.numpy.empty",
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtype=helpers.get_dtypes("valid", full=False),
    test_with_out=st.just(False),
)
def test_jax_empty(
    shape,
    dtype,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        shape=shape,
        dtype=dtype[0],
    )


# empty_like
@handle_frontend_test(
    fn_tree="jax.numpy.empty_like",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    shape=helpers.get_shape(
        allow_none=True,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtype=helpers.get_dtypes("valid", full=False),
    test_with_out=st.just(False),
)
def test_jax_empty_like(
    dtype_and_x,
    shape,
    dtype,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        prototype=x[0],
        dtype=dtype[0],
        shape=shape,
    )


# eye
@handle_frontend_test(
    fn_tree="jax.numpy.eye",
    n=helpers.ints(min_value=3, max_value=10),
    m=st.none() | helpers.ints(min_value=3, max_value=10),
    k=helpers.ints(min_value=-2, max_value=2),
    dtypes=helpers.get_dtypes("valid", full=False),
    test_with_out=st.just(False),
)
def test_jax_eye(
    *,
    n,
    m,
    k,
    dtypes,
    on_device,
    fn_tree,
    test_flags,
    frontend,
    backend_fw,
):
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        N=n,
        M=m,
        k=k,
        dtype=dtypes[0],
    )


# from_dlpack
@handle_frontend_test(
    fn_tree="jax.numpy.from_dlpack",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric")
    ),
)
def test_jax_from_dlpack(
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
        x=x[0],
        backend_to_test=backend_fw,
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
    )


@handle_frontend_test(
    fn_tree="jax.numpy.frombuffer",
    dtype_buffer_count_offset=_get_dtype_buffer_count_offset(),
    test_with_out=st.just(False),
)
def test_jax_frombuffer(
    *,
    dtype_buffer_count_offset,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, buffer, count, offset = dtype_buffer_count_offset
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        buffer=buffer,
        dtype=input_dtype[0],
        count=count,
        offset=offset,
    )


# full
@handle_frontend_test(
    fn_tree="jax.numpy.full",
    shape=helpers.get_shape(
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    input_fill_dtype=_input_fill_and_dtype(),
    test_with_out=st.just(False),
)
def test_jax_full(
    shape,
    input_fill_dtype,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, _, fill_value, dtype = input_fill_dtype
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        shape=shape,
        fill_value=fill_value,
        dtype=dtype,
    )


# full_like
@handle_frontend_test(
    fn_tree="jax.numpy.full_like",
    input_fill_dtype=_input_fill_and_dtype(),
    test_with_out=st.just(False),
)
def test_jax_full_like(
    input_fill_dtype,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x, fill_value, dtype = input_fill_dtype
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        fill_value=fill_value,
        dtype=dtype,
    )


@handle_frontend_test(
    fn_tree="jax.numpy.geomspace",
    dtype_start_stop=_get_dtype_and_range(),
    num=helpers.ints(min_value=5, max_value=50),
    endpoint=st.booleans(),
    test_with_out=st.just(False),
)
def test_jax_geomspace(
    dtype_start_stop,
    num,
    endpoint,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtypes, start, stop = dtype_start_stop
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-1,
        start=start,
        stop=stop,
        num=num,
        endpoint=endpoint,
        dtype=input_dtypes[0],
    )


# hstack
@handle_frontend_test(
    fn_tree="jax.numpy.hstack",
    dtype_and_tup=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        shared_dtype=True,
        num_arrays=st.integers(min_value=2, max_value=2),
        shape=helpers.get_shape(
            min_num_dims=1, max_num_dims=3, min_dim_size=1, max_dim_size=5
        ),
    ),
    test_with_out=st.just(False),
)
def test_jax_hstack(
    dtype_and_tup,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
):
    input_dtype, x = dtype_and_tup
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        tup=x,
    )


# identity
@handle_frontend_test(
    fn_tree="jax.numpy.identity",
    n=helpers.ints(min_value=3, max_value=10),
    dtypes=helpers.get_dtypes("valid", full=False),
    test_with_out=st.just(False),
)
def test_jax_identity(
    *,
    n,
    dtypes,
    on_device,
    fn_tree,
    test_flags,
    frontend,
    backend_fw,
):
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        n=n,
        dtype=dtypes[0],
    )


@handle_frontend_test(
    fn_tree="jax.numpy.in1d",
    dtype_and_a=helpers.dtype_and_values(min_num_dims=1, max_num_dims=1),
    dtype_and_b=helpers.dtype_and_values(min_num_dims=1, max_num_dims=1),
    assume_unique=st.booleans(),
    invert=st.booleans(),
)
def test_jax_in1d(
    *,
    dtype_and_a,
    dtype_and_b,
    assume_unique,
    invert,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype_a, a = dtype_and_a
    input_dtype_b, b = dtype_and_b

    helpers.test_frontend_function(
        input_dtypes=input_dtype_a + input_dtype_b,
        frontend=frontend,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        ar1=a[0],
        ar2=b[0],
        assume_unique=assume_unique,
        invert=invert,
    )


@handle_frontend_test(
    fn_tree="jax.numpy.iterable",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_jax_iterable(
    dtype_and_x,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        y=x[0],
    )


# linspace
@handle_frontend_test(
    fn_tree="jax.numpy.linspace",
    dtype_start_stop=_get_dtype_and_range(),
    num=helpers.ints(min_value=2, max_value=5),
    axis=helpers.ints(min_value=-1, max_value=0),
    test_with_out=st.just(False),
)
def test_jax_linspace(
    dtype_start_stop,
    num,
    axis,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtypes, start, stop = dtype_start_stop
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        start=start,
        stop=stop,
        num=num,
        endpoint=True,
        retstep=False,
        dtype=input_dtypes[0],
        axis=axis,
        atol=1e-05,
        rtol=1e-05,
    )


# logspace
@handle_frontend_test(
    fn_tree="jax.numpy.logspace",
    dtype_start_stop=_get_dtype_and_range(),
    num=helpers.ints(min_value=5, max_value=50),
    base=helpers.ints(min_value=2, max_value=10),
    axis=helpers.ints(min_value=-1, max_value=0),
    test_with_out=st.just(False),
)
def test_jax_logspace(
    dtype_start_stop,
    num,
    base,
    axis,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtypes, start, stop = dtype_start_stop
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-01,
        start=start,
        stop=stop,
        num=num,
        endpoint=True,
        base=base,
        dtype=input_dtypes[0],
        axis=axis,
    )


# meshgrid
@handle_frontend_test(
    fn_tree="jax.numpy.meshgrid",
    dtype_and_arrays=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=st.integers(min_value=1, max_value=5),
        min_num_dims=1,
        max_num_dims=1,
        shared_dtype=True,
    ),
    sparse=st.booleans(),
    indexing=st.sampled_from(["xy", "ij"]),
    test_with_out=st.just(False),
    test_with_copy=st.just(True),
)
def test_jax_meshgrid(
    dtype_and_arrays,
    sparse,
    indexing,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
):
    dtype, arrays = dtype_and_arrays
    kw = {}
    i = 0
    for x_ in arrays:
        kw[f"x{i}"] = x_
        i += 1
    test_flags.num_positional_args = len(arrays)
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        **kw,
        sparse=sparse,
        indexing=indexing,
    )


# ndim
@handle_frontend_test(
    fn_tree="jax.numpy.ndim",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
)
def test_jax_ndim(
    dtype_and_x,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
    )


# ones
@handle_frontend_test(
    fn_tree="jax.numpy.ones",
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtype=helpers.get_dtypes("valid", full=False),
    test_with_out=st.just(False),
)
def test_jax_ones(
    shape,
    dtype,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        shape=shape,
        dtype=dtype[0],
    )


# ones_like
@handle_frontend_test(
    fn_tree="jax.numpy.ones_like",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    shape=helpers.get_shape(
        allow_none=True,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtype=helpers.get_dtypes("valid", full=False),
    test_with_out=st.just(False),
)
def test_jax_ones_like(
    dtype_and_x,
    shape,
    dtype,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
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
        dtype=dtype[0],
        shape=shape,
    )


@handle_frontend_test(
    fn_tree="jax.numpy.setdiff1d",
    dtype_and_a=helpers.dtype_and_values(
        min_num_dims=1,
        max_num_dims=1,
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    dtype_and_b=helpers.dtype_and_values(
        min_num_dims=1,
        max_num_dims=1,
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    use_size=st.booleans(),
    size=st.integers(min_value=1, max_value=100),
    assume_unique=st.booleans(),
)
def test_jax_setdiff1d(
    *,
    dtype_and_a,
    dtype_and_b,
    use_size,
    size,
    assume_unique,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype_a, a = dtype_and_a
    input_dtype_b, b = dtype_and_b
    if not use_size:
        size = None

    helpers.test_frontend_function(
        input_dtypes=input_dtype_a + input_dtype_b,
        frontend=frontend,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        ar1=a[0],
        ar2=b[0],
        assume_unique=assume_unique,
        size=size,
    )


# single
@handle_frontend_test(
    fn_tree="jax.numpy.single",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_jax_single(
    dtype_and_x,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="jax.numpy.size",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        abs_smallest_val=0,
        num_arrays=1,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=100,
        valid_axis=True,
        allow_neg_axes=True,
        force_int_axis=True,
    ),
)
def test_jax_size(
    dtype_x_axis,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
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
        a=x[0],
        axis=axis,
    )


# triu
@handle_frontend_test(
    fn_tree="jax.numpy.triu",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=1,
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    k=helpers.ints(min_value=-10, max_value=10),
    test_with_out=st.just(False),
)
def test_jax_triu(
    dtype_and_x,
    k,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        m=x[0],
        k=k,
    )


# vander
@handle_frontend_test(
    fn_tree="jax.numpy.vander",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        shape=st.tuples(
            st.integers(min_value=1, max_value=5),
        ),
    ),
    N=st.integers(min_value=0, max_value=5),
    increasing=st.booleans(),
)
def test_jax_vander(
    *,
    dtype_and_x,
    N,
    increasing,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        N=N,
        increasing=increasing,
    )


# zeros
@handle_frontend_test(
    fn_tree="jax.numpy.zeros",
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtypes=helpers.get_dtypes("numeric", full=False),
    test_with_out=st.just(False),
)
def test_jax_zeros(
    *,
    dtypes,
    shape,
    on_device,
    fn_tree,
    test_flags,
    frontend,
    backend_fw,
):
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        shape=shape,
        dtype=dtypes[0],
    )


# zeros_like
@handle_frontend_test(
    fn_tree="jax.numpy.zeros_like",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    shape=helpers.get_shape(
        allow_none=True,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtype=helpers.get_dtypes("valid", full=False),
    test_with_out=st.just(False),
)
def test_jax_zeros_like(
    dtype_and_x,
    dtype,
    shape,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
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
        dtype=dtype[0],
        shape=shape,
    )
