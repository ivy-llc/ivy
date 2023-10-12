# global
from hypothesis import assume, strategies as st
import numpy as np

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
import ivy


@st.composite
def _get_clip_inputs(draw):
    shape = draw(
        helpers.get_shape(
            min_num_dims=1, max_num_dims=5, min_dim_size=2, max_dim_size=10
        )
    )
    x_dtype, x, casting, dtype = draw(
        np_frontend_helpers.dtypes_values_casting_dtype(
            arr_func=[
                lambda: helpers.dtype_and_values(
                    available_dtypes=helpers.get_dtypes("valid"),
                    shape=shape,
                )
            ],
        ),
    )

    min = draw(
        helpers.array_values(dtype=x_dtype[0], shape=shape, min_value=-50, max_value=5)
    )
    max = draw(
        helpers.array_values(dtype=x_dtype[0], shape=shape, min_value=6, max_value=50)
    )
    return x_dtype, x, min, max, casting, dtype


# clip
@handle_frontend_test(
    fn_tree="numpy.clip",
    input_and_ranges=_get_clip_inputs(),
    where=np_frontend_helpers.where(),
    number_positional_args=np_frontend_helpers.get_num_positional_args_ufunc(
        fn_name="clip"
    ),
    test_with_out=st.just(False),
)
def test_numpy_clip(
    input_and_ranges,
    where,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtypes, x, min, max, casting, dtype = input_and_ranges
    where, input_dtypes, test_flags = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        a_min=min,
        a_max=max,
        out=None,
        where=where,
        casting=casting,
        order="K",
        dtype=dtype,
        subok=True,
    )


# cbrt
@handle_frontend_test(
    fn_tree="numpy.cbrt",
    dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(
        arr_func=[
            lambda: helpers.dtype_and_values(
                available_dtypes=helpers.get_dtypes("float"),
            )
        ],
    ),
    where=np_frontend_helpers.where(),
    number_positional_args=np_frontend_helpers.get_num_positional_args_ufunc(
        fn_name="cbrt"
    ),
)
def test_numpy_cbrt(
    dtypes_values_casting,
    where,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtypes, x, casting, dtype = dtypes_values_casting
    where, input_dtypes, test_flags = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-2,
        atol=1e-2,
        x=x[0],
        out=None,
        where=where,
        casting=casting,
        order="K",
        dtype=dtype,
        subok=True,
    )


# sqrt
@handle_frontend_test(
    fn_tree="numpy.sqrt",
    dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(
        arr_func=[
            lambda: helpers.dtype_and_values(
                available_dtypes=helpers.get_dtypes("float"),
            )
        ],
    ),
    where=np_frontend_helpers.where(),
    number_positional_args=np_frontend_helpers.get_num_positional_args_ufunc(
        fn_name="sqrt"
    ),
)
def test_numpy_sqrt(
    dtypes_values_casting,
    where,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtypes, x, casting, dtype = dtypes_values_casting
    where, input_dtypes, test_flags = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        atol=1e-2,
        out=None,
        where=where,
        casting=casting,
        order="K",
        dtype=dtype,
        subok=True,
    )


# reciprocal
@handle_frontend_test(
    fn_tree="numpy.reciprocal",
    dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(
        arr_func=[
            lambda: helpers.dtype_and_values(
                available_dtypes=helpers.get_dtypes("float"),
            )
        ],
    ),
    where=np_frontend_helpers.where(),
    number_positional_args=np_frontend_helpers.get_num_positional_args_ufunc(
        fn_name="reciprocal"
    ),
)
def test_numpy_reciprocal(
    dtypes_values_casting,
    where,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtypes, x, casting, dtype = dtypes_values_casting
    where, input_dtypes, test_flags = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        atol=1e-2,
        out=None,
        where=where,
        casting=casting,
        order="K",
        dtype=dtype,
        subok=True,
    )


# square
@handle_frontend_test(
    fn_tree="numpy.square",
    dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(
        arr_func=[
            lambda: helpers.dtype_and_values(
                available_dtypes=helpers.get_dtypes("numeric"),
            )
        ],
    ),
    where=np_frontend_helpers.where(),
    number_positional_args=np_frontend_helpers.get_num_positional_args_ufunc(
        fn_name="square"
    ),
)
def test_numpy_square(
    dtypes_values_casting,
    where,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtypes, x, casting, dtype = dtypes_values_casting
    where, input_dtypes, test_flags = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        out=None,
        where=where,
        casting=casting,
        order="K",
        dtype=dtype,
        subok=True,
    )


# absolute
@handle_frontend_test(
    fn_tree="numpy.absolute",
    dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(
        arr_func=[
            lambda: helpers.dtype_and_values(
                available_dtypes=helpers.get_dtypes("numeric"),
            )
        ],
    ),
    where=np_frontend_helpers.where(),
    number_positional_args=np_frontend_helpers.get_num_positional_args_ufunc(
        fn_name="absolute"
    ),
)
def test_numpy_absolute(
    dtypes_values_casting,
    where,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtypes, x, casting, dtype = dtypes_values_casting
    where, input_dtypes, test_flags = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        out=None,
        where=where,
        casting=casting,
        order="K",
        dtype=dtype,
        subok=True,
    )


# fabs
@handle_frontend_test(
    fn_tree="numpy.fabs",
    dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(
        arr_func=[
            lambda: helpers.dtype_and_values(
                available_dtypes=helpers.get_dtypes("float"),
            )
        ],
    ),
    where=np_frontend_helpers.where(),
    number_positional_args=np_frontend_helpers.get_num_positional_args_ufunc(
        fn_name="fabs"
    ),
)
def test_numpy_fabs(
    dtypes_values_casting,
    where,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtypes, x, casting, dtype = dtypes_values_casting
    where, input_dtypes, test_flags = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        out=None,
        where=where,
        casting=casting,
        order="K",
        dtype=dtype,
        subok=True,
    )


# sign
@handle_frontend_test(
    fn_tree="numpy.sign",
    dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(
        arr_func=[
            lambda: helpers.dtype_and_values(
                available_dtypes=helpers.get_dtypes("numeric"),
            )
        ],
    ),
    where=np_frontend_helpers.where(),
    number_positional_args=np_frontend_helpers.get_num_positional_args_ufunc(
        fn_name="sign"
    ),
)
def test_numpy_sign(
    dtypes_values_casting,
    where,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtypes, x, casting, dtype = dtypes_values_casting
    where, input_dtypes, test_flags = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )

    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        out=None,
        where=where,
        casting=casting,
        order="K",
        dtype=dtype,
        subok=True,
    )


# heaviside
@handle_frontend_test(
    fn_tree="numpy.heaviside",
    dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(
        arr_func=[
            lambda: helpers.dtype_and_values(
                available_dtypes=helpers.get_dtypes("float"),
                num_arrays=2,
                shared_dtype=True,
            )
        ],
    ),
    where=np_frontend_helpers.where(),
    number_positional_args=np_frontend_helpers.get_num_positional_args_ufunc(
        fn_name="heaviside"
    ),
)
def test_numpy_heaviside(
    dtypes_values_casting,
    where,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtypes, (x1_list, x2_list), casting, dtype = dtypes_values_casting
    where, input_dtypes, test_flags = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=x1_list,
        x2=x2_list,
        out=None,
        where=where,
        casting=casting,
        order="K",
        dtype=dtype,
        subok=True,
    )


# nan_to_num
@handle_frontend_test(
    fn_tree="numpy.nan_to_num",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-np.inf,
        max_value=+np.inf,
        allow_nan=True,
    ),
    posinf=st.one_of(st.none(), st.floats(min_value=0, max_value=10000)),
    neginf=st.one_of(st.none(), st.floats(min_value=-10000, max_value=0)),
    nan=st.floats(min_value=0, max_value=10),
    copy=st.booleans(),
    test_with_out=st.just(False),
)
def test_numpy_nan_to_num(
    dtype_and_x,
    copy,
    nan,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
    on_device,
    posinf,
    neginf,
):
    input_dtype, x = dtype_and_x
    # to avoid overflow errors of tf as you can't easily create a tensor
    # close to tf.dtype.max or tf.dtype.min, we need to assume that
    if ivy.current_backend_str() == "tensorflow":
        assume(posinf is not None and neginf is not None)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        copy=copy,
        nan=nan,
        posinf=posinf,
        neginf=neginf,
    )


# real_if_close
@handle_frontend_test(
    fn_tree="numpy.real_if_close",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    test_with_out=st.just(False),
)
def test_numpy_real_if_close(
    dtype_and_x,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
    )


# interp
@handle_frontend_test(
    fn_tree="numpy.interp",
    xp_and_fp=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=3,
        min_value=-10000,
        max_value=10000,
    ),
    x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    left=st.one_of(st.none(), st.floats()),
    right=st.one_of(st.none(), st.floats()),
    period=st.one_of(
        st.none(),
        st.floats(
            allow_nan=False,
            allow_infinity=False,
            allow_subnormal=False,
            min_value=0.1,
            max_value=1.0e5,
            exclude_min=True,
        ),
    ),
    test_with_out=st.just(False),
)
def test_numpy_interp(
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
    xp_and_fp,
    x,
    left,
    right,
    period,
):
    input_dtypes, xp_fp = xp_and_fp
    xp = ivy.array(xp_fp[0])
    fp = ivy.array(xp_fp[1])
    if period is None:
        xp_order = ivy.argsort(xp)
        xp = xp[xp_order]
        fp = fp[xp_order]
        previous = xp[0]
        for i in xp[1:]:
            assume(i > previous)
            previous = i
    x_dtype, x = x
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes + x_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        xp=xp,
        fp=fp,
        left=left,
        right=right,
        period=period,
    )


@handle_frontend_test(
    fn_tree="numpy.convolve",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        max_num_dims=1,
        num_arrays=2,
        min_value=-10,
        max_value=10,
        shared_dtype=True,
    ),
    mode=st.sampled_from(["valid", "same", "full"]),
    test_with_out=st.just(False),
)
def test_numpy_convolve(
    dtype_and_x,
    mode,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtypes, xs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=xs[0],
        v=xs[1],
        mode=mode,
    )


# copysign
@handle_frontend_test(
    fn_tree="numpy.copysign",
    dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(
        arr_func=[
            lambda: helpers.dtype_and_values(
                available_dtypes=helpers.get_dtypes("float"),
                num_arrays=2,
                shared_dtype=True,
                min_value=-100,
                max_value=100,
            )
        ],
    ),
    where=np_frontend_helpers.where(),
    test_with_out=st.just(False),
    number_positional_args=np_frontend_helpers.get_num_positional_args_ufunc(
        fn_name="copysign"
    ),
)
def test_numpy_copysign(
    dtypes_values_casting,
    where,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtypes, xs, casting, dtype = dtypes_values_casting
    where, input_dtypes, test_flags = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-2,
        atol=1e-2,
        x1=xs[0],
        x2=xs[1],
        out=None,
        where=where,
        casting=casting,
        order="K",
        dtype=dtype,
        subok=True,
    )


# lcm
@handle_frontend_test(
    fn_tree="numpy.lcm",
    dtype_and_inputs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=True,
        min_num_dims=1,
        max_num_dims=3,
        min_value=-100,
        max_value=100,
        allow_nan=False,
    ),
    where=np_frontend_helpers.where(),
    number_positional_args=np_frontend_helpers.get_num_positional_args_ufunc(
        fn_name="lcm"
    ),
)
def test_numpy_lcm(
    dtype_and_inputs,
    where,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtypes, xs = dtype_and_inputs
    where, input_dtypes, test_flags = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=xs[0],
        x2=xs[1],
        out=None,
        where=where,
    )


# gcd
@handle_frontend_test(
    fn_tree="numpy.gcd",
    dtype_and_inputs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=False,
        min_num_dims=1,
        max_num_dims=3,
        min_value=-100,
        max_value=100,
        allow_nan=False,
    ),
    where=np_frontend_helpers.where(),
    number_positional_args=np_frontend_helpers.get_num_positional_args_ufunc(
        fn_name="gcd"
    ),
)
def test_numpy_gcd(
    dtype_and_inputs,
    where,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtypes, xs = dtype_and_inputs
    where, input_dtypes, test_flags = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        test_flags=test_flags,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=xs[0],
        x2=xs[1],
        out=None,
        where=where,
    )
