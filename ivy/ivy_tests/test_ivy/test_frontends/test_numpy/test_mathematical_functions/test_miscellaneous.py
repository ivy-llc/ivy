# global
import numpy as np
from hypothesis import assume, given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args
from ivy import argsort, array


@handle_cmd_line_args
@given(
    x_min_n_max=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"), num_arrays=3, shared_dtype=True
    ),
    dtype=helpers.get_dtypes("float", full=False, none=True),
    where=np_frontend_helpers.where(),
    as_variable=helpers.array_bools(num_arrays=1),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.clip"
    ),
    native_array=helpers.array_bools(num_arrays=1),
)
def test_numpy_clip(
    x_min_n_max,
    dtype,
    where,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    (x_dtype, min_dtype, max_dtype), (x_list, min_val_list, max_val_list) = x_min_n_max
    assume(np.all(np.less(np.asarray(min_val_list), np.asarray(max_val_list))))
    input_dtype = [x_dtype]
    where = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtype,
        as_variable=as_variable,
        native_array=native_array,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="clip",
        x=np.asarray(x_list, dtype=input_dtype[0]),
        a_min=min_val_list,
        a_max=max_val_list,
        out=None,
        where=where,
        casting="same_kind",
        order="k",
        dtype=dtype,
        subok=True,
        test_values=False,
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric")
    ),
    dtype=helpers.get_dtypes("numeric", full=False, none=True),
    where=np_frontend_helpers.where(),
    as_variable=helpers.array_bools(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.cbrt"
    ),
    native_array=helpers.array_bools(),
)
def test_numpy_cbrt(
    dtype_and_x,
    dtype,
    where,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x
    input_dtype = [input_dtype]
    where = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtype,
        as_variable=as_variable,
        native_array=native_array,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="cbrt",
        x=np.asarray(x, dtype=input_dtype[0]),
        out=None,
        where=where,
        casting="same_kind",
        order="k",
        dtype=dtype,
        subok=True,
        test_values=False,
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric")
    ),
    dtype=helpers.get_dtypes("numeric", full=False, none=True),
    where=np_frontend_helpers.where(),
    as_variable=helpers.array_bools(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.sqrt"
    ),
    native_array=helpers.array_bools(),
)
def test_numpy_sqrt(
    dtype_and_x,
    dtype,
    where,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x
    input_dtype = [input_dtype]
    where = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtype,
        as_variable=as_variable,
        native_array=native_array,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="sqrt",
        x=np.asarray(x, dtype=input_dtype[0]),
        out=None,
        where=where,
        casting="same_kind",
        order="k",
        dtype=dtype,
        subok=True,
        test_values=False,
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric")
    ),
    dtype=helpers.get_dtypes("numeric", full=False, none=True),
    where=np_frontend_helpers.where(),
    as_variable=helpers.array_bools(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.square"
    ),
    native_array=helpers.array_bools(),
)
def test_numpy_square(
    dtype_and_x,
    dtype,
    where,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x
    input_dtype = [input_dtype]
    where = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtype,
        as_variable=as_variable,
        native_array=native_array,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="square",
        x=np.asarray(x, dtype=input_dtype[0]),
        out=None,
        where=where,
        casting="same_kind",
        order="k",
        dtype=dtype,
        subok=True,
        test_values=False,
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    dtype=helpers.get_dtypes("float", full=False, none=True),
    where=np_frontend_helpers.where(),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.absolute"
    ),
    native_array=st.booleans(),
)
def test_numpy_absolute(
    dtype_and_x,
    dtype,
    where,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="absolute",
        x=np.asarray(x, dtype=input_dtype),
        out=None,
        where=where,
        casting="same_kind",
        order="k",
        dtype=dtype,
        subok=False,
        test_values=False,
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    dtype=helpers.get_dtypes("float", full=False, none=True),
    where=np_frontend_helpers.where(),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.fabs"
    ),
    native_array=st.booleans(),
)
def test_numpy_fabs(
    dtype_and_x,
    dtype,
    where,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="fabs",
        x=np.asarray(x, dtype=input_dtype),
        out=None,
        where=where,
        casting="same_kind",
        order="k",
        dtype=dtype,
        subok=False,
        test_values=False,
    )


@handle_cmd_line_args
@given(
    x1_x2=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
    dtype=helpers.get_dtypes("float", full=False, none=True),
    where=np_frontend_helpers.where(),
    as_variable=helpers.array_bools(num_arrays=1),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.heaviside"
    ),
    native_array=helpers.array_bools(num_arrays=1),
)
def test_numpy_heaviside(
    x1_x2,
    dtype,
    where,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    (x1_dtype, x2_dtype), (x1_list, x2_list) = x1_x2
    input_dtype = [x1_dtype, x2_dtype]
    where = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtype,
        as_variable=as_variable,
        native_array=native_array,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="heaviside",
        x1=np.asarray(x1_list, dtype=input_dtype[0]),
        x2=np.asarray(x2_list, dtype=input_dtype[0]),
        out=None,
        where=where,
        casting="same_kind",
        order="k",
        dtype=dtype,
        subok=True,
        test_values=False,
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    as_variable=helpers.array_bools(num_arrays=1),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.nan_to_num"
    ),
    native_array=helpers.array_bools(num_arrays=1),
    posinf=st.one_of(st.none(), st.floats()),
    neginf=st.one_of(st.none(), st.floats()),
)
def test_numpy_nan_to_num(
    dtype_and_x, as_variable, num_positional_args, native_array, posinf, neginf, fw
):
    input_dtype, x = dtype_and_x
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="nan_to_num",
        x=np.asarray(x, dtype=input_dtype),
        nan=0.0,
        posinf=posinf,
        neginf=neginf,
    )


@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.real_if_close"
    ),
    native_array=st.booleans(),
)
def test_numpy_real_if_close(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="real_if_close",
        a=np.asarray(x, dtype=input_dtype),
    )


@handle_cmd_line_args
@given(
    as_variable=helpers.array_bools(num_arrays=3),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.interp"
    ),
    native_array=helpers.array_bools(num_arrays=3),
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
)
def test_numpy_interp(
    as_variable,
    num_positional_args,
    native_array,
    xp_and_fp,
    x,
    left,
    right,
    period,
    fw,
):
    (xp_dtype, fp_dtype), (xp, fp) = xp_and_fp
    xp_order = argsort(xp)
    xp = array(xp)[xp_order]
    fp = array(fp)[xp_order]
    previous = xp[0]
    for i in xp[1:]:
        assume(i > previous)
        previous = i
    x_dtype, x = x
    np_frontend_helpers.test_frontend_function(
        input_dtypes=[x_dtype, xp_dtype, fp_dtype],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="interp",
        x=np.asarray(x, dtype=x_dtype),
        xp=np.asarray(xp, dtype=xp_dtype),
        fp=np.asarray(fp, dtype=fp_dtype),
        left=left,
        right=right,
        period=period,
    )
