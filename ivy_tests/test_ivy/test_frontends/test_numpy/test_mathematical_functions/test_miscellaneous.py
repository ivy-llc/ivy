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
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.clip"
    ),
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
    input_dtypes, (x_list, min_val_list, max_val_list) = x_min_n_max
    # ToDo should generate the data so we don't have to use assume here
    # same that is done in JAX clamp
    assume(np.all(np.less(np.asarray(min_val_list), np.asarray(max_val_list))))
    where, as_variable, native_array = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        as_variable=as_variable,
        native_array=native_array,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="clip",
        x=x_list,
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
    dtype=helpers.get_dtypes("float", full=False, none=True),
    where=np_frontend_helpers.where(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.cbrt"
    ),
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
    where, as_variable, native_array = np_frontend_helpers.handle_where_and_array_bools(
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
        x=x[0],
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
    dtype=helpers.get_dtypes("float", full=False, none=True),
    where=np_frontend_helpers.where(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.sqrt"
    ),
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
    where, as_variable, native_array = np_frontend_helpers.handle_where_and_array_bools(
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
        x=x[0],
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
    dtype=helpers.get_dtypes("float", full=False, none=True),
    where=np_frontend_helpers.where(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.square"
    ),
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
    where, as_variable, native_array = np_frontend_helpers.handle_where_and_array_bools(
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
        x=x[0],
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
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.absolute"
    ),
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
        x=x[0],
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
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.fabs"
    ),
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
        x=x[0],
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
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.heaviside"
    ),
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
    input_dtypes, (x1_list, x2_list) = x1_x2
    where, as_variable, native_array = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        as_variable=as_variable,
        native_array=native_array,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="heaviside",
        x1=x1_list,
        x2=x2_list,
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
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.nan_to_num"
    ),
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
        x=x[0],
        nan=0.0,
        posinf=posinf,
        neginf=neginf,
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.real_if_close"
    ),
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
        a=x[0],
    )


@handle_cmd_line_args
@given(
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.interp"
    ),
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
    input_dtypes, (xp, fp) = xp_and_fp
    xp_order = argsort(xp)
    xp = array(xp)[xp_order]
    fp = array(fp)[xp_order]
    previous = xp[0]
    for i in xp[1:]:
        assume(i > previous)
        previous = i
    x_dtype, x = x
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes + x_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="interp",
        x=x[0],
        xp=xp,
        fp=fp,
        left=left,
        right=right,
        period=period,
    )
