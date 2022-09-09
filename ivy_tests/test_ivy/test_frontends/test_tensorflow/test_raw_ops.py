# global
import ivy
import numpy as np
from hypothesis import given, strategies as st
from functools import reduce  # for making strategy
from operator import mul  # for making strategy

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# Functions to use in strategy #
# ---------------------------- #

# from array-api repo
def _broadcast_shapes(shape1, shape2):
    """Broadcasts `shape1` and `shape2`"""
    N1 = len(shape1)
    N2 = len(shape2)
    N = max(N1, N2)
    shape = [None for _ in range(N)]
    i = N - 1
    while i >= 0:
        n1 = N1 - N + i
        if N1 - N + i >= 0:
            d1 = shape1[n1]
        else:
            d1 = 1
        n2 = N2 - N + i
        if N2 - N + i >= 0:
            d2 = shape2[n2]
        else:
            d2 = 1

        if d1 == 1:
            shape[i] = d2
        elif d2 == 1:
            shape[i] = d1
        elif d1 == d2:
            shape[i] = d1
        else:
            raise Exception("Broadcast error")

        i = i - 1

    return tuple(shape)


# from array-api repo
def broadcast_shapes(*shapes):
    if len(shapes) == 0:
        raise ValueError("shapes=[] must be non-empty")
    elif len(shapes) == 1:
        return shapes[0]
    result = _broadcast_shapes(shapes[0], shapes[1])
    for i in range(2, len(shapes)):
        result = _broadcast_shapes(result, shapes[i])
    return result


# np.prod and others have overflow and math.prod is Python 3.8+ only
def prod(seq):
    return reduce(mul, seq, 1)


# from array-api repo
def mutually_broadcastable_shapes(
    num_shapes: int,
    *,
    base_shape=(),
    min_dims: int = 1,
    max_dims: int = 4,
    min_side: int = 1,
    max_side: int = 4,
):
    if max_dims is None:
        max_dims = min(max(len(base_shape), min_dims) + 5, 32)
    if max_side is None:
        max_side = max(base_shape[-max_dims:] + (min_side,)) + 5
    return (
        helpers.nph.mutually_broadcastable_shapes(
            num_shapes=num_shapes,
            base_shape=base_shape,
            min_dims=min_dims,
            max_dims=max_dims,
            min_side=min_side,
            max_side=max_side,
        )
        .map(lambda BS: BS.input_shapes)
        .filter(lambda shapes: all(prod(i for i in s if i > 0) < 1000 for s in shapes))
    )


# for data generation in multiple tests
dtype_shared = st.shared(st.sampled_from(ivy.valid_dtypes), key="dtype")


# Acos
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.Acos"
    ),
)
def test_tensorflow_Acos(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="raw_ops.Acos",
        x=np.asarray(x, dtype=input_dtype),
    )


# Acosh
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.Acosh"
    ),
)
def test_tensorflow_Acosh(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="raw_ops.Acosh",
        x=np.asarray(x, dtype=input_dtype),
    )


def _get_broadcastable_shapes(draw):
    to_shape = draw(helpers.get_shape(allow_none=False, min_num_dims=2))
    in_shape = draw(helpers.nph.broadcastable_shapes(shape=to_shape, min_dims=3))
    return in_shape, to_shape


@st.composite
def array_and_broadcastable_shape(draw, dtype):
    in_shape = draw(helpers.nph.array_shapes(min_dims=1, max_dims=4))
    x = draw(helpers.nph.arrays(shape=in_shape, dtype=dtype))
    to_shape = draw(
        mutually_broadcastable_shapes(1, base_shape=in_shape)
        .map(lambda S: S[0])
        .filter(lambda s: broadcast_shapes(in_shape, s) == s),
        label="shape",
    )
    return (x, to_shape)


# BroadcastTo
@handle_cmd_line_args
@given(
    array_and_shape=array_and_broadcastable_shape(dtype_shared),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.BroadcastTo"
    ),
    native_array=st.booleans(),
)
def test_tensorflow_BroadcastTo(
    array_and_shape, as_variable, num_positional_args, native_array, fw
):
    x, to_shape = array_and_shape
    helpers.test_frontend_function(
        input_dtypes=x.dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="raw_ops.BroadcastTo",
        input=x,
        shape=to_shape,
    )


# noinspection DuplicatedCode
@st.composite
def _arrays_idx_n_dtypes(draw):
    num_dims = draw(st.shared(helpers.ints(min_value=1, max_value=4), key="num_dims"))
    num_arrays = draw(
        st.shared(helpers.ints(min_value=2, max_value=4), key="num_arrays")
    )
    common_shape = draw(
        helpers.lists(
            arg=helpers.ints(min_value=2, max_value=3),
            min_size=num_dims - 1,
            max_size=num_dims - 1,
        )
    )
    unique_idx = draw(helpers.ints(min_value=0, max_value=num_dims - 1))
    unique_dims = draw(
        helpers.lists(
            arg=helpers.ints(min_value=2, max_value=3),
            min_size=num_arrays,
            max_size=num_arrays,
        )
    )
    xs = list()
    input_dtypes = draw(
        helpers.array_dtypes(
            available_dtypes=draw(helpers.get_dtypes("float")), shared_dtype=True
        )
    )
    for ud, dt in zip(unique_dims, input_dtypes):
        x = draw(
            helpers.array_values(
                shape=common_shape[:unique_idx] + [ud] + common_shape[unique_idx:],
                dtype=dt,
            )
        )
        xs.append(x)
    return xs, input_dtypes, unique_idx


# Concat
@handle_cmd_line_args
@given(
    xs_n_input_dtypes_n_unique_idx=_arrays_idx_n_dtypes(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.Concat"
    ),
)
def test_tensorflow_Concat(
    xs_n_input_dtypes_n_unique_idx,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    xs, input_dtypes, unique_idx = xs_n_input_dtypes_n_unique_idx
    xs = [np.asarray(x, dtype=dt) for x, dt in zip(xs, input_dtypes)]
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="raw_ops.Concat",
        concat_dim=unique_idx,
        values=xs,
    )


# Cos
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.Cos"
    ),
)
def test_tensorflow_Cos(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="raw_ops.Cos",
        x=np.asarray(x, dtype=input_dtype),
    )


# Cosh
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.Cosh"
    ),
)
def test_tensorflow_Cosh(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="raw_ops.Cosh",
        x=np.asarray(x, dtype=input_dtype),
    )


@st.composite
def _dtypes(draw):
    return draw(
        st.shared(
            helpers.list_of_length(
                x=st.sampled_from(draw(helpers.get_dtypes("numeric"))), length=1
            ),
            key="dtype",
        )
    )


@st.composite
def _fill_value(draw):
    dtype = draw(_dtypes())[0]
    if ivy.is_uint_dtype(dtype):
        return draw(helpers.ints(min_value=0, max_value=5))
    elif ivy.is_int_dtype(dtype):
        return draw(helpers.ints(min_value=-5, max_value=5))
    return draw(helpers.floats(min_value=-5, max_value=5))


# fill
@handle_cmd_line_args
@given(
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    fill_value=_fill_value(),
    dtypes=_dtypes(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.Fill"
    ),
)
def test_tensorflow_Fill(
    shape,
    fill_value,
    dtypes,
    with_out,
    as_variable,
    native_array,
    num_positional_args,
    fw,
):
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="raw_ops.Fill",
        dims=shape,
        value=fill_value,
        rtol=1e-05,
    )


# Asin
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.Asin"
    ),
)
def test_tensorflow_Asin(
    dtype_and_x, as_variable, num_positional_args, fw, native_array
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="raw_ops.Asin",
        x=np.asarray(x, dtype=dtype),
    )


# argmax
@handle_cmd_line_args
@given(
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
        valid_axis=True,
        force_int_axis=True,
        min_num_dims=1,
        min_value=-5,
        max_value=5,
        allow_inf=False,
    ),
    output_type=st.sampled_from(["int16", "int32", "int64"]),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.ArgMax"
    ),
)
def test_tensorflow_ArgMax(
    dtype_x_axis,
    as_variable,
    with_out,
    num_positional_args,
    fw,
    native_array,
    output_type,
):
    dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="raw_ops.ArgMax",
        input=np.asarray(x, dtype=dtype),
        dimension=axis,
        output_type=output_type,
    )


# Atan
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.Atan"
    ),
)
def test_tensorflow_Atan(
    dtype_and_x, as_variable, num_positional_args, fw, native_array
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="raw_ops.Atan",
        x=np.asarray(x, dtype=dtype),
    )


# BitwiseAnd
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.BitwiseAnd"
    ),
)
def test_tensorflow_BitwiseAnd(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="raw_ops.BitwiseAnd",
        x=np.asarray(x[0], dtype=input_dtype[0]),
        y=np.asarray(x[1], dtype=input_dtype[1]),
    )


# BitwiseOr
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.BitwiseOr"
    ),
)
def test_tensorflow_BitwiseOr(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="raw_ops.BitwiseOr",
        x=np.asarray(x[0], dtype=input_dtype[0]),
        y=np.asarray(x[1], dtype=input_dtype[1]),
    )


# BitwiseXor
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.BitwiseXor"
    ),
)
def test_tensorflow_BitwiseXor(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="raw_ops.BitwiseXor",
        x=np.asarray(x[0], dtype=input_dtype[0]),
        y=np.asarray(x[1], dtype=input_dtype[1]),
    )


# Atanh
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.Atanh"
    ),
)
def test_tensorflow_Atanh(
    dtype_and_x, as_variable, num_positional_args, fw, native_array
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="raw_ops.Atanh",
        x=np.asarray(x, dtype=dtype),
    )


# Tan
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.Tan"
    ),
)
def test_tensorflow_Tan(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="raw_ops.Tan",
        x=np.asarray(x, dtype=input_dtype),
    )


# Square
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.Square"
    ),
)
def test_tensorflow_Square(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="raw_ops.Square",
        x=np.asarray(x, dtype=input_dtype),
    )


# Sqrt
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.Sqrt"
    ),
)
def test_tensorflow_Sqrt(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="raw_ops.Sqrt",
        x=np.asarray(x, dtype=input_dtype),
    )


# Tanh
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.Tanh"
    ),
)
def test_tensorflow_Tanh(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="raw_ops.Tanh",
        x=np.asarray(x, dtype=input_dtype),
    )


@st.composite
def _permute_dims_helper(draw):
    shape = draw(st.shared(helpers.get_shape(min_num_dims=1), key="shape"))
    dims = [x for x in range(len(shape))]
    permutation = draw(st.permutations(dims))
    return permutation


# Transpose
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
    perm=_permute_dims_helper(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.Transpose"
    ),
)
def test_tensorflow_transpose(
    dtype_and_x, perm, as_variable, num_positional_args, fw, native_array
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="raw_ops.Transpose",
        x=np.asarray(x, dtype=dtype),
        perm=perm,
    )


# Maximum
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.Maximum"
    ),
)
def test_tensorflow_Maximum(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="raw_ops.Maximum",
        x=np.asarray(x[0], dtype=input_dtype[0]),
        y=np.asarray(x[1], dtype=input_dtype[1]),
    )


# Minimum
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.Minimum"
    ),
)
def test_tensorflow_Minimum(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="raw_ops.Minimum",
        x=np.asarray(x[0], dtype=input_dtype[0]),
        y=np.asarray(x[1], dtype=input_dtype[1]),
    )


# Sub
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"), num_arrays=2, shared_dtype=True
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.Sub"
    ),
)
def test_tensorflow_Sub(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="raw_ops.Sub",
        x=np.asarray(x[0], dtype=dtype[0]),
        y=np.asarray(x[1], dtype=dtype[1]),
    )


# Less
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.Less"
    ),
)
def test_tensorflow_Less(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="raw_ops.Less",
        x=np.asarray(x[0], dtype=input_dtype[0]),
        y=np.asarray(x[1], dtype=input_dtype[1]),
    )


# LessEqual
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.LessEqual"
    ),
)
def test_tensorflow_LessEqual(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="raw_ops.LessEqual",
        x=np.asarray(x[0], dtype=input_dtype[0]),
        y=np.asarray(x[1], dtype=input_dtype[1]),
    )


# Floor
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.Floor"
    ),
)
def test_tensorflow_Floor(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="raw_ops.Floor",
        x=np.asarray(x, dtype=input_dtype),
    )


# FloorDiv
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.FloorDiv"
    ),
)
def test_tensorflow_FloorDiv(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="raw_ops.FloorDiv",
        x=np.asarray(x[0], dtype=input_dtype[0]),
        y=np.asarray(x[1], dtype=input_dtype[1]),
    )


# Exp
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.Tanh"
    ),
)
def test_tensorflow_Exp(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="raw_ops.Exp",
        x=np.asarray(x, dtype=input_dtype),
    )


# Expm1
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.Expm1"
    ),
)
def test_tensorflow_Expm1(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="raw_ops.Expm1",
        x=np.asarray(x, dtype=input_dtype),
    )


# Log
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.Log"
    ),
)
def test_tensorflow_Log(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="raw_ops.Log",
        x=np.asarray(x, dtype=input_dtype),
    )


# Sinh
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.Sinh"
    ),
)
def test_tensorflow_Sinh(
    dtype_and_x, as_variable, num_positional_args, native_array, fw
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="raw_ops.Sinh",
        x=np.asarray(x, dtype=input_dtype),
    )


# Reshape
@st.composite
def _reshape_helper(draw):
    # generate a shape s.t len(shape) > 0
    shape = draw(helpers.get_shape(min_num_dims=1))
    reshape_shape = draw(helpers.reshape_shapes(shape=shape))
    dtype = draw(helpers.array_dtypes(num_arrays=1))[0]
    x = draw(helpers.array_values(dtype=dtype, shape=shape))
    return x, dtype, reshape_shape


@handle_cmd_line_args
@given(
    x_reshape=_reshape_helper(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.Reshape",
    ),
)
def test_tensorflow_Reshape(
    x_reshape,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    x, dtype, shape = x_reshape
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="raw_ops.Reshape",
        tensor=np.asarray(x, dtype=dtype),
        shape=shape,
    )


# ZerosLike
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.ZerosLike"
    ),
)
def test_tensorflow_zeros_like(
    dtype_and_x, as_variable, num_positional_args, fw, native_array
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        fn_tree="raw_ops.ZerosLike",
        x=np.asarray(x, dtype=dtype),
    )
