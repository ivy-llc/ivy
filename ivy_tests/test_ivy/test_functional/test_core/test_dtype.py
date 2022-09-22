"""Collection of tests for unified dtype functions."""

# global
import numpy as np
import pytest
from hypothesis import given, assume, strategies as st

# local
import ivy
import ivy.functional.backends.jax as ivy_jax
import ivy.functional.backends.numpy as ivy_np
import ivy.functional.backends.tensorflow as ivy_tf
import ivy.functional.backends.torch as ivy_torch
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# dtype objects
@handle_cmd_line_args
def test_dtype_instances(device):
    assert ivy.exists(ivy.int8)
    assert ivy.exists(ivy.int16)
    assert ivy.exists(ivy.int32)
    assert ivy.exists(ivy.int64)
    assert ivy.exists(ivy.uint8)
    if ivy.current_backend_str() != "torch":
        assert ivy.exists(ivy.uint16)
        assert ivy.exists(ivy.uint32)
        assert ivy.exists(ivy.uint64)
    assert ivy.exists(ivy.float32)
    assert ivy.exists(ivy.float64)
    assert ivy.exists(ivy.bool)


# for data generation in multiple tests
dtype_shared = st.shared(st.sampled_from(ivy_np.valid_dtypes), key="dtype")


@st.composite
def dtypes_shared(draw, num_dtypes):
    if isinstance(num_dtypes, str):
        num_dtypes = draw(st.shared(helpers.ints(), key=num_dtypes))
    return draw(
        st.shared(
            st.lists(
                st.sampled_from(ivy_np.valid_dtypes),
                min_size=num_dtypes,
                max_size=num_dtypes,
            ),
            key="dtypes",
        )
    )


# Array API Standard Function Tests #
# --------------------------------- #


@st.composite
def _astype_helper(draw):
    dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            num_arrays=1,
            small_abs_safety_factor=4,
            large_abs_safety_factor=4,
            safety_factor_scale="log",
        )
    )

    cast_dtype = draw(
        helpers.get_castable_dtype(draw(helpers.get_dtypes("valid")), dtype, x)
    )
    return dtype, x, cast_dtype


# astype
@handle_cmd_line_args
@given(
    dtype_and_x_and_cast_dtype=_astype_helper(),
    num_positional_args=helpers.num_positional_args(fn_name="astype"),
)
def test_astype(
    dtype_and_x_and_cast_dtype,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x, cast_dtype = dtype_and_x_and_cast_dtype
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="astype",
        rtol_=1e-3,
        atol_=1e-3,
        x=np.asarray(x, dtype=input_dtype),
        dtype=cast_dtype,
    )


# broadcast arrays
@st.composite
def broadcastable_arrays(draw, dtypes):
    num_arrays = st.shared(helpers.ints(min_value=2, max_value=5), key="num_arrays")
    shapes = draw(num_arrays.flatmap(helpers.mutually_broadcastable_shapes))
    dtypes = draw(dtypes)
    arrays = []
    for c, (shape, dtype) in enumerate(zip(shapes, dtypes), 1):
        x = draw(helpers.nph.arrays(dtype=dtype, shape=shape), label=f"x{c}").tolist()
        arrays.append(x)
    return arrays


@handle_cmd_line_args
@given(
    arrays=broadcastable_arrays(dtypes_shared("num_arrays")),
    input_dtypes=dtypes_shared("num_arrays"),
)
def test_broadcast_arrays(
    arrays,
    input_dtypes,
    as_variable,
    native_array,
    container,
    instance_method,
    fw,
):
    kw = {}
    for i, (array, dtype) in enumerate(zip(arrays, input_dtypes)):
        kw["x{}".format(i)] = np.asarray(array, dtype=dtype)
    num_positional_args = len(kw)
    helpers.test_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="broadcast_arrays",
        **kw,
    )


@handle_cmd_line_args
@given(
    array_and_shape=helpers.array_and_broadcastable_shape(dtype_shared),
    input_dtype=dtype_shared,
    num_positional_args=helpers.num_positional_args(fn_name="broadcast_to"),
)
def test_broadcast_to(
    array_and_shape,
    input_dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    array, to_shape = array_and_shape
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="broadcast_to",
        x=array,
        shape=to_shape,
    )


# can_cast
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", full=True), num_arrays=1
    ),
    to_dtype=helpers.get_dtypes("valid", full=False),
    num_positional_args=helpers.num_positional_args(fn_name="can_cast"),
)
def test_can_cast(
    dtype_and_x,
    to_dtype,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="can_cast",
        from_=np.array(x, dtype=input_dtype),
        to=to_dtype,
    )


@st.composite
def _array_or_type(draw, float_or_int):
    valid_dtypes = {
        "float": draw(helpers.get_dtypes("float")),
        "int": draw(helpers.get_dtypes("integer")),
    }[float_or_int]
    return draw(
        st.sampled_from(
            (
                draw(
                    helpers.dtype_and_values(
                        available_dtypes=valid_dtypes, num_arrays=1
                    )
                ),
                draw(st.sampled_from(valid_dtypes)),
            )
        )
    )


# finfo
@handle_cmd_line_args
@given(
    type=_array_or_type("float"),
    num_positional_args=helpers.num_positional_args(fn_name="finfo"),
)
def test_finfo(
    type,
    num_positional_args,
    fw,
):
    if isinstance(type, str):
        input_dtype = type
    else:
        input_dtype, x = type
        type = np.array(x, dtype=input_dtype)
    ret = helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=False,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=False,
        container_flags=False,
        instance_method=False,
        fw=fw,
        fn_name="finfo",
        type=type,
        test_values=False,
    )
    if not ivy.exists(ret):
        return
    mach_lims, mach_lims_np = ret
    assert np.allclose(mach_lims.min, mach_lims_np.min, rtol=1e-2, atol=1e-2)
    assert np.allclose(mach_lims.max, mach_lims_np.max, rtol=1e-2, atol=1e-2)
    assert np.allclose(mach_lims.eps, mach_lims_np.eps, rtol=1e-2, atol=1e-2)
    assert mach_lims.bits == mach_lims_np.bits


# iinfo
@handle_cmd_line_args
@given(
    type=_array_or_type("int"),
    num_positional_args=helpers.num_positional_args(fn_name="iinfo"),
)
def test_iinfo(
    type,
    num_positional_args,
    fw,
):
    if isinstance(type, str):
        input_dtype = type
    else:
        input_dtype, x = type
        type = np.array(x, dtype=input_dtype)
    ret = helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=False,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=False,
        container_flags=False,
        instance_method=False,
        fw=fw,
        fn_name="iinfo",
        type=type,
        test_values=False,
    )

    if not ivy.exists(ret):
        return
    mach_lims, mach_lims_np = ret
    assert mach_lims.min == mach_lims_np.min
    assert mach_lims.max == mach_lims_np.max
    assert mach_lims.dtype == mach_lims_np.dtype
    assert mach_lims.bits == mach_lims_np.bits


# result_type
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", full=True),
        num_arrays=st.shared(helpers.ints(min_value=2, max_value=5), key="num_arrays"),
        shared_dtype=False,
    ),
)
def test_result_type(
    dtype_and_x,
    as_variable,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, x = helpers.as_lists(*dtype_and_x)
    kw = {}
    for i, (dtype_, x_) in enumerate(zip(dtype, x)):
        kw["x{}".format(i)] = np.asarray(x_, dtype=dtype_)
    num_positional_args = len(kw)
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="result_type",
        **kw,
    )


# Extra Ivy Function Tests #
# ------------------------ #


# as_ivy_dtype
@handle_cmd_line_args
@given(
    input_dtype=st.sampled_from(ivy.valid_dtypes),
)
def test_as_ivy_dtype(
    input_dtype,
):
    res = ivy.as_ivy_dtype(input_dtype)
    if isinstance(input_dtype, str):
        assert isinstance(res, str)
        return

    assert isinstance(input_dtype, ivy.Dtype) or isinstance(
        input_dtype, str
    ), f"input_dtype={input_dtype!r}, but should be str or ivy.Dtype"
    assert isinstance(res, str), f"result={res!r}, but should be str"


_valid_dtype_in_all_frameworks = [
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "float16",
    "float32",
    "float64",
    "bool",
]


# as_native_dtype
@handle_cmd_line_args
@given(input_dtype=st.sampled_from(_valid_dtype_in_all_frameworks))
def test_as_native_dtype(
    input_dtype,
):
    res = ivy.as_native_dtype(input_dtype)
    if isinstance(input_dtype, ivy.NativeDtype):
        assert isinstance(res, ivy.NativeDtype)
        return

    assert isinstance(input_dtype, ivy.Dtype) or isinstance(
        input_dtype, str
    ), f"input_dtype={input_dtype!r}, but should be str or ivy.Dtype"
    assert isinstance(
        res, ivy.NativeDtype
    ), f"result={res!r}, but should be ivy.NativeDtype"


# closest_valid_dtypes
@handle_cmd_line_args
@given(input_dtype=st.sampled_from(_valid_dtype_in_all_frameworks))
def test_closest_valid_dtype(
    input_dtype,
):
    res = ivy.closest_valid_dtype(input_dtype)
    assert isinstance(input_dtype, ivy.Dtype) or isinstance(input_dtype, str)
    assert isinstance(res, ivy.Dtype) or isinstance(
        res, str
    ), f"result={res!r}, but should be str or ivy.Dtype"


# default_dtype
@handle_cmd_line_args
@given(
    input_dtype=helpers.get_dtypes("valid", full=False),
    as_native=st.booleans(),
)
def test_default_dtype(
    input_dtype,
    as_native,
):
    assume(input_dtype in ivy.valid_dtypes)

    res = ivy.default_dtype(dtype=input_dtype, as_native=as_native)
    assert (
        isinstance(input_dtype, ivy.Dtype)
        or isinstance(input_dtype, str)
        or isinstance(input_dtype, ivy.NativeDtype)
    )
    assert isinstance(res, ivy.Dtype) or isinstance(
        input_dtype, str
    ), f"input_dtype={input_dtype!r}, but should be str or ivy.Dtype"


# dtype
@handle_cmd_line_args
@given(
    array=helpers.nph.arrays(
        dtype=dtype_shared,
        shape=helpers.lists(
            arg=helpers.ints(min_value=1, max_value=5),
            min_size="num_dims",
            max_size="num_dims",
            size_bounds=[1, 5],
        ),
    ),
    input_dtype=dtype_shared,
    as_native=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="dtype"),
)
def test_dtype(
    array,
    input_dtype,
    as_native,
    as_variable,
    num_positional_args,
    native_array,
    container,
    fw,
):
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=False,
        fw=fw,
        fn_name="dtype",
        x=array,
        as_native=as_native,
        test_values=False,
    )


# dtype_bits
@handle_cmd_line_args
@given(
    input_dtype=helpers.get_dtypes("valid", full=False),
    num_positional_args=helpers.num_positional_args(fn_name="dtype_bits"),
)
def test_dtype_bits(
    input_dtype,
    num_positional_args,
    fw,
):
    ret = helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=False,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=True,
        container_flags=False,
        instance_method=False,
        fw=fw,
        fn_name="dtype_bits",
        dtype_in=input_dtype,
        test_values=False,
    )
    if not ivy.exists(ret):
        return
    num_bits, num_bits_np = ret
    assert num_bits == num_bits_np


# is_bool_dtype
@handle_cmd_line_args
@given(
    array=helpers.nph.arrays(
        dtype=dtype_shared,
        shape=helpers.lists(
            arg=helpers.ints(min_value=1, max_value=5),
            min_size="num_dims",
            max_size="num_dims",
            size_bounds=[1, 5],
        ),
    ),
    input_dtype=dtype_shared,
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="is_bool_dtype"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_is_bool_dtype(
    array,
    input_dtype,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="is_bool_dtype",
        dtype_in=array,
        test_values=False,
    )


# is_float_dtype
@handle_cmd_line_args
@given(
    array=helpers.nph.arrays(
        dtype=dtype_shared,
        shape=helpers.lists(
            arg=helpers.ints(min_value=1, max_value=5),
            min_size="num_dims",
            max_size="num_dims",
            size_bounds=[1, 5],
        ),
    ),
    input_dtype=dtype_shared,
    num_positional_args=helpers.num_positional_args(fn_name="is_float_dtype"),
)
def test_is_float_dtype(
    array,
    input_dtype,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="is_float_dtype",
        dtype_in=array,
    )


# is_int_dtype
@handle_cmd_line_args
@given(
    array=helpers.nph.arrays(
        dtype=dtype_shared,
        shape=helpers.lists(
            arg=helpers.ints(min_value=1, max_value=5),
            min_size="num_dims",
            max_size="num_dims",
            size_bounds=[1, 5],
        ),
    ),
    input_dtype=dtype_shared,
    num_positional_args=helpers.num_positional_args(fn_name="is_int_dtype"),
)
def test_is_int_dtype(
    array,
    input_dtype,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="is_int_dtype",
        dtype_in=array,
    )


# promote_types
@handle_cmd_line_args
@given(
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        shared_dtype=False,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="promote_types"),
)
def test_promote_types(
    dtype_and_values,
    as_variable,
    num_positional_args,
    native_array,
    container,
    fw,
):
    types, arrays = dtype_and_values
    type1, type2 = types
    input_dtype = [type1, type2]
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=False,
        fw=fw,
        fn_name="promote_types",
        type1=type1,
        type2=type2,
        test_values=False,
    )


# type_promote_arrays
@handle_cmd_line_args
@given(
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", full=True),
        num_arrays=2,
        shared_dtype=False,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="type_promote_arrays"),
)
def test_type_promote_arrays(
    dtype_and_values,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    types, arrays = dtype_and_values
    type1, type2 = types
    x1, x2 = arrays
    input_dtype = [type1, type2]
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=False,
        instance_method=False,
        fw=fw,
        fn_name="type_promote_arrays",
        x1=np.array(x1),
        x2=np.array(x2),
        test_values=True,
    )


# default_float_dtype
@pytest.mark.parametrize("float_dtype", [ivy.float16, ivy.float32, ivy.float64, None])
@pytest.mark.parametrize(
    "input",
    [
        [(5.0, 25.0), (6.0, 36.0), (7.0, 49.0)],
        np.array([10.0, 0.0, -3.0]),
        10,
        None,
    ],
)
@pytest.mark.parametrize("as_native", [True, False])
def test_default_float_dtype(*, input, float_dtype, as_native):
    res = ivy.default_float_dtype(
        input=input, float_dtype=float_dtype, as_native=as_native
    )
    assert (
        isinstance(res, ivy.Dtype)
        or isinstance(res, ivy.NativeDtype)
        or isinstance(res, str)
    )
    assert (
        ivy.default_float_dtype(input=None, float_dtype=None, as_native=False)
        == ivy.float32
    )
    assert ivy.default_float_dtype(float_dtype=ivy.float16) == ivy.float16
    assert ivy.default_float_dtype() == ivy.float32


# default_int_dtype
@pytest.mark.parametrize("int_dtype", [ivy.int16, ivy.int32, ivy.int64, None])
@pytest.mark.parametrize(
    "input",
    [
        [(5, 25), (6, 36), (7, 49)],
        np.array([10, 0, -3]),
        10,
        10.0,
        None,
    ],
)
@pytest.mark.parametrize("as_native", [True, False])
def test_default_int_dtype(*, input, int_dtype, as_native):
    res = ivy.default_int_dtype(input=input, int_dtype=int_dtype, as_native=as_native)
    assert (
        isinstance(res, ivy.Dtype)
        or isinstance(res, ivy.NativeDtype)
        or isinstance(res, str)
    )
    assert (
        ivy.default_int_dtype(input=None, int_dtype=None, as_native=False) == ivy.int32
    )
    assert ivy.default_int_dtype(int_dtype=ivy.int16) == ivy.int16
    assert ivy.default_int_dtype() == ivy.int32


@st.composite
def dtypes_list(draw):
    num = draw(st.one_of(helpers.ints(min_value=1, max_value=5)))
    return draw(
        st.lists(
            st.sampled_from(ivy.valid_dtypes),
            min_size=num,
            max_size=num,
        )
    )


def _composition_1():
    return ivy.relu().argmax()


def _composition_2():
    a = ivy.floor
    return ivy.ceil() or a


# function_unsupported_dtypes
@pytest.mark.parametrize(
    "func, expected",
    [
        (
            _composition_1,
            [
                "bool",
                "int8",
                "int16",
                "int32",
                "int64",
                "uint8",
                "uint16",
                "uint32",
                "uint64",
                "bfloat16",
                "float16",
                "float32",
                "float64",
            ],
        ),
        (
            _composition_2,
            [
                "bool",
                "int8",
                "int16",
                "int32",
                "int64",
                "uint8",
                "uint16",
                "uint32",
                "uint64",
                "bfloat16",
                "float16",
                "float32",
                "float64",
            ],
        ),
    ],
)
def test_function_supported_dtypes(func, expected):
    res = ivy.function_supported_dtypes(func)
    exp = set.intersection(set(expected), set(ivy.valid_dtypes))

    # Since float16 is only not valid for the torch backend, we remove from expected
    if "torch" in ivy.current_backend_str():
        exp.remove("float16")

    assert sorted(tuple(exp)) == sorted(res)


# function_unsupported_dtypes
@pytest.mark.parametrize(
    "func, expected",
    [
        (_composition_1, []),
        (_composition_2, []),
    ],
)
def test_function_unsupported_dtypes(func, expected):
    res = ivy.function_unsupported_dtypes(func)
    exp = set.union(set(expected), set(ivy.invalid_dtypes))

    # Since float16 is only not valid for the torch backend, we add to expected
    if "torch" in ivy.current_backend_str():
        exp.add("float16")

    assert sorted(tuple(exp)) == sorted(res)


@pytest.mark.parametrize(
    "func_and_version", [
        {"torch": {"cumsum": {"1.11.0": {"bfloat16", "uint8"}, "1.12.1": set()}}},
    ]
)
def test_function_dtype_versioning(func_and_version, fw):
    for key in func_and_version:
        if key != fw:
            continue
        var = ivy.get_backend().version

        for key1 in func_and_version[key]:
            for key2 in func_and_version[key][key1]:
                var['version'] = key2
                fn = getattr(ivy.get_backend(), key1)
                expected = func_and_version[key][key1][key2]
                res = fn.unsupported_dtypes
                if res is None:
                    res=set()
                else: res=set(res)
                if res != expected:
                    print(res, expected)
                    raise Exception
        return True


# invalid_dtype
@handle_cmd_line_args
@given(
    dtype_in=st.sampled_from(ivy.valid_dtypes),
)
def test_invalid_dtype(dtype_in, fw):
    res = ivy.invalid_dtype(dtype_in)
    fw_invalid_dtypes = {
        "torch": ivy_torch.invalid_dtypes,
        "tensorflow": ivy_tf.invalid_dtypes,
        "jax": ivy_jax.invalid_dtypes,
        "numpy": ivy_np.invalid_dtypes,
    }
    if dtype_in in fw_invalid_dtypes[fw]:
        assert res is True, (
            f"fDtype = {dtype_in!r} is a valid dtype for {fw}, but" f"result = {res}"
        )
    else:
        assert res is False, (
            f"fDtype = {dtype_in!r} is not a valid dtype for {fw}, but"
            f"result = {res}"
        )


# unset_default_dtype()
@handle_cmd_line_args
@given(
    dtype=st.sampled_from(ivy.valid_dtypes),
)
def test_unset_default_dtype(dtype):
    stack_size_before = len(ivy.default_dtype_stack)
    ivy.set_default_dtype(dtype)
    ivy.unset_default_dtype()
    stack_size_after = len(ivy.default_dtype_stack)
    assert (
        stack_size_before == stack_size_after
    ), f"Default dtype not unset. Stack size= {stack_size_after!r}"


# unset_default_float_dtype()
@handle_cmd_line_args
@given(
    dtype=st.sampled_from(ivy.valid_float_dtypes),
)
def test_unset_default_float_dtype(dtype):
    stack_size_before = len(ivy.default_float_dtype_stack)
    ivy.set_default_float_dtype(dtype)
    ivy.unset_default_float_dtype()
    stack_size_after = len(ivy.default_float_dtype_stack)
    assert (
        stack_size_before == stack_size_after
    ), f"Default float dtype not unset. Stack size= {stack_size_after!r}"


# unset_default_int_dtype()
@handle_cmd_line_args
@given(
    dtype=st.sampled_from(ivy.valid_int_dtypes),
)
def test_unset_default_int_dtype(dtype):
    stack_size_before = len(ivy.default_int_dtype_stack)
    ivy.set_default_int_dtype(dtype)
    ivy.unset_default_int_dtype()
    stack_size_after = len(ivy.default_int_dtype_stack)
    assert (
        stack_size_before == stack_size_after
    ), f"Default int dtype not unset. Stack size= {stack_size_after!r}"


# valid_dtype
@handle_cmd_line_args
@given(
    dtype_in=st.sampled_from(ivy.valid_dtypes),
)
def test_valid_dtype(dtype_in, fw):
    res = ivy.valid_dtype(dtype_in)
    fw_valid_dtypes = {
        "torch": ivy_torch.valid_dtypes,
        "tensorflow": ivy_tf.valid_dtypes,
        "jax": ivy_jax.valid_dtypes,
        "numpy": ivy_np.valid_dtypes,
    }
    if dtype_in in fw_valid_dtypes[fw]:
        assert res is True, (
            f"fDtype = {dtype_in!r} is not a valid dtype for {fw}, but"
            f"result = {res}"
        )
    else:
        assert res is False, (
            f"fDtype = {dtype_in!r} is a valid dtype for {fw}, but" f"result = {res}"
        )
