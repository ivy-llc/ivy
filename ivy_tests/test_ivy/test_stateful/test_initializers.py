# global
from hypothesis import given, strategies as st
import ivy

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@handle_cmd_line_args
@given(
    var_shape=helpers.get_shape(),
    constant=helpers.floats(
        large_abs_safety_factor=4, small_abs_safety_factor=4, safety_factor_scale="log"
    ),
    init_with_v=st.booleans(),
    method_with_v=st.booleans(),
)
def test_constant(
    var_shape,
    constant,
    init_with_v,
    method_with_v,
    as_variable,
    native_array,
):
    ret_ivy, ret_gt = helpers.test_method(
        input_dtypes_init=[],
        as_variable_flags_init=[],
        num_positional_args_init=0,
        native_array_flags_init=[],
        all_as_kwargs_np_init={"constant": constant},
        input_dtypes_method=[],
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        container_flags_method=[False],
        all_as_kwargs_np_method={
            "var_shape": var_shape,
            "device": "cpu",
        },
        class_name="Constant",
        method_name="create_variables",
        init_with_v=init_with_v,
        method_with_v=method_with_v,
        test_values=False,
        ground_truth_backend="numpy",
    )

    assert ret_ivy.shape == ret_gt.shape
    assert ret_ivy.dtype == ret_gt.dtype
    assert ivy.all(ivy.equal(ret_ivy, ivy.array(constant)))


@handle_cmd_line_args
@given(
    var_shape=helpers.get_shape(),
    init_with_v=st.booleans(),
    method_with_v=st.booleans(),
)
def test_zeros(var_shape, init_with_v, method_with_v, as_variable, native_array):
    ret_ivy, ret_gt = helpers.test_method(
        input_dtypes_init=[],
        as_variable_flags_init=[],
        num_positional_args_init=0,
        native_array_flags_init=[],
        all_as_kwargs_np_init={},
        input_dtypes_method=[],
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        container_flags_method=[False],
        all_as_kwargs_np_method={
            "var_shape": var_shape,
            "device": "cpu",
        },
        class_name="Zeros",
        method_name="create_variables",
        init_with_v=init_with_v,
        method_with_v=method_with_v,
        test_values=False,
        ground_truth_backend="numpy",
    )

    assert ret_ivy.shape == ret_gt.shape
    assert ret_ivy.dtype == ret_gt.dtype
    assert ivy.all(ivy.equal(ret_ivy, ivy.array(0.0)))


@handle_cmd_line_args
@given(
    var_shape=helpers.get_shape(),
    init_with_v=st.booleans(),
    method_with_v=st.booleans(),
)
def test_ones(var_shape, init_with_v, method_with_v, as_variable, native_array):
    ret_ivy, ret_gt = helpers.test_method(
        input_dtypes_init=[],
        as_variable_flags_init=[],
        num_positional_args_init=0,
        native_array_flags_init=[],
        all_as_kwargs_np_init={},
        input_dtypes_method=[],
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        container_flags_method=[False],
        all_as_kwargs_np_method={
            "var_shape": var_shape,
            "device": "cpu",
        },
        class_name="Ones",
        method_name="create_variables",
        init_with_v=init_with_v,
        method_with_v=method_with_v,
        test_values=False,
        ground_truth_backend="numpy",
    )

    assert ret_ivy.shape == ret_gt.shape
    assert ret_ivy.dtype == ret_gt.dtype
    assert ivy.all(ivy.equal(ret_ivy, ivy.array(1.0)))


@handle_cmd_line_args
@given(
    numerator=helpers.floats(min_value=1.0, max_value=10.0),
    fan_mode=st.sampled_from(["fan_in", "fan_out", "fan_sum", "fan_avg"]),
    power=helpers.floats(min_value=0.1, max_value=3.0),
    gain=helpers.floats(min_value=0.1, max_value=10.0),
    var_shape=helpers.get_shape(),
    fan_in=helpers.ints(min_value=1, safety_factor=4),
    fan_out=helpers.ints(min_value=1, safety_factor=4),
    init_with_v=st.booleans(),
    method_with_v=st.booleans(),
)
def test_uniform(
    numerator,
    fan_mode,
    power,
    gain,
    var_shape,
    fan_in,
    fan_out,
    init_with_v,
    method_with_v,
    as_variable,
    native_array,
):
    ret_ivy, ret_gt = helpers.test_method(
        input_dtypes_init=[],
        as_variable_flags_init=[],
        num_positional_args_init=0,
        native_array_flags_init=[],
        all_as_kwargs_np_init={
            "numerator": numerator,
            "fan_mode": fan_mode,
            "power": power,
            "gain": gain,
        },
        input_dtypes_method=[],
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        container_flags_method=[False],
        all_as_kwargs_np_method={
            "var_shape": var_shape,
            "device": "cpu",
            "fan_in": fan_in,
            "fan_out": fan_out,
        },
        class_name="Uniform",
        method_name="create_variables",
        init_with_v=init_with_v,
        method_with_v=method_with_v,
        test_values=False,
        ground_truth_backend="numpy",
    )
    if fan_mode == "fan_in":
        fan = fan_in
    elif fan_mode == "fan_out":
        fan = fan_out
    elif fan_mode == "fan_sum":
        fan = fan_in + fan_out
    elif fan_mode == "fan_avg":
        fan = (fan_in + fan_out) / 2

    bound = gain * (numerator / fan) ** power
    assert ret_ivy.shape == ret_gt.shape
    assert ret_ivy.dtype == ret_gt.dtype
    assert ivy.all(ivy.less(ivy.abs(ret_ivy), ivy.array(bound)))


@handle_cmd_line_args
@given(
    var_shape=helpers.get_shape(),
    fan_in=helpers.ints(min_value=1),
    fan_out=helpers.ints(min_value=1),
    init_with_v=st.booleans(),
    method_with_v=st.booleans(),
)
def test_glorot_uniform(
    var_shape,
    fan_in,
    fan_out,
    init_with_v,
    method_with_v,
    as_variable,
    native_array,
):
    ret_ivy, ret_gt = helpers.test_method(
        input_dtypes_init=[],
        as_variable_flags_init=[],
        num_positional_args_init=0,
        native_array_flags_init=[],
        all_as_kwargs_np_init={},
        input_dtypes_method=[],
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        container_flags_method=[False],
        all_as_kwargs_np_method={
            "var_shape": var_shape,
            "device": "cpu",
            "fan_in": fan_in,
            "fan_out": fan_out,
        },
        class_name="GlorotUniform",
        method_name="create_variables",
        init_with_v=init_with_v,
        method_with_v=method_with_v,
        test_values=False,
        ground_truth_backend="numpy",
    )

    bound = (6 / (fan_in + fan_out)) ** 0.5
    assert ret_ivy.shape == ret_gt.shape
    assert ret_ivy.dtype == ret_gt.dtype
    assert ivy.all(ivy.less(ivy.abs(ret_ivy), ivy.array(bound)))


@handle_cmd_line_args
@given(
    var_shape=helpers.get_shape(),
    fan_in=helpers.ints(
        min_value=1,
    ),
    init_with_v=st.booleans(),
    method_with_v=st.booleans(),
)
def test_first_layer_siren(
    var_shape,
    fan_in,
    init_with_v,
    method_with_v,
    as_variable,
    native_array,
):
    ret_ivy, ret_gt = helpers.test_method(
        input_dtypes_init=[],
        as_variable_flags_init=[],
        num_positional_args_init=0,
        native_array_flags_init=[],
        all_as_kwargs_np_init={},
        input_dtypes_method=[],
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        container_flags_method=[False],
        all_as_kwargs_np_method={
            "var_shape": var_shape,
            "device": "cpu",
            "fan_in": fan_in,
        },
        class_name="FirstLayerSiren",
        method_name="create_variables",
        init_with_v=init_with_v,
        method_with_v=method_with_v,
        test_values=False,
        ground_truth_backend="numpy",
    )

    bound = fan_in
    assert ret_ivy.shape == ret_gt.shape
    assert ret_ivy.dtype == ret_gt.dtype
    assert ivy.all(ivy.less(ivy.abs(ret_ivy), ivy.array(bound)))


@handle_cmd_line_args
@given(
    var_shape=helpers.get_shape(),
    w0=helpers.floats(min_value=1.0, max_value=100.0),
    fan_in=st.integers(min_value=1),
    init_with_v=st.booleans(),
    method_with_v=st.booleans(),
)
def test_siren(
    var_shape,
    w0,
    fan_in,
    init_with_v,
    method_with_v,
    as_variable,
    native_array,
):
    ret_ivy, ret_gt = helpers.test_method(
        input_dtypes_init=[],
        as_variable_flags_init=[],
        num_positional_args_init=0,
        native_array_flags_init=[],
        all_as_kwargs_np_init={"w0": w0},
        input_dtypes_method=[],
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        container_flags_method=[False],
        all_as_kwargs_np_method={
            "var_shape": var_shape,
            "device": "cpu",
            "fan_in": fan_in,
        },
        class_name="Siren",
        method_name="create_variables",
        init_with_v=init_with_v,
        method_with_v=method_with_v,
        test_values=False,
        ground_truth_backend="numpy",
    )

    bound = ((6 / fan_in) ** 0.5) / w0
    assert ret_ivy.shape == ret_gt.shape
    assert ret_ivy.dtype == ret_gt.dtype
    assert ivy.all(ivy.less(ivy.abs(ret_ivy), ivy.array(bound)))


@handle_cmd_line_args
@given(
    mean=helpers.floats(large_abs_safety_factor=4, small_abs_safety_factor=4),
    fan_mode=st.sampled_from(["fan_in", "fan_out", "fan_sum", "fan_avg"]),
    var_shape=helpers.get_shape(),
    fan_in=helpers.ints(min_value=1, safety_factor=4),
    fan_out=helpers.ints(min_value=1, safety_factor=4),
    negative_slope=helpers.floats(min_value=0.0, max_value=5.0),
    # should be replaced with helpers.get_dtypes() but somehow it causes inconsistent data generation # noqa
    dtype=st.sampled_from([None, "float64", "float32", "float16"]),
    init_with_v=st.booleans(),
    method_with_v=st.booleans(),
)
def test_kaiming_normal(
    mean,
    fan_mode,
    var_shape,
    fan_in,
    fan_out,
    negative_slope,
    dtype,
    init_with_v,
    method_with_v,
    as_variable,
    native_array,
):
    ret_ivy, ret_gt = helpers.test_method(
        input_dtypes_init=[],
        as_variable_flags_init=[],
        num_positional_args_init=0,
        native_array_flags_init=[],
        all_as_kwargs_np_init={
            "mean": mean,
            "fan_mode": fan_mode,
        },
        input_dtypes_method=[],
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        container_flags_method=[False],
        all_as_kwargs_np_method={
            "var_shape": var_shape,
            "device": "cpu",
            "fan_in": fan_in,
            "fan_out": fan_out,
            "negative_slope": negative_slope,
            "dtype": dtype,
        },
        class_name="KaimingNormal",
        method_name="create_variables",
        init_with_v=init_with_v,
        method_with_v=method_with_v,
        test_values=False,
        ground_truth_backend="numpy",
    )
    assert ret_ivy.shape == ret_gt.shape
    assert ret_ivy.dtype == ret_gt.dtype
