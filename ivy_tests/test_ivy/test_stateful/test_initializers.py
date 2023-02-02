# global
from hypothesis import strategies as st
import ivy

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_method


@handle_method(
    method_tree="Constant.create_variables",
    ground_truth_backend="numpy",
    var_shape=helpers.get_shape(),
    constant=helpers.floats(
        large_abs_safety_factor=4, small_abs_safety_factor=4, safety_factor_scale="log"
    ),
    init_with_v=st.booleans(),
    method_with_v=st.booleans(),
    init_as_variable_flags=st.just([False]),
    init_num_positional_args=st.just(0),
    init_native_arrays=st.just([False]),
    method_as_variable_flags=st.just([False]),
    method_num_positional_args=st.just(0),
    method_native_arrays=st.just([False]),
    method_container_flags=st.just([False]),
)
def test_constant(
    var_shape,
    constant,
    init_with_v,
    method_with_v,
    class_name,
    method_name,
    ground_truth_backend,
    init_flags,
    method_flags,
):
    ret_ivy, ret_gt = helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_input_dtypes=[],
        init_all_as_kwargs_np={"constant": constant},
        method_input_dtypes=[],
        method_all_as_kwargs_np={
            "var_shape": var_shape,
            "device": "cpu",
        },
        class_name=class_name,
        method_name=method_name,
        init_with_v=init_with_v,
        method_with_v=method_with_v,
        test_values=False,
    )

    assert ret_ivy.shape == ret_gt.shape
    assert ret_ivy.dtype == ret_gt.dtype
    assert ivy.all(ivy.equal(ret_ivy, ivy.array(constant)))


@handle_method(
    method_tree="Zeros.create_variables",
    ground_truth_backend="numpy",
    var_shape=helpers.get_shape(),
    init_with_v=st.booleans(),
    method_with_v=st.booleans(),
    init_as_variable_flags=st.just([False]),
    init_num_positional_args=st.just(0),
    init_native_arrays=st.just([False]),
    method_as_variable_flags=st.just([False]),
    method_num_positional_args=st.just(0),
    method_native_arrays=st.just([False]),
    method_container_flags=st.just([False]),
)
def test_zeros(
    var_shape,
    init_with_v,
    method_with_v,
    class_name,
    method_name,
    ground_truth_backend,
    init_flags,
    method_flags,
):
    ret_ivy, ret_gt = helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_input_dtypes=[],
        init_all_as_kwargs_np={},
        method_input_dtypes=[],
        method_all_as_kwargs_np={
            "var_shape": var_shape,
            "device": "cpu",
        },
        class_name=class_name,
        method_name=method_name,
        init_with_v=init_with_v,
        method_with_v=method_with_v,
        test_values=False,
    )

    assert ret_ivy.shape == ret_gt.shape
    assert ret_ivy.dtype == ret_gt.dtype
    assert ivy.all(ivy.equal(ret_ivy, ivy.array(0.0)))


@handle_method(
    method_tree="Ones.create_variables",
    ground_truth_backend="numpy",
    var_shape=helpers.get_shape(),
    init_with_v=st.booleans(),
    method_with_v=st.booleans(),
    init_as_variable_flags=st.just([False]),
    init_num_positional_args=st.just(0),
    init_native_arrays=st.just([False]),
    method_as_variable_flags=st.just([False]),
    method_num_positional_args=st.just(0),
    method_native_arrays=st.just([False]),
    method_container_flags=st.just([False]),
)
def test_ones(
    var_shape,
    init_with_v,
    method_with_v,
    class_name,
    method_name,
    ground_truth_backend,
    init_flags,
    method_flags,
):
    ret_ivy, ret_gt = helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_input_dtypes=[],
        init_all_as_kwargs_np={},
        method_input_dtypes=[],
        method_all_as_kwargs_np={
            "var_shape": var_shape,
            "device": "cpu",
        },
        class_name=class_name,
        method_name=method_name,
        init_with_v=init_with_v,
        method_with_v=method_with_v,
        test_values=False,
    )

    assert ret_ivy.shape == ret_gt.shape
    assert ret_ivy.dtype == ret_gt.dtype
    assert ivy.all(ivy.equal(ret_ivy, ivy.array(1.0)))


@handle_method(
    method_tree="Uniform.create_variables",
    ground_truth_backend="numpy",
    numerator=helpers.floats(min_value=1.0, max_value=10.0),
    fan_mode=st.sampled_from(["fan_in", "fan_out", "fan_sum", "fan_avg"]),
    power=helpers.floats(min_value=1.0, max_value=3.0),
    gain=helpers.floats(min_value=1.0, max_value=10.0),
    var_shape=helpers.get_shape(),
    fan_in=helpers.ints(min_value=1, max_value=100),
    fan_out=helpers.ints(min_value=1, max_value=100),
    init_with_v=st.booleans(),
    method_with_v=st.booleans(),
    init_as_variable_flags=st.just([False]),
    init_num_positional_args=st.just(0),
    init_native_arrays=st.just([False]),
    method_as_variable_flags=st.just([False]),
    method_num_positional_args=st.just(0),
    method_native_arrays=st.just([False]),
    method_container_flags=st.just([False]),
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
    class_name,
    method_name,
    ground_truth_backend,
    init_flags,
    method_flags,
):
    ret_ivy, ret_gt = helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_input_dtypes=[],
        init_all_as_kwargs_np={
            "numerator": numerator,
            "fan_mode": fan_mode,
            "power": power,
            "gain": gain,
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={
            "var_shape": var_shape,
            "device": "cpu",
            "fan_in": fan_in,
            "fan_out": fan_out,
        },
        class_name=class_name,
        method_name=method_name,
        init_with_v=init_with_v,
        method_with_v=method_with_v,
        test_values=False,
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


@handle_method(
    method_tree="GlorotUniform.create_variables",
    ground_truth_backend="numpy",
    var_shape=helpers.get_shape(),
    fan_in=helpers.ints(min_value=1, max_value=100),
    fan_out=helpers.ints(min_value=1, max_value=100),
    init_with_v=st.booleans(),
    method_with_v=st.booleans(),
    init_as_variable_flags=st.just([False]),
    init_num_positional_args=st.just(0),
    init_native_arrays=st.just([False]),
    method_as_variable_flags=st.just([False]),
    method_num_positional_args=st.just(0),
    method_native_arrays=st.just([False]),
    method_container_flags=st.just([False]),
)
def test_glorot_uniform(
    var_shape,
    fan_in,
    fan_out,
    init_with_v,
    method_with_v,
    class_name,
    method_name,
    ground_truth_backend,
    init_flags,
    method_flags,
):
    ret_ivy, ret_gt = helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_input_dtypes=[],
        init_all_as_kwargs_np={},
        method_input_dtypes=[],
        method_all_as_kwargs_np={
            "var_shape": var_shape,
            "device": "cpu",
            "fan_in": fan_in,
            "fan_out": fan_out,
        },
        class_name=class_name,
        method_name=method_name,
        init_with_v=init_with_v,
        method_with_v=method_with_v,
        test_values=False,
    )

    bound = (6 / (fan_in + fan_out)) ** 0.5
    assert ret_ivy.shape == ret_gt.shape
    assert ret_ivy.dtype == ret_gt.dtype
    assert ivy.all(ivy.less(ivy.abs(ret_ivy), ivy.array(bound)))


@handle_method(
    method_tree="FirstLayerSiren.create_variables",
    ground_truth_backend="jax",
    var_shape=helpers.get_shape(),
    fan_in=helpers.ints(
        min_value=1,
        safety_factor=4,
        safety_factor_scale="log",
    ),
    init_with_v=st.booleans(),
    method_with_v=st.booleans(),
    init_as_variable_flags=st.just([False]),
    init_num_positional_args=st.just(0),
    init_native_arrays=st.just([False]),
    method_as_variable_flags=st.just([False]),
    method_num_positional_args=st.just(0),
    method_native_arrays=st.just([False]),
    method_container_flags=st.just([False]),
)
def test_first_layer_siren(
    var_shape,
    fan_in,
    init_with_v,
    method_with_v,
    class_name,
    method_name,
    ground_truth_backend,
    init_flags,
    method_flags,
):
    ret_ivy, ret_gt = helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_input_dtypes=[],
        init_all_as_kwargs_np={},
        method_input_dtypes=[],
        method_all_as_kwargs_np={
            "var_shape": var_shape,
            "device": "cpu",
            "fan_in": fan_in,
        },
        class_name=class_name,
        method_name=method_name,
        init_with_v=init_with_v,
        method_with_v=method_with_v,
        test_values=False,
    )

    bound = fan_in
    assert ret_ivy.shape == ret_gt.shape
    assert ret_ivy.dtype == ret_gt.dtype
    assert ivy.all(ivy.less(ivy.abs(ret_ivy), ivy.array(bound)))


@handle_method(
    method_tree="Siren.create_variables",
    ground_truth_backend="numpy",
    var_shape=helpers.get_shape(),
    w0=helpers.floats(min_value=1.0, max_value=100.0),
    fan_in=st.integers(min_value=1),
    init_with_v=st.booleans(),
    method_with_v=st.booleans(),
    init_as_variable_flags=st.just([False]),
    init_native_arrays=st.just([False]),
    init_num_positional_args=st.just(0),
    method_as_variable_flags=st.just([False]),
    method_native_arrays=st.just([False]),
    method_container_flags=st.just([False]),
    method_num_positional_args=st.just(0),
)
def test_siren(
    var_shape,
    w0,
    fan_in,
    init_with_v,
    method_with_v,
    class_name,
    method_name,
    ground_truth_backend,
    init_flags,
    method_flags,
):
    ret_ivy, ret_gt = helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_input_dtypes=[],
        init_all_as_kwargs_np={"w0": w0},
        method_input_dtypes=[],
        method_all_as_kwargs_np={
            "var_shape": var_shape,
            "device": "cpu",
            "fan_in": fan_in,
        },
        class_name=class_name,
        method_name=method_name,
        init_with_v=init_with_v,
        method_with_v=method_with_v,
        test_values=False,
    )

    bound = ((6 / fan_in) ** 0.5) / w0
    assert ret_ivy.shape == ret_gt.shape
    assert ret_ivy.dtype == ret_gt.dtype
    assert ivy.all(ivy.less(ivy.abs(ret_ivy), ivy.array(bound)))


@handle_method(
    method_tree="KaimingNormal.create_variables",
    mean=helpers.floats(
        min_value=-1e5,
        max_value=1e5,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
    fan_mode=st.sampled_from(["fan_in", "fan_out", "fan_sum", "fan_avg"]),
    var_shape=helpers.get_shape(),
    fan_in=helpers.ints(min_value=1, safety_factor=8, safety_factor_scale="log"),
    fan_out=helpers.ints(min_value=1, safety_factor=8, safety_factor_scale="log"),
    negative_slope=helpers.floats(
        min_value=1e-5,
        max_value=5.0,
    ),
    # should be replaced with helpers.get_dtypes() but somehow it causes inconsistent data generation # noqa
    dtype=st.sampled_from([None, "float64", "float32", "float16"]),
    init_with_v=st.booleans(),
    method_with_v=st.booleans(),
    ground_truth_backend="numpy",
    init_as_variable_flags=st.just([False]),
    init_num_positional_args=st.just(0),
    init_native_arrays=st.just([False]),
    method_as_variable_flags=st.just([False]),
    method_num_positional_args=st.just(0),
    method_native_arrays=st.just([False]),
    method_container_flags=st.just([False]),
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
    class_name,
    method_name,
    ground_truth_backend,
    init_flags,
    method_flags,
):
    ret_ivy, ret_gt = helpers.test_method(
        ground_truth_backend=ground_truth_backend,
        init_flags=init_flags,
        method_flags=method_flags,
        init_input_dtypes=[],
        init_all_as_kwargs_np={
            "mean": mean,
            "fan_mode": fan_mode,
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={
            "var_shape": var_shape,
            "device": "cpu",
            "fan_in": fan_in,
            "fan_out": fan_out,
            "negative_slope": negative_slope,
            "dtype": dtype,
        },
        class_name=class_name,
        method_name=method_name,
        init_with_v=init_with_v,
        method_with_v=method_with_v,
        test_values=False,
    )
    assert ret_ivy.shape == ret_gt.shape
    assert ret_ivy.dtype == ret_gt.dtype
