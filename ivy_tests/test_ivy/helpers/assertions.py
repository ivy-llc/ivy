import ivy
import numpy as np

TOLERANCE_DICT = {
    "float16": 1e-2,
    "bfloat16": 1e-2,
    "float32": 1e-5,
    "float64": 1e-5,
    None: 1e-5,
}


def assert_all_close(
    ret_np,
    ret_from_gt_np,
    backend: str,
    rtol=1e-05,
    atol=1e-08,
    ground_truth_backend="TensorFlow",
):
    """Match the ret_np and ret_from_gt_np inputs element-by-element to ensure
    that they are the same.

    Parameters
    ----------
    ret_np
        Return from the framework to test. Ivy Container or Numpy Array.
    ret_from_gt_np
        Return from the ground truth framework. Ivy Container or Numpy Array.
    rtol
        Relative Tolerance Value.
    atol
        Absolute Tolerance Value.
    ground_truth_backend
        Ground Truth Backend Framework.

    Returns
    -------
    None if the test passes, else marks the test as failed.
    """
    ret_dtype = str(ret_np.dtype)
    ret_from_gt_dtype = str(ret_from_gt_np.dtype).replace("longlong", "int64")
    assert ret_dtype == ret_from_gt_dtype, (
        f"the ground truth framework {ground_truth_backend} returned a"
        f" {ret_from_gt_dtype} datatype while the backend {backend} returned a"
        f" {ret_dtype} datatype"
    )
    # TODO enable
    # if ivy.is_ivy_container(ret_np) and ivy.is_ivy_container(ret_from_gt_np):
    #     ivy.Container.cont_multi_map(assert_all_close, [ret_np, ret_from_gt_np])
    # else:
    if ret_np.dtype == "bfloat16" or ret_from_gt_np.dtype == "bfloat16":
        ret_np = ret_np.astype("float64")
        ret_from_gt_np = ret_from_gt_np.astype("float64")
    assert np.allclose(
        np.nan_to_num(ret_np), np.nan_to_num(ret_from_gt_np), rtol=rtol, atol=atol
    ), (
        f" the results from backend {backend} "
        f"and ground truth framework {ground_truth_backend} "
        f"do not match\n {ret_np}!={ret_from_gt_np} \n\n"
        "The mismatching elements are at `False` indices:\n\n"
        f"{ret_np == ret_from_gt_np} \n\n"
    )


def assert_same_type_and_shape(values, this_key_chain=None):
    x_, y_ = values
    for x, y in zip(x_, y_):
        if isinstance(x, np.ndarray):
            x_d = str(x.dtype).replace("longlong", "int64")
            y_d = str(y.dtype).replace("longlong", "int64")
            assert (
                x.shape == y.shape
            ), f"returned shape = {x.shape}, ground-truth returned shape = {y.shape}"
            assert (
                x_d == y_d
            ), f"returned dtype = {x_d}, ground-truth returned dtype = {y_d}"


def assert_same_type(ret_from_target, ret_from_gt, backend_to_test, gt_backend):
    """Assert that the return types from the target and ground truth frameworks
    are the same.

    checks with a string comparison because with_backend returns
    different objects. Doesn't check recursively.
    """

    def _assert_same_type(x, y):
        assert_msg = (
            f"ground truth backend ({gt_backend}) returned"
            f" {type(y)} but target backend ({backend_to_test}) returned"
            f" {type(x)}"
        )
        assert str(type(x)) == str(type(y)), assert_msg

    ivy.nested_multi_map(
        lambda x, _: _assert_same_type(x[0], x[1]), [ret_from_target, ret_from_gt]
    )


def value_test(
    *,
    ret_np_flat,
    ret_np_from_gt_flat,
    rtol=None,
    atol=1e-6,
    specific_tolerance_dict=None,
    backend: str,
    ground_truth_backend="TensorFlow",
):
    """Perform a value test for matching the arrays in ret_np_flat and
    ret_from_np_gt_flat.

    Parameters
    ----------
    ret_np_flat
        A list (flattened) containing Numpy arrays. Return from the
        framework to test.
    ret_np_from_gt_flat
        A list (flattened) containing Numpy arrays. Return from the ground
        truth framework.
    rtol
        Relative Tolerance Value.
    atol
        Absolute Tolerance Value.
    specific_tolerance_dict
        (Optional) Dictionary of specific rtol and atol values according to the dtype.
    ground_truth_backend
        Ground Truth Backend Framework.

    Returns
    -------
    None if the value test passes, else marks the test as failed.
    """
    assert_same_type_and_shape([ret_np_flat, ret_np_from_gt_flat])

    if type(ret_np_flat) != list:  # noqa: E721
        ret_np_flat = [ret_np_flat]
    if type(ret_np_from_gt_flat) != list:  # noqa: E721
        ret_np_from_gt_flat = [ret_np_from_gt_flat]
    assert len(ret_np_flat) == len(ret_np_from_gt_flat), (
        f"The length of results from backend {backend} and ground truth framework"
        f" {ground_truth_backend} does not match\n\nlen(ret_np_flat) !="
        f" len(ret_np_from_gt_flat):\n\nret_np_flat:\n\n{ret_np_flat}\n\n"
        f"ret_np_from_gt_flat:\n\n{ret_np_from_gt_flat}"
    )
    # value tests, iterating through each array in the flattened returns
    if specific_tolerance_dict is not None:
        for ret_np, ret_np_from_gt in zip(ret_np_flat, ret_np_from_gt_flat):
            dtype = str(ret_np_from_gt.dtype)
            if specific_tolerance_dict.get(dtype) is not None:
                rtol = specific_tolerance_dict.get(dtype)
            else:
                rtol = TOLERANCE_DICT.get(dtype, 1e-03) if rtol is None else rtol
            assert_all_close(
                ret_np,
                ret_np_from_gt,
                backend=backend,
                rtol=rtol,
                atol=atol,
                ground_truth_backend=ground_truth_backend,
            )
    elif rtol is not None:
        for ret_np, ret_np_from_gt in zip(ret_np_flat, ret_np_from_gt_flat):
            assert_all_close(
                ret_np,
                ret_np_from_gt,
                backend=backend,
                rtol=rtol,
                atol=atol,
                ground_truth_backend=ground_truth_backend,
            )
    else:
        for ret_np, ret_np_from_gt in zip(ret_np_flat, ret_np_from_gt_flat):
            rtol = TOLERANCE_DICT.get(str(ret_np_from_gt.dtype), 1e-03)
            assert_all_close(
                ret_np,
                ret_np_from_gt,
                backend=backend,
                rtol=rtol,
                atol=atol,
                ground_truth_backend=ground_truth_backend,
            )


def check_unsupported_dtype(*, fn, input_dtypes, all_as_kwargs_np):
    """Check whether a function does not support the input data types or the
    output data type.

    Parameters
    ----------
    fn
        The function to check.
    input_dtypes
        data-types of the input arguments and keyword-arguments.
    all_as_kwargs_np
        All arguments in Numpy Format, to check for the presence of dtype argument.

    Returns
    -------
    True if the function does not support the given input or output data types, False
    otherwise.
    """
    test_unsupported = False
    unsupported_dtypes_fn = ivy.function_unsupported_dtypes(fn)
    supported_dtypes_fn = ivy.function_supported_dtypes(fn)
    if unsupported_dtypes_fn:
        for d in input_dtypes:
            if d in unsupported_dtypes_fn:
                test_unsupported = True
                break
        if (
            "dtype" in all_as_kwargs_np
            and all_as_kwargs_np["dtype"] is not None
            and all_as_kwargs_np["dtype"] in unsupported_dtypes_fn
        ):
            test_unsupported = True
    if supported_dtypes_fn and not test_unsupported:
        for d in input_dtypes:
            if d not in supported_dtypes_fn:
                test_unsupported = True
                break
        if (
            "dtype" in all_as_kwargs_np
            and all_as_kwargs_np["dtype"] is not None
            and all_as_kwargs_np["dtype"] not in supported_dtypes_fn
        ):
            test_unsupported = True
    return test_unsupported


def check_unsupported_device(*, fn, input_device, all_as_kwargs_np):
    """Check whether a function does not support a given device.

    Parameters
    ----------
    fn
        The function to check.
    input_device
        The backend device.
    all_as_kwargs_np
        All arguments in Numpy Format, to check for the presence of dtype argument.

    Returns
    -------
    True if the function does not support the given device, False otherwise.
    """
    test_unsupported = False
    unsupported_devices_fn = ivy.function_unsupported_devices(fn)
    supported_devices_fn = ivy.function_supported_devices(fn)
    if unsupported_devices_fn:
        if input_device in unsupported_devices_fn:
            test_unsupported = True
        if (
            "device" in all_as_kwargs_np
            and all_as_kwargs_np["device"] in unsupported_devices_fn
        ):
            test_unsupported = True
    if supported_devices_fn and not test_unsupported:
        if input_device not in supported_devices_fn:
            test_unsupported = True
        if (
            "device" in all_as_kwargs_np
            and all_as_kwargs_np["device"] not in supported_devices_fn
        ):
            test_unsupported = True
    return test_unsupported


def check_unsupported_device_and_dtype(*, fn, device, input_dtypes, all_as_kwargs_np):
    """Check whether a function does not support a given device or data types.

    Parameters
    ----------
    fn
        The function to check.
    device
        The backend device to check.
    input_dtypes
        data-types of the input arguments and keyword-arguments.
    all_as_kwargs_np
        All arguments in Numpy Format, to check for the presence of dtype argument.

    Returns
    -------
    True if the function does not support both the device and any data type, False
    otherwise.
    """
    unsupported_devices_dtypes_fn = ivy.function_unsupported_devices_and_dtypes(fn)

    if device in unsupported_devices_dtypes_fn:
        for d in input_dtypes:
            if d in unsupported_devices_dtypes_fn[device]:
                return True

    if "device" in all_as_kwargs_np and "dtype" in all_as_kwargs_np:
        dev = all_as_kwargs_np["device"]
        dtype = all_as_kwargs_np["dtype"]
        if dtype in unsupported_devices_dtypes_fn.get(dev, []):
            return True

    return False


def test_unsupported_function(*, fn, args, kwargs):
    """Test a function with an unsupported datatype to raise an exception.

    Parameters
    ----------
    fn
        callable function to test.
    args
        arguments to the function.
    kwargs
        keyword-arguments to the function.
    """
    try:
        fn(*args, **kwargs)
        assert False
    except:  # noqa
        return
