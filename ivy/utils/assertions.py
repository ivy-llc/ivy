import numpy as np
import ivy


# Helpers #
# ------- #


def _broadcast_inputs(x1, x2):
    x1_, x2_ = x1, x2
    iterables = (list, tuple, ivy.Shape)
    if not isinstance(x1_, iterables):
        x1_, x2_ = x2, x1
        if not isinstance(x1_, iterables):
            return [x1], [x2]
    if not isinstance(x2_, iterables):
        x1 = [x1] * len(x2)
    return x1, x2


# General with Custom Message #
# --------------------------- #


def check_less(x1, x2, allow_equal=False, message="", as_array=True):
    comp_fn = lambda x1, x2: (ivy.any(x1 > x2), ivy.any(x1 >= x2))
    if not as_array:
        iter_comp_fn = lambda x1_, x2_: (
            any(x1 > x2 for x1, x2 in zip(x1_, x2_)),
            any(x1 >= x2 for x1, x2 in zip(x1_, x2_)),
        )
        comp_fn = lambda x1, x2: iter_comp_fn(*_broadcast_inputs(x1, x2))
    gt, gt_eq = comp_fn(x1, x2)
    # less_equal
    if allow_equal and gt:
        raise ivy.utils.exceptions.IvyException(
            "{} must be lesser than or equal to {}".format(x1, x2)
            if message == ""
            else message
        )
    # less
    elif not allow_equal and gt_eq:
        raise ivy.utils.exceptions.IvyException(
            "{} must be lesser than {}".format(x1, x2) if message == "" else message
        )


def check_greater(x1, x2, allow_equal=False, message="", as_array=True):
    comp_fn = lambda x1, x2: (ivy.any(x1 < x2), ivy.any(x1 <= x2))
    if not as_array:
        iter_comp_fn = lambda x1_, x2_: (
            any(x1 < x2 for x1, x2 in zip(x1_, x2_)),
            any(x1 <= x2 for x1, x2 in zip(x1_, x2_)),
        )
        comp_fn = lambda x1, x2: iter_comp_fn(*_broadcast_inputs(x1, x2))
    lt, lt_eq = comp_fn(x1, x2)
    # greater_equal
    if allow_equal and lt:
        raise ivy.utils.exceptions.IvyException(
            "{} must be greater than or equal to {}".format(x1, x2)
            if message == ""
            else message
        )
    # greater
    elif not allow_equal and lt_eq:
        raise ivy.utils.exceptions.IvyException(
            "{} must be greater than {}".format(x1, x2) if message == "" else message
        )


def check_equal(x1, x2, inverse=False, message="", as_array=True):
    # not_equal
    eq_fn = lambda x1, x2: (x1 == x2 if inverse else x1 != x2)
    comp_fn = lambda x1, x2: ivy.any(eq_fn(x1, x2))
    if not as_array:
        iter_comp_fn = lambda x1_, x2_: any(eq_fn(x1, x2) for x1, x2 in zip(x1_, x2_))
        comp_fn = lambda x1, x2: iter_comp_fn(*_broadcast_inputs(x1, x2))
    eq = comp_fn(x1, x2)
    if inverse and eq:
        raise ivy.utils.exceptions.IvyException(
            "{} must not be equal to {}".format(x1, x2) if message == "" else message
        )
    # equal
    elif not inverse and eq:
        raise ivy.utils.exceptions.IvyException(
            "{} must be equal to {}".format(x1, x2) if message == "" else message
        )


def check_isinstance(x, allowed_types, message=""):
    if not isinstance(x, allowed_types):
        raise ivy.utils.exceptions.IvyException(
            "type of x: {} must be one of the allowed types: {}".format(
                type(x), allowed_types
            )
            if message == ""
            else message
        )


def check_exists(x, inverse=False, message=""):
    # not_exists
    if inverse and ivy.exists(x):
        raise ivy.utils.exceptions.IvyException(
            "arg must be None" if message == "" else message
        )
    # exists
    elif not inverse and not ivy.exists(x):
        raise ivy.utils.exceptions.IvyException(
            "arg must not be None" if message == "" else message
        )


def check_elem_in_list(elem, list, inverse=False, message=""):
    if inverse and elem in list:
        raise ivy.utils.exceptions.IvyException(
            message if message != "" else "{} must not be one of {}".format(elem, list)
        )
    elif not inverse and elem not in list:
        raise ivy.utils.exceptions.IvyException(
            message if message != "" else "{} must be one of {}".format(elem, list)
        )


def check_true(expression, message="expression must be True"):
    if not expression:
        raise ivy.utils.exceptions.IvyException(message)


def check_false(expression, message="expression must be False"):
    if expression:
        raise ivy.utils.exceptions.IvyException(message)


def check_all(results, message="one of the args is False", as_array=True):
    if (as_array and not ivy.all(results)) or (not as_array and not all(results)):
        raise ivy.utils.exceptions.IvyException(message)


def check_any(results, message="all of the args are False", as_array=True):
    if (as_array and not ivy.any(results)) or (not as_array and not any(results)):
        raise ivy.utils.exceptions.IvyException(message)


def check_all_or_any_fn(
    *args,
    fn,
    type="all",
    limit=[0],
    message="args must exist according to type and limit given",
    as_array=True,
):
    if type == "all":
        check_all([fn(arg) for arg in args], message, as_array=as_array)
    elif type == "any":
        count = 0
        for arg in args:
            count = count + 1 if fn(arg) else count
        if count not in limit:
            raise ivy.utils.exceptions.IvyException(message)
    else:
        raise ivy.utils.exceptions.IvyException("type must be all or any")


def check_shape(x1, x2, message=""):
    message = (
        message
        if message != ""
        else "{} and {} must have the same shape ({} vs {})".format(
            x1, x2, ivy.shape(x1), ivy.shape(x2)
        )
    )
    if ivy.shape(x1)[:] != ivy.shape(x2)[:]:
        raise ivy.utils.exceptions.IvyException(message)


def check_same_dtype(x1, x2, message=""):
    if ivy.dtype(x1) != ivy.dtype(x2):
        message = (
            message
            if message != ""
            else "{} and {} must have the same dtype ({} vs {})".format(
                x1, x2, ivy.dtype(x1), ivy.dtype(x2)
            )
        )
        raise ivy.utils.exceptions.IvyException(message)


# Creation #
# -------- #


def check_fill_value_and_dtype_are_compatible(fill_value, dtype):
    if (
        not (
            (ivy.is_int_dtype(dtype) or ivy.is_uint_dtype(dtype))
            and (isinstance(fill_value, int) or ivy.isinf(fill_value))
        )
        and not (
            ivy.is_complex_dtype(dtype) and isinstance(fill_value, (float, complex))
        )
        and not (
            ivy.is_float_dtype(dtype)
            and isinstance(fill_value, (float, np.float32))
            or isinstance(fill_value, bool)
        )
    ):
        raise ivy.utils.exceptions.IvyException(
            "the fill_value: {} and data type: {} are not compatible".format(
                fill_value, dtype
            )
        )


def check_unsorted_segment_min_valid_params(data, segment_ids, num_segments):
    if not (isinstance(num_segments, int)):
        raise ValueError("num_segments must be of integer type")

    valid_dtypes = [
        ivy.int32,
        ivy.int64,
    ]

    if ivy.backend == "torch":
        import torch

        valid_dtypes = [
            torch.int32,
            torch.int64,
        ]
        if isinstance(num_segments, torch.Tensor):
            num_segments = num_segments.item()
    elif ivy.backend == "paddle":
        import paddle

        valid_dtypes = [
            paddle.int32,
            paddle.int64,
        ]
        if isinstance(num_segments, paddle.Tensor):
            num_segments = num_segments.item()

    if segment_ids.dtype not in valid_dtypes:
        raise ValueError("segment_ids must have an integer dtype")

    if data.shape[0] != segment_ids.shape[0]:
        raise ValueError("The length of segment_ids should be equal to data.shape[0].")

    if ivy.max(segment_ids) >= num_segments:
        error_message = (
            f"segment_ids[{ivy.argmax(segment_ids)}] = "
            f"{ivy.max(segment_ids)} is out of range [0, {num_segments})"
        )
        raise ValueError(error_message)
    if num_segments <= 0:
        raise ValueError("num_segments must be positive")


# General #
# ------- #


def check_gather_input_valid(params, indices, axis, batch_dims):
    if batch_dims > axis:
        raise ivy.utils.exceptions.IvyException(
            "batch_dims ({}) must be less than or equal to axis ({}).".format(
                batch_dims, axis
            )
        )
    if params.shape[0:batch_dims] != indices.shape[0:batch_dims]:
        raise ivy.utils.exceptions.IvyException(
            "batch dimensions must match in `params` and `indices`;"
            + " saw {} vs. {}".format(
                params.shape[0:batch_dims], indices.shape[0:batch_dims]
            )
        )


def check_gather_nd_input_valid(params, indices, batch_dims):
    if batch_dims >= len(params.shape):
        raise ivy.utils.exceptions.IvyException(
            "batch_dims = {} must be less than rank(`params`) = {}.".format(
                batch_dims, len(params.shape)
            )
        )
    if batch_dims >= len(indices.shape):
        raise ivy.utils.exceptions.IvyException(
            "batch_dims = {}  must be less than rank(`indices`) = {}.".format(
                batch_dims, len(indices.shape)
            )
        )
    if params.shape[0:batch_dims] != indices.shape[0:batch_dims]:
        raise ivy.utils.exceptions.IvyException(
            "batch dimensions must match in `params` and `indices`;"
            + " saw {} vs. {}".format(
                params.shape[0:batch_dims], indices.shape[0:batch_dims]
            )
        )
    if indices.shape[-1] > (len(params.shape[batch_dims:])):
        raise ivy.utils.exceptions.IvyException(
            "index innermost dimension length must be <= "
            + "rank(`params[batch_dims:]`); saw: {} vs. {} .".format(
                indices.shape[-1], len(params.shape[batch_dims:])
            )
        )


def check_one_way_broadcastable(x1, x2):
    if len(x1) > len(x2):
        return False
    for a, b in zip(x1[::-1], x2[::-1]):
        if a == 1 or a == b:
            pass
        else:
            return False
    return True


def check_inplace_sizes_valid(var, data):
    if not check_one_way_broadcastable(data.shape, var.shape):
        raise ivy.utils.exceptions.IvyException(
            "Could not output values of shape {} into array with shape {}.".format(
                var.shape, data.shape
            )
        )


def check_shapes_broadcastable(var, data):
    if not check_one_way_broadcastable(var, data):
        raise ivy.utils.exceptions.IvyBroadcastShapeError(
            "Could not broadcast shape {} to shape {}.".format(data, var)
        )


def check_dimensions(x):
    if len(x.shape) <= 1:
        raise ivy.utils.exceptions.IvyException(
            "input must have greater than one dimension; "
            + " {} has {} dimensions".format(x, len(x.shape))
        )


def check_kernel_padding_size(kernel_size, padding_size):
    for i in range(len(kernel_size)):
        if (
            padding_size[i][0] > kernel_size[i] // 2
            or padding_size[i][1] > kernel_size[i] // 2
        ):
            raise ValueError(
                "Padding size should be less than or equal to half of the kernel size. "
                "Got kernel_size: {} and padding_size: {}".format(
                    kernel_size, padding_size
                )
            )


def check_dev_correct_formatting(device):
    assert device[0:3] in ["gpu", "tpu", "cpu"]
    if device != "cpu":
        assert device[3] == ":"
        assert device[4:].isnumeric()


# Jax Specific #
# ------- #


def _check_jax_x64_flag(dtype):
    if (
        ivy.backend == "jax"
        and not ivy.functional.backends.jax.jax.config.jax_enable_x64
    ):
        ivy.utils.assertions.check_elem_in_list(
            dtype,
            ["float64", "int64", "uint64", "complex128"],
            inverse=True,
            message=(
                f"{dtype} output not supported while jax_enable_x64"
                " is set to False, please import jax and enable the flag using "
                "jax.config.update('jax_enable_x64', True)"
            ),
        )
