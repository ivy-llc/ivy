import sys
import inspect
import importlib
from unittest import mock
from dataclasses import dataclass
from hypothesis import find
import numpy as np  # base framework for testing

import ivy
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.helpers.globals as test_globals


BACKENDS = ["jax", "numpy", "paddle", "tensorflow", "torch"]
DEVICES = ["cpu", "gpu:0", "tpu:0"]

# TODO: get these in a smarter way
FN_NAMES = [
    "gelu",
    "hardswish",
    "leaky_relu",
    "log_softmax",
    "mish",
    "relu",
    "sigmoid",
    "softmax",
    "softplus",
]


ivy.set_inplace_mode("strict")


def is_dtype_err_jax(e):
    return "does not accept dtype" in str(e)


def is_dtype_err_np(e):
    return "not supported for the input types" in str(e)


def is_dtype_err_paddle(e):
    return "Selected wrong DataType" in str(e)


def is_dtype_err_tf(e: ivy.exceptions.IvyException):
    return ("Value for attr 'T' of" in str(e)) or ("`features.dtype` must be" in str(e))


def is_dtype_err_torch(e):
    return "not implemented for" in str(e)


is_dtype_err = {
    "jax": is_dtype_err_jax,
    "numpy": is_dtype_err_np,
    "paddle": is_dtype_err_paddle,
    "tensorflow": is_dtype_err_tf,
    "torch": is_dtype_err_torch,
}


results = {}


@dataclass
class TestCase:
    backend: str = "numpy"
    fn_name: str = "add"
    device: ivy.Device = "cpu"
    dtype: ivy.Dtype = "float32"


test_case = TestCase()


def _set_result(res, err):
    global results, test_case
    if err is None:
        results[test_case.backend][test_case.fn_name][test_case.device][res].add(
            test_case.dtype
        )
    else:
        results[test_case.backend][test_case.fn_name][test_case.device][res].add(
            (test_case.dtype, err)
        )


def _get_nested_np_arrays(nest):
    indices = ivy.nested_argwhere(nest, lambda x: isinstance(x, np.ndarray))

    ret = ivy.multi_index_nest(nest, indices)
    return ret, indices, len(ret)


def mock_test_function(
    *,
    input_dtypes,
    test_flags,
    fn_name,
    rtol_=None,
    atol_=1e-06,
    tolerance_dict=None,
    test_values=True,
    xs_grad_idxs=None,
    ret_grad_idxs=None,
    backend_to_test,
    on_device,
    return_flat_np_arrays=False,
    **all_as_kwargs_np,
):
    # split the arguments into their positional and keyword components
    args_np, kwargs_np = helpers.kwargs_to_args_n_kwargs(
        num_positional_args=test_flags.num_positional_args, kwargs=all_as_kwargs_np
    )

    # Extract all arrays from the arguments and keyword arguments
    arg_np_arrays, arrays_args_indices, n_args_arrays = _get_nested_np_arrays(args_np)
    kwarg_np_arrays, arrays_kwargs_indices, n_kwargs_arrays = _get_nested_np_arrays(
        kwargs_np
    )

    # Make all array-specific test flags and dtypes equal in length
    total_num_arrays = n_args_arrays + n_kwargs_arrays
    if len(input_dtypes) < total_num_arrays:
        input_dtypes = [input_dtypes[0] for _ in range(total_num_arrays)]
    if len(test_flags.as_variable) < total_num_arrays:
        test_flags.as_variable = [
            test_flags.as_variable[0] for _ in range(total_num_arrays)
        ]
    if len(test_flags.native_arrays) < total_num_arrays:
        test_flags.native_arrays = [
            test_flags.native_arrays[0] for _ in range(total_num_arrays)
        ]
    if len(test_flags.container) < total_num_arrays:
        test_flags.container = [
            test_flags.container[0] for _ in range(total_num_arrays)
        ]

    with helpers.BackendHandler.update_backend(backend_to_test) as ivy_backend:
        # Update variable flags to be compatible with float dtype and with_out args
        test_flags.as_variable = [
            v if ivy_backend.is_float_dtype(d) and not test_flags.with_out else False
            for v, d in zip(test_flags.as_variable, input_dtypes)
        ]

    # update instance_method flag to only be considered if the
    # first term is either an ivy.Array or ivy.Container
    instance_method = test_flags.instance_method and (
        not test_flags.native_arrays[0] or test_flags.container[0]
    )

    args, kwargs = helpers.create_args_kwargs(
        backend=backend_to_test,
        args_np=args_np,
        arg_np_vals=arg_np_arrays,
        args_idxs=arrays_args_indices,
        kwargs_np=kwargs_np,
        kwarg_np_vals=kwarg_np_arrays,
        kwargs_idxs=arrays_kwargs_indices,
        input_dtypes=input_dtypes,
        test_flags=test_flags,
        on_device=on_device,
    )

    with helpers.BackendHandler.update_backend(backend_to_test) as ivy_backend:
        # If function doesn't have an out argument but an out argument is given
        # or a test with out flag is True
        if ("out" in kwargs or test_flags.with_out) and "out" not in inspect.signature(
            getattr(ivy_backend, fn_name)
        ).parameters:
            raise Exception(f"Function {fn_name} does not have an out parameter")

        # Run either as an instance method or from the API directly
        instance = None
        if instance_method:
            array_or_container_mask = [
                (not native_flag) or container_flag
                for native_flag, container_flag in zip(
                    test_flags.native_arrays, test_flags.container
                )
            ]

            # Boolean mask for args and kwargs True if an entry's
            # test Array flag is True or test Container flag is true
            args_instance_mask = array_or_container_mask[
                : test_flags.num_positional_args
            ]
            kwargs_instance_mask = array_or_container_mask[
                test_flags.num_positional_args :
            ]

            if any(args_instance_mask):
                instance, args = helpers._find_instance_in_args(
                    backend_to_test, args, arrays_args_indices, args_instance_mask
                )
            else:
                instance, kwargs = helpers._find_instance_in_args(
                    backend_to_test, kwargs, arrays_kwargs_indices, kwargs_instance_mask
                )

            if test_flags.test_compile:
                target_fn = lambda instance, *args, **kwargs: instance.__getattribute__(
                    fn_name
                )(*args, **kwargs)
                args = [instance, *args]
            else:
                target_fn = instance.__getattribute__(fn_name)
        else:
            target_fn = ivy_backend.__dict__[fn_name]

        helpers.get_ret_and_flattened_np_array(
            backend_to_test,
            target_fn,
            *args,
            test_compile=test_flags.test_compile,
            precision_mode=test_flags.precision_mode,
            **kwargs,
        )


def mock_get_dtypes(*args, **kwargs):
    global test_case
    return [test_case.dtype]


helpers.test_function = mock.Mock(wraps=mock_test_function)
helpers.get_dtypes = mock.Mock(wraps=mock_get_dtypes)
sys.modules["ivy_tests.test_ivy.helpers"] = helpers

test_globals._set_backend("numpy")

for backend in BACKENDS:
    test_case.backend = backend
    if backend not in results:
        results[backend] = {}

    ivy.set_backend(test_case.backend)
    import ivy_tests.test_ivy.test_functional.test_nn.test_activations as test_file

    for dtype in ivy.valid_dtypes:
        if dtype == "bfloat16":
            continue
        test_case.dtype = dtype

        importlib.reload(test_file)

        for fn_name in FN_NAMES:
            test_case.fn_name = fn_name
            if fn_name not in results[test_case.backend]:
                results[test_case.backend][fn_name] = {}

            for device in DEVICES:
                if "gpu" in device and not ivy.gpu_is_available():
                    continue
                if "tpu" in device and not ivy.tpu_is_available():
                    continue
                test_case.device = device
                if device not in results[test_case.backend][test_case.fn_name]:
                    results[test_case.backend][test_case.fn_name][device] = {
                        "supported": set(),
                        "unsupported": set(),
                        "unsure": set(),
                    }

                kwargs = (
                    test_file.__dict__[f"test_{test_case.fn_name}"]
                    .__dict__["hypothesis"]
                    ._given_kwargs
                )
                min_example = {}
                for k, v in kwargs.items():
                    min_example[k] = find(specifier=v, condition=lambda _: True)

                try:
                    test_file.__dict__[f"test_{test_case.fn_name}"].original(
                        **min_example,
                        backend_fw=test_case.backend,
                        on_device=test_case.device,
                    )
                    print("Valid dtype:", test_case.dtype)
                    _set_result("supported", None)
                except Exception as e:
                    if is_dtype_err[test_case.backend](e):
                        print("Invalid Dtype:", test_case.dtype)
                        _set_result("unsupported", e)
                    else:
                        print(
                            f"Potentially valid dtype: {test_case.dtype}"
                            f" (error message: {str(e)})"
                        )
                        _set_result("unsure", e)

test_globals._unset_backend()

for b in results.keys():
    for f in results[b].keys():
        for d in results[b][f].keys():
            print(b, f, d)
            print("Supported:", results[b][f][d]["supported"])
            print("Unsupported:", results[b][f][d]["unsupported"])
            print("Unsure:", results[b][f][d]["unsure"])
            print("\n\n")
