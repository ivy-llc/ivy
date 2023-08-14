# global
import sys
import importlib

from ivy_tests.test_ivy.helpers.hypothesis_helpers.array_helpers import (
    array_helpers_dtype_info_helper,
)

# local
from .testing_helpers import (
    _get_supported_devices_dtypes_helper,
    _get_method_supported_devices_dtypes_helper,
)
from .function_testing import (
    test_function_backend_computation,
    test_function_ground_truth_computation,
    test_method_backend_computation,
    test_method_ground_truth_computation,
    test_gradient_backend_computation,
    test_gradient_ground_truth_computation,
)

framework_path = "/opt/fw/"


def backend_proc(input_queue, output_queue):
    # first argument is going to be the framework and its path
    framework = input_queue.get()
    path = framework_path + framework
    sys.path.insert(1, path)
    framework = framework.split("/")[0]
    framework = importlib.import_module(framework)
    # if jax, do more stuff
    if framework.__name__ == "jax":
        framework.config.update("jax_enable_x64", True)

    while True:
        # subsequent arguments will be passed
        data = input_queue.get()
        if data[0] == "supported dtypes":
            # stage 1, calculating and returning supported dtypes
            # of each backend
            pass

            _, fn_module, fn_name, b = data
            output_queue.put(
                (_get_supported_devices_dtypes_helper(b, fn_module, fn_name))
            )
        elif data[0] == "method supported dtypes":
            # again stage 1, calculating and returning supported dtypes
            _, method_name, class_module, class_name, backend_str = data
            # since class module is name, we will import it to make it a module
            class_module = importlib.import_module(class_module)
            organized_dtypes = _get_method_supported_devices_dtypes_helper(
                method_name, class_module, class_name, backend_str
            )
            output_queue.put(organized_dtypes)

        elif data[0] == "dtype_info_helper":
            _, backend, kind_dtype, dtype = data
            dtype_info = array_helpers_dtype_info_helper(backend, kind_dtype, dtype)
            output_queue.put(dtype_info)

        elif data[0] == "function_backend_computation":
            # it's the backend return computation
            _, fw, test_flags, all_as_kwargs_np, input_dtypes, on_device, fn_name = data
            (
                ret_from_target,
                ret_np_flat_from_target,
                ret_device,
                args_np,
                arg_np_arrays,
                arrays_args_indices,
                kwargs_np,
                arrays_kwargs_indices,
                kwarg_np_arrays,
                test_flags,
            ) = test_function_backend_computation(
                fw, test_flags, all_as_kwargs_np, input_dtypes, on_device, fn_name
            )
            output_queue.put(
                (
                    ret_from_target,
                    ret_np_flat_from_target,
                    ret_device,
                    args_np,
                    arg_np_arrays,
                    arrays_args_indices,
                    kwargs_np,
                    arrays_kwargs_indices,
                    kwarg_np_arrays,
                    test_flags,
                )
            )
        elif data[0] == "function_ground_truth_computation":
            # it's the ground_truth return computation
            (
                _,
                ground_truth_backend,
                on_device,
                args_np,
                arg_np_arrays,
                arrays_args_indices,
                kwargs_np,
                arrays_kwargs_indices,
                kwarg_np_arrays,
                input_dtypes,
                test_flags,
                fn_name,
            ) = data
            (
                ret_from_gt,
                ret_np_from_gt_flat,
                ret_from_gt_device,
                test_flags,
                fw_list,
            ) = test_function_ground_truth_computation(
                ground_truth_backend,
                on_device,
                args_np,
                arg_np_arrays,
                arrays_args_indices,
                kwargs_np,
                arrays_kwargs_indices,
                kwarg_np_arrays,
                input_dtypes,
                test_flags,
                fn_name,
            )
            output_queue.put(
                (
                    ret_from_gt,
                    ret_np_from_gt_flat,
                    ret_from_gt_device,
                    test_flags,
                    fw_list,
                )
            )
        elif data[0] == "gradient_backend_computation":
            # gradient testing , part where it uses the backend
            (
                _,
                backend_to_test,
                args_np,
                arg_np_vals,
                args_idxs,
                kwargs_np,
                kwarg_np_vals,
                kwargs_idxs,
                input_dtypes,
                test_flags,
                on_device,
                fn,
                test_compile,
                xs_grad_idxs,
                ret_grad_idxs,
            ) = data
            grads_np_flat = test_gradient_backend_computation(
                backend_to_test,
                args_np,
                arg_np_vals,
                args_idxs,
                kwargs_np,
                kwarg_np_vals,
                kwargs_idxs,
                input_dtypes,
                test_flags,
                on_device,
                fn,
                test_compile,
                xs_grad_idxs,
                ret_grad_idxs,
            )
            output_queue.put(grads_np_flat)

        elif data[0] == "gradient_ground_truth_computation":
            # gradient testing, part where it uses ground truth
            (
                _,
                ground_truth_backend,
                on_device,
                fn,
                input_dtypes,
                all_as_kwargs_np,
                args_np,
                arg_np_vals,
                args_idxs,
                kwargs_np,
                kwarg_np_vals,
                test_flags,
                kwargs_idxs,
                test_compile,
                xs_grad_idxs,
                ret_grad_idxs,
            ) = data
            grads_np_from_gt_flat = test_gradient_ground_truth_computation(
                ground_truth_backend,
                on_device,
                fn,
                input_dtypes,
                all_as_kwargs_np,
                args_np,
                arg_np_vals,
                args_idxs,
                kwargs_np,
                kwarg_np_vals,
                test_flags,
                kwargs_idxs,
                test_compile,
                xs_grad_idxs,
                ret_grad_idxs,
            )
            output_queue.put(grads_np_from_gt_flat)

        elif data[0] == "method_backend_computation":
            (
                _,
                init_input_dtypes,
                init_flags,
                backend_to_test,
                init_all_as_kwargs_np,
                on_device,
                method_input_dtypes,
                method_flags,
                method_all_as_kwargs_np,
                class_name,
                method_name,
                init_with_v,
                test_compile,
                method_with_v,
            ) = data
            (
                ret,
                ret_np_flat,
                ret_device,
                org_con_data,
                args_np_method,
                met_arg_np_vals,
                met_args_idxs,
                kwargs_np_method,
                met_kwarg_np_vals,
                met_kwargs_idxs,
                v_np,
                fw_list,
            ) = test_method_backend_computation(
                init_input_dtypes,
                init_flags,
                backend_to_test,
                init_all_as_kwargs_np,
                on_device,
                method_input_dtypes,
                method_flags,
                method_all_as_kwargs_np,
                class_name,
                method_name,
                init_with_v,
                test_compile,
                method_with_v,
            )
            output_queue.put(
                (
                    ret,
                    ret_np_flat,
                    ret_device,
                    org_con_data,
                    args_np_method,
                    met_arg_np_vals,
                    met_args_idxs,
                    kwargs_np_method,
                    met_kwarg_np_vals,
                    met_kwargs_idxs,
                    v_np,
                    fw_list,
                )
            )

        elif data[0] == "method_ground_truth_computation":
            (
                _,
                ground_truth_backend,
                on_device,
                org_con_data,
                args_np_method,
                met_arg_np_vals,
                met_args_idxs,
                kwargs_np_method,
                met_kwarg_np_vals,
                met_kwargs_idxs,
                method_input_dtypes,
                method_flags,
                class_name,
                method_name,
                test_compile,
                v_np,
            ) = data
            (
                ret_from_gt,
                ret_np_from_gt_flat,
                ret_from_gt_device,
                fw_list2,
            ) = test_method_ground_truth_computation(
                ground_truth_backend,
                on_device,
                org_con_data,
                args_np_method,
                met_arg_np_vals,
                met_args_idxs,
                kwargs_np_method,
                met_kwarg_np_vals,
                met_kwargs_idxs,
                method_input_dtypes,
                method_flags,
                class_name,
                method_name,
                test_compile,
                v_np,
            )
            output_queue.put(
                (ret_from_gt, ret_np_from_gt_flat, ret_from_gt_device, fw_list2)
            )

        if not data:
            break
        # process the data


# TODO incomplete
def frontend_proc(input_queue, output_queue):
    # first argument is going to be the framework and its path
    framework = input_queue.get()
    sys.path.insert(1, f"{framework_path}{framework}")
    importlib.import_module(framework.split("/")[0])
    while True:
        # subsequent arguments will be passed
        data = input_queue.get()
        if not data:
            break
        # process the data
