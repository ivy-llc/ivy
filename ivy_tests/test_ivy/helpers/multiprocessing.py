# global
import sys
import importlib

# local
from .testing_helpers import (
    _get_supported_devices_dtypes_helper,
    _get_method_supported_devices_dtypes_helper,
)
from .function_testing import (
    test_function_backend_computation,
    test_function_ground_truth_computation,
    gradient_test,
    test_method_backend_computation,
    test_method_ground_truth_computation,
)


def backend_proc(input_queue, output_queue):
    # first argument is going to be the framework and its path
    framework = input_queue.get()
    path = "/opt/fw/" + framework
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
            import ivy

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
        elif data[0] == "function_gradient_computation":
            # it's gradient testing
            import numpy as np

            (
                _,
                fn_name,
                all_as_kwargs_np,
                args_np,
                kwargs_np,
                input_dtypes,
                test_flags,
                rtol_,
                atol_,
                xs_grad_idxs,
                ret_grad_idxs,
                ground_truth_backend,
                on_device,
            ) = data
            if (
                test_flags.test_gradients
                and not test_flags.instance_method
                and "bool" not in input_dtypes
                and not any(ivy.is_complex_dtype(d) for d in input_dtypes)
            ):
                if fw not in fw_list or not ivy.nested_argwhere(
                    all_as_kwargs_np,
                    lambda x: (
                        x.dtype in fw_list[fw] if isinstance(x, np.ndarray) else None
                    ),
                ):
                    gradient_test(
                        fn=fn_name,
                        all_as_kwargs_np=all_as_kwargs_np,
                        args_np=args_np,
                        kwargs_np=kwargs_np,
                        input_dtypes=input_dtypes,
                        test_flags=test_flags,
                        rtol_=rtol_,
                        atol_=atol_,
                        xs_grad_idxs=xs_grad_idxs,
                        ret_grad_idxs=ret_grad_idxs,
                        ground_truth_backend=ground_truth_backend,
                        on_device=on_device,
                    )
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


def frontend_proc(input_queue, output_queue):
    # first argument is going to be the framework and its path
    framework = input_queue.get()
    sys.path.insert(1, f"/opt/fw/{framework}")
    importlib.import_module(framework.split("/")[0])
    while True:
        # subsequent arguments will be passed
        data = input_queue.get()
        if not data:
            break
        # process the data
