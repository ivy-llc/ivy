# global
import sys
import importlib

# local
from .testing_helpers  import _get_supported_devices_dtypes_helper
from .function_testing import test_function_backend_computation


def backend_proc(input_queue, output_queue):
    # first argument is going to be the framework and its path

    framework = input_queue.get()
    path = "/opt/fw/" + framework
    sys.path.insert(1, path)
    framework = framework.split("/")[0]
    framework = importlib.import_module(framework)
    print(framework.__version__)
    while True:
        # subsequent arguments will be passed
        data = input_queue.get()
        if data[0]=='supported dtypes':
            # stage 1, calculating and returning supported dtypes
            # of each backend
            import ivy
            _, fn_module, fn_name, b = data
            output_queue.put((_get_supported_devices_dtypes_helper(b,fn_module,fn_name)))
        elif data[0]=='function_backend_computation':
            _,test_flags,all_as_kwargs_np,input_dtypes, on_device, fn_name =data
            ret_from_target,ret_np_flat_from_target,ret_device,args_np,arg_np_arrays,arrays_args_indices,kwargs_np,arrays_kwargs_indices,kwarg_np_arrays, test_flags =test_function_backend_computation(test_flags,all_as_kwargs_np,input_dtypes, on_device, fn_name)
            output_queue.put((ret_from_target,ret_np_flat_from_target,ret_device,args_np,arg_np_arrays,arrays_args_indices,kwargs_np,arrays_kwargs_indices,kwarg_np_arrays, test_flags))



        if not data:
            break
        # process the data


def frontend_proc(input_queue, output_queue):
    # first argument is going to be the framework and its path
    framework = queue.get()
    sys.path.insert(1, f"/opt/fw/{framework}")
    importlib.import_module(framework.split("/")[0])
    while True:
        # subsequent arguments will be passed
        data = queue.get()
        if not data:
            break
        # process the data