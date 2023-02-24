from ivy_tests import config
from ivy_tests.test_ivy.helpers.structs import FrontendMethodData
import sys
import jsonpickle
import importlib
from ivy_tests.test_ivy.helpers.testing_helpers import (
    _import_fn,
    _get_supported_devices_dtypes,
)


def available_frameworks():
    available_frameworks_lis = ["numpy", "jax", "tensorflow", "torch"]
    try:
        import jax

        assert jax, "jax is imported to see if the user has it installed"
    except ImportError:
        available_frameworks_lis.remove("jax")

    try:
        import tensorflow as tf

        assert tf, "tensorflow is imported to see if the user has it installed"
    except ImportError:
        available_frameworks_lis.remove("tensorflow")

    try:
        import torch

        assert torch, "torch is imported to see if the user has it installed"
    except ImportError:
        available_frameworks_lis.remove("torch")
    return available_frameworks_lis


def convtrue(argument):
    """Convert NativeClass in argument to true framework counter part"""
    if isinstance(argument, NativeClass):
        return argument._native_class
    return argument


class NativeClass:
    """
    An empty class to represent a class that only exist in a specific framework.

    Attributes
    ----------
    _native_class : class reference
        A reference to the framework-specific class.
    """

    def __init__(self, native_class):
        """
        Constructs the native class object.

        Parameters
        ----------
        native_class : class reference
            A reperence to the framework-specific class being represented.
        """
        self._native_class = native_class


def _get_fn_dtypes(framework, fn_tree, device=None, kind="valid"):
    callable_fn, fn_name, fn_mod = _import_fn(fn_tree)
    supported_device_dtypes = _get_supported_devices_dtypes(fn_name, fn_mod)
    return supported_device_dtypes[framework][device][kind]


def _get_type_dict(framework, fn_tree, device=None, kind="valid"):
    if kind == "valid":
        return framework.valid_dtypes
    elif kind == "numeric":
        return framework.valid_numeric_dtypes
    elif kind == "integer":
        return framework.valid_int_dtypes
    elif kind == "float":
        return framework.valid_float_dtypes
    elif kind == "unsigned":
        return framework.valid_int_dtypes
    elif kind == "signed_integer":
        return tuple(
            set(framework.valid_int_dtypes).difference(framework.valid_uint_dtypes)
        )
    elif kind == "complex":
        return framework.valid_complex_dtypes
    elif kind == "real_and_complex":
        return tuple(
            set(framework.valid_numeric_dtypes).union(framework.valid_complex_dtypes)
        )
    elif kind == "float_and_complex":
        return tuple(
            set(framework.valid_float_dtypes).union(framework.valid_complex_dtypes)
        )
    elif kind == "bool":
        return tuple(
            set(framework.valid_dtypes).difference(framework.valid_numeric_dtypes)
        )
    else:
        raise RuntimeError("{} is an unknown kind!".format(kind))


def dtype_handler(framework):
    z = input()
    retrieval_fn = globals()[z]
    z = input()
    kind = z
    z = input()
    device = z
    z = input()
    fn_tree = z

    if retrieval_fn.__name__ == "_get_type_dict":
        framework = importlib.import_module("ivy.functional.backends." + framework)
    dtypes = retrieval_fn(framework, fn_tree, device, kind)
    dtypes = jsonpickle.dumps(dtypes)
    print(dtypes)


def test_frontend_method():
    z = input()
    pickle_dict = jsonpickle.loads(z)
    z = pickle_dict
    (
        args_constructor_np,
        kwargs_constructor_np,
        args_method_np,
        kwargs_method_np,
        frontend_method_data,
    ) = (z["a"], z["b"], z["c"], z["d"], z["e"])

    frontend_method_data = FrontendMethodData(
        ivy_init_module=frontend_method_data.ivy_init_module,
        framework_init_module=importlib.import_module(
            frontend_method_data.framework_init_module
        ),
        init_name=frontend_method_data.init_name,
        method_name=frontend_method_data.method_name,
    )
    args_constructor_frontend = ivy.nested_map(
        args_constructor_np,
        lambda x: ivy.native_array(x) if isinstance(x, numpy.ndarray) else x,
        shallow=False,
    )
    kwargs_constructor_frontend = ivy.nested_map(
        kwargs_constructor_np,
        lambda x: ivy.native_array(x) if isinstance(x, numpy.ndarray) else x,
        shallow=False,
    )
    args_method_frontend = ivy.nested_map(
        args_method_np,
        lambda x: ivy.native_array(x)
        if isinstance(x, numpy.ndarray)
        else ivy.as_native_dtype(x)
        if isinstance(x, ivy.Dtype)
        else ivy.as_native_dev(x)
        if isinstance(x, ivy.Device)
        else x,
        shallow=False,
    )
    kwargs_method_frontend = ivy.nested_map(
        kwargs_method_np,
        lambda x: ivy.native_array(x) if isinstance(x, numpy.ndarray) else x,
        shallow=False,
    )

    # change ivy dtypes to native dtypes
    if "dtype" in kwargs_method_frontend:
        kwargs_method_frontend["dtype"] = ivy.as_native_dtype(
            kwargs_method_frontend["dtype"]
        )

    # change ivy device to native devices
    if "device" in kwargs_method_frontend:
        kwargs_method_frontend["device"] = ivy.as_native_dev(
            kwargs_method_frontend["device"]
        )
    frontend_creation_fn = getattr(
        frontend_method_data.framework_init_module, frontend_method_data.init_name
    )
    ins_gt = frontend_creation_fn(
        *args_constructor_frontend, **kwargs_constructor_frontend
    )
    frontend_ret = ins_gt.__getattribute__(frontend_method_data.method_name)(
        *args_method_frontend, **kwargs_method_frontend
    )
    try:
        tensorflow = importlib.import_module("tensorflow")
    except:
        tensorflow = None
    if ivy.current_backend_str() == "tensorflow" and isinstance(
        frontend_ret, getattr(tensorflow, "TensorShape", None)
    ):
        frontend_ret_np_flat = [numpy.asarray(frontend_ret, dtype=numpy.int32)]
        ret = jsonpickle.dumps({"a": 0, "b": frontend_ret_np_flat})
        print(ret)
    elif ivy.isscalar(frontend_ret):
        frontend_ret_np_flat = [numpy.asarray(frontend_ret)]
        ret = jsonpickle.dumps({"a": 0, "b": frontend_ret_np_flat})
        print(ret)
    else:
        ret = jsonpickle.dumps({"a": 1, "b": ivy.to_numpy(frontend_ret)})
        print(ret)


if __name__ == "__main__":

    arg_lis = sys.argv
    fw_lis = []
    for i in arg_lis[1:]:
        if i.split("/")[0] == "jax":
            fw_lis.append(i.split("/")[0] + "/" + i.split("/")[1])
            fw_lis.append(i.split("/")[2] + "/" + i.split("/")[3])
        else:
            fw_lis.append(i)
    config.allow_global_framework_imports(fw=fw_lis)

    j = 1
    import ivy

    # ivy.bfloat16
    ivy.set_backend(arg_lis[2].split("/")[0])
    import numpy

    while j:
        try:
            z = input()
            if z == "1":
                dtype_handler(arg_lis[2].split("/")[0])
                continue
            if z == "2":
                test_frontend_method()
                continue
            pickle_dict = jsonpickle.loads(z)
            frontend_fw = input()

            frontend_fw = importlib.import_module(frontend_fw)

            func = input()

            args_np, kwargs_np = pickle_dict["a"], pickle_dict["b"]
            args_frontend = ivy.nested_map(
                args_np,
                lambda x: ivy.native_array(x)
                if isinstance(x, numpy.ndarray)
                else ivy.as_native_dtype(x)
                if isinstance(x, ivy.Dtype)
                else x,
                shallow=False,
            )
            kwargs_frontend = ivy.nested_map(
                kwargs_np,
                lambda x: ivy.native_array(x) if isinstance(x, numpy.ndarray) else x,
                shallow=False,
            )

            # change ivy dtypes to native dtypes
            if "dtype" in kwargs_frontend:
                kwargs_frontend["dtype"] = ivy.as_native_dtype(kwargs_frontend["dtype"])

            # change ivy device to native devices
            if "device" in kwargs_frontend:
                kwargs_frontend["device"] = ivy.as_native_dev(kwargs_frontend["device"])

            # check and replace the NativeClass objects in arguments
            # with true counterparts
            args_frontend = ivy.nested_map(
                args_frontend, fn=convtrue, include_derived=True, max_depth=10
            )
            kwargs_frontend = ivy.nested_map(
                kwargs_frontend, fn=convtrue, include_derived=True, max_depth=10
            )

            frontend_ret = frontend_fw.__dict__[func](*args_frontend, **kwargs_frontend)
            if isinstance(frontend_ret, tuple) or isinstance(frontend_ret, list):
                frontend_ret = ivy.nested_map(frontend_ret, ivy.to_numpy)
            else:
                frontend_ret = ivy.to_numpy(frontend_ret)
            frontend_ret = jsonpickle.dumps(frontend_ret)
            print(frontend_ret)
        except EOFError:
            continue
        except Exception as e:
            print(frontend_ret.shape)
            raise e
