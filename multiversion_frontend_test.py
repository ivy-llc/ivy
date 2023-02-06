from ivy_tests import config
import sys
import jsonpickle
import importlib
import codecs
import io
import os
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





if __name__=='__main__':

    arg_lis=sys.argv
    fw_lis=[]
    jaxlib_flag=0
    for i in arg_lis[1:]:
        if i.split('/')[0]=='jax':
            fw_lis.append(i.split('/')[0]+'/'+i.split('/')[1])
            fw_lis.append(i.split('/')[2]+'/'+i.split('/')[3])
            jaxlib_flag=1
        else:
            fw_lis.append(i)
    config.allow_global_framework_imports(fw=fw_lis)
    if jaxlib_flag:
        import jaxlib
    j=1
    import ivy
    # ivy.bfloat16
    ivy.set_backend(arg_lis[2].split('/')[0])
    import numpy
    while j:
        try:
            z=input()
            pickle_dict=jsonpickle.loads(z)
            frontend_fw=input()

            frontend_fw = importlib.import_module(frontend_fw)

            func=input()

            args_np,kwargs_np=pickle_dict['a'],pickle_dict['b']
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
                kwargs_frontend["dtype"] = ivy.as_native_dtype(
                    kwargs_frontend["dtype"]
                )

            # change ivy device to native devices
            if "device" in kwargs_frontend:
                kwargs_frontend["device"] = ivy.as_native_dev(
                    kwargs_frontend["device"]
                )

            # check and replace the NativeClass objects in arguments
            # with true counterparts
            args_frontend = ivy.nested_map(
                args_frontend, fn=convtrue, include_derived=True, max_depth=10
            )
            kwargs_frontend = ivy.nested_map(
                kwargs_frontend, fn=convtrue, include_derived=True, max_depth=10
            )

            frontend_ret = frontend_fw.__dict__[func](
                *args_frontend , **kwargs_frontend
            )
            frontend_ret=ivy.to_ivy(frontend_ret)
            frontend_ret=jsonpickle.dumps(frontend_ret)
            print(frontend_ret)
        except EOFError:
            continue
        except Exception as e:
            raise e
