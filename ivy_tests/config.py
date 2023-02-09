# flake8: noqa
import os
import sys
import importlib

global_temp_sys_module = {}


def allow_global_framework_imports(fw=["numpy/1.23.1"]):
    # since no framework installed right now we quickly store a copy of the sys.modules
    global global_temp_sys_module
    if not global_temp_sys_module:
        global_temp_sys_module = sys.modules.copy()
    for framework in fw:
        sys.path.insert(1, os.path.abspath("/opt/miniconda/fw/" + framework))


def try_except():
    try:
        import numpy
    except ImportError:
        allow_global_framework_imports()


def return_global_temp_sys_module():
    return global_temp_sys_module


def reset_sys_modules_to_base():
    if global_temp_sys_module != sys.modules:
        sys.modules.clear()
        sys.modules.update(global_temp_sys_module)


# to import a specific pkg along with version name, to be used by the test functions
def custom_import(
    pkg, module_name, base="/opt/miniconda/fw/", globally_done=None
):  # format is pkg_name/version , globally_done means
    # if we have imported any framework before globally
    if globally_done:  # i.e import numpy etc
        if pkg == globally_done:
            ret = importlib.import_module(module_name)
            return ret
        sys.path.remove(os.path.abspath(base + globally_done))
        temp = sys.modules.copy()
        sys.modules.clear()
        sys.modules.update(global_temp_sys_module)
        sys.path.insert(1, os.path.abspath(base + pkg))
        ret = importlib.import_module(module_name)
        sys.path.remove(base + pkg)
        sys.path.insert(1, base + globally_done)
        sys.modules.clear()
        sys.modules.update(temp)
        return ret

    temp = sys.modules.copy()
    sys.path.insert(1, os.path.abspath(base + pkg))
    ret = importlib.import_module(module_name)
    sys.path.remove(os.path.abspath(base + pkg))
    sys.modules.clear()
    sys.modules.update(temp)

    return ret
