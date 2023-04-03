# flake8: noqa
import subprocess
import os
import sys
import importlib
import json

import jsonpickle
from distutils.dir_util import copy_tree


subprocess.run("python --version", shell=True)

# install ivy
subprocess.run("conda install -y git", shell=True)
subprocess.run(
    "pip install   git+https://github.com/RickSanchezStoic/ivy.git", shell=True
)

# subprocess.run('pip uninstall numpy',shell=True)
# list of versions required for torch, and so on (maybe passed as args)
torch_req = ["torch/1.4.0"]
tensorflow_req = ["tensorflow/2.2.0", "tensorflow/2.2.1"]
jax_req = ["jax/0.1.60"]
numpy_req = ["numpy/1.17.3", "numpy/1.17.4", "numpy/1.23.1"]

# we create a directory for each framework and install different versions in different directories as per requirements
def directory_generator(req, base="fw/"):
    for versions in req:
        pkg, ver = versions.split("/")
        path = base + pkg + "/" + ver
        if not os.path.exists(path):
            install_pkg(path, pkg + "==" + ver)


def install_pkg(path, pkg, base="fw/"):
    subprocess.run(
        f"pip install {pkg} --default-timeout=100 --target={path}", shell=True
    )


# to import a specific pkg along with version name, to be used by the test functions
def custom_import(pkg, base="fw/", globally_done=None):
    """
    Import a specific package with version.
    
    :param pkg: package name and version separated by "/"
    :param base: base directory for the package installation
    :param globally_done: the previously imported framework
    :return: the imported module
    """
    if globally_done:
        # i.e import numpy etc
        if pkg == globally_done:
            ret = importlib.import_module(pkg.split("/")[0])
            return ret
        sys.path.remove(os.path.abspath(base + globally_done))
        temp = sys.modules.copy()
        sys.modules.clear()
        sys.modules.update(global_temp_sys_module)
        sys.path.insert(1, os.path.abspath(base + pkg))
        ret = importlib.import_module(pkg.split("/")[0])
        sys.path.remove(os.path.abspath(base + pkg))
        sys.path.insert(1, os.path.abspath(base + globally_done))
        sys.modules.clear()
        sys.modules.update(temp)
        return ret

    temp = sys.modules.copy()
    sys.path.insert(1, os.path.abspath(base + pkg))
    os.listdir("fw/")
    ret = importlib.import_module(pkg.split("/")[0])
    sys.path.remove(os.path.abspath(base + pkg))
    sys.modules.clear()
    sys.modules.update(temp)

    return ret


global_temp_sys_module = {}


def allow_global_framework_imports(fw=["numpy/1.23.1/"]):
    """
    Allow importing frameworks globally.
    
    :param fw: list of frameworks
    """
    # since no framework installed right Now we quickly store a copy of the sys.modules
    
    global global_temp_sys_module
    global_temp_sys_module = sys.modules.copy()
    for framework in fw:
        sys.path.insert(1, os.path.abspath("fw/" + framework))
    print(sys.path)


# we install numpy requirements
directory_generator(numpy_req)
directory_generator(tensorflow_req)

allow_global_framework_imports(fw=["numpy/1.23.1/"])

tens_v1 = custom_import("tensorflow/2.2.0")
tens_v2 = custom_import("tensorflow/2.2.1")


import ivy

ivy.set_backend("numpy")
print(ivy.backend_version)


print(tens_v1.__version__)
print(tens_v2.__version__)
