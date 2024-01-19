import os
import sys
import glob
import importlib

dir_path = os.path.dirname(os.path.realpath(__file__))
so_files = glob.glob(dir_path + "/*.so")
sys.path.append(dir_path)

__all__ = []

for so_file in so_files:
    # if os.path.basename(so_file) != "add.so":
    #     continue
    module_name = os.path.splitext(os.path.basename(so_file))[0]

    locals()[module_name] = importlib.import_module(module_name)

    if module_name + "_wrapper" in locals()[module_name].__dict__.keys():
        locals()[module_name + "_wrapper"] = getattr(
            locals()[module_name], module_name + "_wrapper"
        )
        __all__.append(module_name + "_wrapper")

del dir_path
del so_files

import utils
from utils import *
