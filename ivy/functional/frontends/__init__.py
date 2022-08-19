# flake8: noqa
from . import numpy
from . import jax
from . import torch
from . import tensorflow
import importlib


def version_extractor(name, version):
    version = str(version)
    if version.find("+") != -1:
        version = int(version[: version.index("+")].replace(".", ""))
    else:
        version = int(version.replace(".", ""))
    if "_to_" in name:
        i = name.index("_v_")
        e = name.index("_to_")
        version_start = name[i + 3 : e]
        version_start = int(version_start.replace("p", ""))
        version_end = name[e + 4 :]
        version_end = int(version_end.replace("p", ""))
        if version in range(version_start, version_end + 1):
            return name[0:i]
    elif "_and_above" in name:
        i = name.index("_v_")
        e = name.index("_and_")
        version_start = name[i + 3 : e]
        version_start = int(version_start.replace("p", ""))
        if version >= version_start:
            return name[0:i]
    else:
        i = name.index("_v_")
        e = name.index("_and_")
        version_start = name[i + 3 : e]
        version_start = int(version_start.replace("p", ""))
        if version <= version_start:
            return name[0:i]


def version_handler(frontend):
    f = str(frontend.__name__)
    f = f[f.index("frontends") + 10 :]
    f = importlib.import_module(f)
    f_version = f.__version__
    for i in list(frontend.__dict__):
        if hasattr(frontend.__dict__[i], "version"):
            orig_name = version_extractor(i, f_version)
            if orig_name:
                frontend.__dict__[orig_name] = frontend.__dict__[i]


version_handler(torch)
version_handler(tensorflow)
version_handler(jax)
