import importlib


versions = {
    "torch": "2.0.1",
    "tensorflow": "2.13.0",
    "numpy": "1.25.1",
    "jax": "0.4.13",
    "scipy": "1.10.1",
    "paddle": "2.5.1",
}


def fn_name_from_version_specific_fn_name(name, version):
    """

    Parameters
    ----------
    name
        the version specific name of the function for which the version support is to be
        provided.
    version
        the version of the current framework for which the support is to be provided,
        the version is inferred by importing the framework in the case of frontend
        version support and defaults to the highest available version in case of import
        failure
    Returns
    -------
        the name of the original function which will then point to the version specific
        function

    """
    version = str(version)
    if version.find("+") != -1:
        version = tuple(map(int, version[: version.index("+")].split(".")))
        # version = int(version[: version.index("+")].replace(".", ""))
    else:
        version = tuple(map(int, version.split(".")))
        # version = int(version.replace(".", ""))
    if "_to_" in name:
        i = name.index("_v_")
        e = name.index("_to_")
        version_start = name[i + 3 : e]
        version_start = tuple(map(int, version_start.split("p")))
        version_end = name[e + 4 :]
        version_end = tuple(map(int, version_end.split("p")))
        if version_start <= version <= version_end:
            return name[0:i]
    elif "_and_above" in name:
        i = name.index("_v_")
        e = name.index("_and_")
        version_start = name[i + 3 : e]
        version_start = tuple(map(int, version_start.split("p")))
        if version >= version_start:
            return name[0:i]
    else:
        i = name.index("_v_")
        e = name.index("_and_")
        version_start = name[i + 3 : e]
        version_start = tuple(map(int, version_start.split("p")))
        if version <= version_start:
            return name[0:i]


def set_frontend_to_specific_version(frontend):
    """

    Parameters
    ----------
    frontend
        the frontend module for which we provide the version support

    Returns
    -------
        The function doesn't return anything and updates the frontend __dict__
        to make the original function name to point to the version specific one
    """
    f = str(frontend.__name__)
    f = f[f.index("frontends") + 10 :]
    str_f = str(f)
    try:
        f = importlib.import_module(f)
        f_version = f.__version__
    except (ImportError, AttributeError):
        f_version = versions[str_f]

    for i in list(frontend.__dict__):
        if "_v_" in i:
            orig_name = fn_name_from_version_specific_fn_name(i, f_version)
            if orig_name:
                frontend.__dict__[orig_name] = frontend.__dict__[i]
