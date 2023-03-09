import sys
import subprocess
import pprint
from colorama import Fore, Style, init
from importlib import import_module
from importlib.util import find_spec


_backend_generation_path = "/ivy/functional/backends/"
_imported_backend = None
_backend_is_installed = False

backend = {"name": None, "alias": None}
config = {
    "NativeArray": None,
    "NativeVariable": None,
    "NativeDevice": None,
    "NativeDtype": None,
    "NativeShape": None,
    "NativeSparseArray": None,
}
config_lists = {
    "valid_devices": ["cpu"],
}
config_flags = {
    "native_inplace_support": False,
    "supports_gradients": False,
}


def _get_user_input(fn, *args, **kwargs):
    while True:
        try:
            ret = fn(*args, **kwargs)
            if ret:
                break
        except KeyboardInterrupt:
            print("Aborted.")
            exit()


def _query_input(key):
    while True:
        try:
            val = input(
                f"Enter a value for {Style.BRIGHT + key + Style.NORMAL},"
                "default: '{Style.BRIGHT}{config[key]}{Style.NORMAL}'. "
                "Press ENTER to skip: "
            )
            if val.strip(" ") == "":
                return
        except KeyboardInterrupt:
            print("Aborted.")
            exit()


def _should_install_backend(package_name):
    ret = input(
        f"Backend {package_name} isn't installed locally, "
        "would you like to install it? [y/N]\n"
    )
    if ret.lower() == "y":
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package_name]
            )
            global _backend_is_installed
            _backend_is_installed = True
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                Fore.RED + f"Installing {package_name} failed. {e}"
            ) from e
    elif ret.lower() == "n":
        print(
            Fore.YELLOW + "Will continue without backend installed, "
            "type checking won't be available.\n"
        )
    else:
        print(Fore.RED + f"{ret} not understood.")
        return False
    return True


def _get_backend():
    package_name = input(
        "Enter backend name (same as Python package name, case sensitive): "
    )
    package_name = package_name.strip(" ")
    if package_name.strip(" ") == "":
        return False
    backend_spec = find_spec(package_name)
    if backend_spec is None:
        try:
            _get_user_input(_should_install_backend, package_name)
        except Exception as e:
            print(e)
            return False
    else:
        global _backend_is_installed
        _backend_is_installed = True
        print(f"{Fore.GREEN}Backend {package_name} found.", end=" ")
        print(f"Installed at {backend_spec.origin}\n")

    _get_user_input(_add_alias_for_backend)

    if _backend_is_installed:
        global _imported_backend
        global backend
        backend["name"] = package_name
        print(Style.BRIGHT + f"Importing {package_name} for type checking...")
        _imported_backend = import_module(package_name)

    return True


def _add_alias_for_backend():
    ret = input("Enter alias for Python import (Press ENTER to skip): ")
    ret = ret.strip(" ")
    if ret == "":
        return True
    backend["alias"] = ret
    return True


if __name__ == "__main__":
    init(autoreset=True)

    _get_user_input(_get_backend)
    for key in config:
        _query_input(key)

    print("\n==== Backend ====")
    pprint.pprint(backend, sort_dicts=False)
    print("==== Config  ====\n")
    pprint.pprint(config, sort_dicts=False)
    pprint.pprint(config_lists, sort_dicts=False)
    pprint.pprint(config_flags, sort_dicts=False)

    print("\nConfirm generation? [y/N]\n")
