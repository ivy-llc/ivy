import ivy
import sys
import subprocess
import pprint
import inspect
from colorama import Fore, Style, init
from importlib import import_module
from importlib.util import find_spec


_imported_backend = None
_backend_is_installed = False

backend = {"name": None, "alias": None}
config_natives = {
    "NativeArray": None,
    "NativeVariable": None,
    "NativeDevice": None,
    "NativeDtype": None,
    "NativeShape": None,
    "NativeSparseArray": None,
}
config_valids = {
    "valid_devices": list(ivy.all_devices),
    "valid_int_dtypes": list(ivy.all_int_dtypes),
    "valid_float_dtypes": list(ivy.all_float_dtypes),
    "valid_complex_dtypes": list(ivy.all_complex_dtypes),
}
config_flags = {
    "native_inplace_support": False,
    "supports_gradients": False,
}


def _get_user_input(fn, *args, **kwargs):
    # A basic loop to get user input and handle keyboard interrupt
    while True:
        try:
            ret = fn(*args, **kwargs)
            if ret:
                break
        except KeyboardInterrupt:
            print("Aborted.")
            exit()


def _update_native_config_value(key):
    # Handle the logic for updating native config
    ret = input(
        "\nPress ENTER to skip\n"
        f"Enter a value for {Style.BRIGHT + key + Style.NORMAL} "
        "(case sensistive) "
        f"default: '{Style.BRIGHT}{config_natives[key]}{Style.NORMAL}': "
    )
    ret = ret.strip(" ")
    if ret != "" and _imported_backend is not None:
        try:
            # TODO handle nested classes
            # Check if it's in the backend dict
            obj = _imported_backend.__dict__[ret]
            if not inspect.isclass(obj):
                print(Fore.RED + f"{obj} is not a class.")
                return False
            print(Fore.GREEN + f"Found class: {obj}")
            config_natives[key] = obj
            return True
        except KeyError:
            print(Fore.RED + f"Couldn't find {backend['name']}.{ret}")
            return False
    return True


def _should_install_backend(package_name):
    # Check if backend is installed, otherwise install it locally for type hints
    ret = input(
        f"Backend {package_name} isn't installed locally, "
        "would you like to install it? [Y/n]\n"
    )
    if ret.lower() == "y":
        try:
            # TODO add it to Dockerfile/requirements
            # Install backend
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
    # Main function to query backend
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
    # Handle adding an alias for backend import
    ret = input("Enter alias for Python import (Press ENTER to skip): ")
    ret = ret.strip(" ")
    if ret == "":
        return True
    backend["alias"] = ret
    return True


def _update_flag_config_value(key):
    # Handle flag input and update it's value
    ret = input(
        f"\nToggle flag {Style.BRIGHT}{key}{Style.NORMAL} [Y/n]? "
        f"default: {Fore.RED}'{config_flags[key]}'"
        f"{Style.RESET_ALL}: "
    )
    ret = ret.strip(" ").lower()
    if ret == "y":
        config_flags[key] = not config_flags[key]
        return True
    elif ret == "n" or ret == "":
        return True
    print(Fore.RED + f"{ret} not understood.")
    return False


def _update_valid_config_value(key):
    # Handle valids selection
    print(f"Select items to remove from list {Style.BRIGHT}{key}:\n")
    for i, item in enumerate(config_valids[key]):
        print(f"{i}. {item}")
    ret = input("\nPress ENTER to skip. Enter numbers (space seperated): ")
    ret = ret.strip("")
    if ret == "":
        return True
    indicies = ret.split(" ")
    indicies = [int(item.strip(" ")) for item in indicies]
    for i in sorted(indicies, reverse=True):
        del config_valids[key][i]
    return True


if __name__ == "__main__":
    init(autoreset=True)

    _get_user_input(_get_backend)

    for key in config_natives:
        _get_user_input(_update_native_config_value, key)

    for key in config_flags:
        _get_user_input(_update_flag_config_value, key)

    for key in config_valids:
        _get_user_input(_update_valid_config_value, key)

    # Add uint dtypes
    int_dtypes = set(ivy.all_int_dtypes).difference(ivy.all_uint_dtypes)
    config_valids["valid_uint_dtypes"] = (
        set(config_valids["valid_int_dtypes"]) - int_dtypes
    )

    # Add numeric dtypes and valid dtypes
    config_valids["valid_numeric_dtypes"] = (
        config_valids["valid_int_dtypes"]
        + config_valids["valid_float_dtypes"]
        + config_valids["valid_complex_dtypes"]
    )
    config_valids["valid_dtypes"] = config_valids["valid_numeric_dtypes"] + ["bool"]

    # Create Invalid dict
    for key, value in config_valids.copy().items():
        all_items = ivy.__dict__[key]
        invalid_items = list(set(all_items).difference(value))
        config_valids["in" + key] = invalid_items

    print("\n:: Backend\n")
    pprint.pprint(backend, sort_dicts=False)
    print("\n:: Config\n")
    pprint.pprint(config_natives, sort_dicts=False)

    # Print valids
    for key in config_valids.keys():
        if key.startswith("in"):
            continue
        valid_items = config_valids[key]
        invalid_items = config_valids["in" + key]
        print("\n:: " + key.partition("_")[-1])
        print(Fore.GREEN + "valid > " + valid_items.__str__())
        print(Fore.RED + "invalid > " + invalid_items.__str__())

    # Print flags
    for key, value in config_flags.items():
        flag_color = Fore.GREEN if value else Fore.RED
        print(f"\n:: {key}: {flag_color}{value}")

    print(Style.BRIGHT + "\n:: Procced with generation? [Y/n]\n")
