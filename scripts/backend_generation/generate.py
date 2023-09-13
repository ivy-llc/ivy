import sys
import subprocess
import pprint
import inspect
import json
from colorama import Fore, Style, init
from importlib import import_module
from importlib.util import find_spec
from tree_generation import generate as generate_backend
from shared import BackendNativeObject
from dataclasses import asdict


all_devices = ("cpu", "gpu", "tpu")
all_ivy_dtypes = (
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "bfloat16",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
    "bool",
)

all_int_dtypes = (
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
)


all_uint_dtypes = (
    "uint8",
    "uint16",
    "uint32",
    "uint64",
)

all_float_dtypes = (
    "bfloat16",
    "float16",
    "float32",
    "float64",
)

all_complex_dtypes = (
    "complex64",
    "complex128",
)

_imported_backend = None
_backend_is_installed = False

backend = {"name": None, "alias": None}
config_natives = {
    "NativeArray": asdict(BackendNativeObject(name="None", namespace="")),
    "NativeVariable": asdict(BackendNativeObject(name="None", namespace="")),
    "NativeDevice": asdict(BackendNativeObject(name="None", namespace="")),
    "NativeDtype": asdict(BackendNativeObject(name="None", namespace="")),
    "NativeShape": asdict(BackendNativeObject(name="None", namespace="")),
    "NativeSparseArray": asdict(BackendNativeObject(name="None", namespace="")),
}
config_valids = {
    "valid_devices": list(all_devices),
    "valid_int_dtypes": list(all_int_dtypes),
    "valid_float_dtypes": list(all_float_dtypes),
    "valid_complex_dtypes": list(all_complex_dtypes),
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
        "\nPress ENTER to skip, use full namespace\n"
        f"Enter a value for {Style.BRIGHT + key + Style.NORMAL} "
        "(case sensistive) "
        f"default: '{Style.BRIGHT}{config_natives[key]['name']}{Style.NORMAL}': "
    )
    if ret != "" and _imported_backend is not None:
        parsed = ret.strip().rpartition(".")
        try:
            if parsed[1] == "":
                # Primitve type
                try:
                    obj = __builtins__.__dict__[parsed[-1]]
                except KeyError:
                    print(Fore.RED + f"{parsed[-1]} is not a primitive object.")
                    return False
            else:
                try:
                    mod = import_module(parsed[0])
                except ModuleNotFoundError:
                    print(Fore.RED + f"failed to import {parsed[0]}")
                    return False
                try:
                    obj = getattr(mod, parsed[-1])
                except AttributeError:
                    print(Fore.RED + f"{parsed[-1]} is not found in module.")
                    return False
            if not inspect.isclass(obj):
                print(Fore.RED + f"{obj} is not a class.")
                return False
            print(Fore.GREEN + f"Found class: {obj}")
            # Use alias if exists
            if backend["alias"] is not None:
                modified_namespace = parsed[0].replace(
                    backend["name"], backend["alias"], 1
                )
            config_natives[key] = asdict(
                BackendNativeObject(name=parsed[-1], namespace=modified_namespace)
            )
            return True
        except KeyError:
            print(Fore.RED + f"Couldn't find {ret}")
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
            # Install backend
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package_name]
            )
            global _backend_is_installed
            _backend_is_installed = True
            with open("../../requirements/optional.txt", "a") as reqr_file:
                reqr_file.write("\n" + package_name + "\n")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                Fore.RED + f"Installing {package_name} failed. {e}"
            ) from e
    elif ret.lower() == "n":
        print(
            Fore.YELLOW
            + "Will continue without backend installed, "
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

        def _import_name():
            ret = (
                input(
                    f"Enter Import name for package {package_name}, "
                    f"Press Enter to use {package_name}: "
                )
                .strip()
                .lower()
            )
            if ret == "":
                backend["name"] = package_name
            else:
                backend["name"] = ret
            return True

        _get_user_input(_import_name)

        global _imported_backend
        print(Style.BRIGHT + f"Importing {backend['name']} for type checking...")
        try:
            _imported_backend = import_module(backend["name"])
            return True
        except Exception as e:
            print(Fore.RED + f"Failed to import {backend['name']}:{e}")
            return False

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


def _call_generate_tree(config_name: str):
    ret = input(Style.BRIGHT + "\n:: Procced with generation? [Y/n]\n").strip().lower()
    if ret == "y":
        generate_backend(config_name)
        return True
    elif ret == "n":
        return True
    return False


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
    int_dtypes = set(all_int_dtypes).difference(all_uint_dtypes)
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
    fullset_mapping = {
        "valid_dtypes": all_ivy_dtypes,
        "valid_numeric_dtypes": all_int_dtypes + all_float_dtypes + all_complex_dtypes,
        "valid_int_dtypes": all_int_dtypes,
        "valid_uint_dtypes": all_uint_dtypes,
        "valid_float_dtypes": all_float_dtypes,
        "valid_complex_dtypes": all_complex_dtypes,
        "valid_devices": all_devices,
    }

    for key, value in config_valids.copy().items():
        all_items = fullset_mapping[key]
        invalid_items = list(set(all_items).difference(value))
        config_valids["in" + key] = invalid_items

    for key in config_valids["valid_dtypes"]:
        new_key = "native_" + key
        config_natives[new_key] = asdict(BackendNativeObject(name="None", namespace=""))
        _get_user_input(_update_native_config_value, new_key)

    for key in config_valids["invalid_dtypes"]:
        new_key = "native_" + key
        config_natives[new_key] = asdict(BackendNativeObject(name="None", namespace=""))

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

    json_config = {**backend, **config_flags, **config_natives}
    for k, v in config_valids.items():
        json_config[k] = list(v)

    file_path = None
    with open("config.json", "w") as file:
        json.dump(json_config, file, indent=4)
        file_path = file.name

    print(f"Config saved to {file_path}.")
    _get_user_input(_call_generate_tree, file_path)
