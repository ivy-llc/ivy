"""Main driver code for the Source-to-Source Translator."""

# global
import ctypes
from dataclasses import is_dataclass
from enum import Enum
import functools
import hashlib
import importlib
import inspect
import logging
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from types import FunctionType, BuiltinFunctionType, MethodType, ModuleType
from typing import Union
import uuid
import warnings
import gast
import ivy  # type: ignore

# local
from .configurations_container import ConfigurationsContainer
from .translators_container import TranslatorsContainer
from .utils.api_utils import copy_module
from .utils.cache_utils import (
    PRELOAD_CACHE,
)
from .utils.conversion_utils import (
    BUILTIN_LIKELY_MODULE_NAMES,
    is_builtin_function,
)
from .utils.inspect_utils import (
    _validate_object,
)
from .utils.source_utils import (
    get_object_from_translated_directory,
    get_new_output_dir_name,
    sanitize_dir_name,
)


# NOTE: needed for infuser
LOG_DATA = True  # whether to log telemetry data to the server
EXISTING_TRANSPILATION = False  # whether the current transpilation is an existing one, i.e., the function has been transpiled before and is saved in the translated directory
VERIFIED_UNRESTRICTED = False  # whether the fn/cls being transpiled is verified to be from a unrestricted module (such as kornia)
TARGET = "tensorflow"


# Helpers #
# ------- #


def _animate(stop_animation, animation_str):
    """
    Displays ellipsis animation during transpilation.
    """
    ellipsis = ["   ", ".  ", ".. ", "..."]
    idx = 0
    terminal_width = shutil.get_terminal_size().columns
    while not stop_animation.is_set():
        truncated_str = animation_str[
            : terminal_width - 4
        ]  # Reserve space for ellipsis
        write_str = f"{truncated_str}{ellipsis[idx % len(ellipsis)]}"
        sys.stdout.write(f"\r{write_str}")
        sys.stdout.flush()
        idx += 1
        time.sleep(0.32)  # speed of animation
    sys.stdout.write("\r" + " " * len(write_str) + "\r")  # clear the line


def _get_machine_id():
    """
    Attempts to generate a unique machine identifier using several methods in a prioritized order.
    Used in infuser.

    This function tries the following methods to obtain a unique identifier:
    1. Uses the MAC address from `uuid.getnode()` if it is globally unique (i.e., not locally administered).
    2. Gathers all MAC addresses using `psutil.net_if_addrs()`, hashes them collectively, and uses this hash if it meets the length criteria.
    3. Combines various system properties (hostname, OS type, architecture, CPU info, and OS version), hashes this combination, and uses it if the hash is sufficiently long.
    4. Uses a random UUID as a last resort if all other methods fail.
    """
    try:
        # First attempt: Use the MAC address via uuid.getnode()
        node = uuid.getnode()
        if (node >> 40) & 0x02 == 0:
            # The "locally administered" bit is not set; node is a valid MAC address
            return str(node)
        else:
            # Second attempt: Use psutil to get MAC addresses
            try:
                import psutil

                mac_addresses = set()
                for interface, addrs in psutil.net_if_addrs().items():
                    for addr in addrs:
                        # Check for MAC address families
                        if (
                            hasattr(psutil, "AF_LINK") and addr.family == psutil.AF_LINK
                        ) or (
                            hasattr(socket, "AF_PACKET")
                            and addr.family == socket.AF_PACKET
                        ):
                            mac = addr.address
                            mac_bytes = bytes(int(b, 16) for b in mac.split(":"))

                            # do not include addresses that can be random
                            if (
                                mac_bytes[0]
                                & 0x02  # Locally administered MAC address (potentially random)
                                or mac_bytes[0] & 0x01  # Multicast MAC address
                            ):
                                continue

                            if mac and mac != "00:00:00:00:00:00":
                                mac_addresses.add(mac)
                if mac_addresses:
                    # Create a consistent and unique identifier
                    mac_addresses = sorted(mac_addresses)
                    unique_id = "".join(mac_addresses)
                    unique_id_hash = hashlib.md5(unique_id.encode("utf-8")).hexdigest()
                    if len(unique_id_hash) > 255:
                        unique_id_hash = unique_id_hash[:255]
                    if (
                        len(unique_id_hash) > 10
                    ):  # Only accept as valid if over a certain length
                        return unique_id_hash
            except:
                pass

            # Third attempt: Get unique id with system-specific methodologies
            # based on https://github.com/keygen-sh/py-machineid/blob/master/machineid/__init__.py
            try:
                try:
                    from winregistry import WinRegistry  # type: ignore
                except ImportError:
                    WinRegistry = None

                def sanitize(id):
                    return re.sub(r"[\x00-\x1f\x7f-\x9f\s]", "", id).strip()

                def execute(cmd):
                    try:
                        return subprocess.run(
                            cmd,
                            shell=True,
                            capture_output=True,
                            check=True,
                            encoding="utf-8",
                        ).stdout.strip()
                    except:
                        return None

                def read(path):
                    try:
                        with open(path) as f:
                            return f.read().strip()
                    except:
                        return None

                def reg(registry, key):
                    try:
                        with WinRegistry() as reg:
                            return reg.read_entry(registry, key).value.strip()
                    except:
                        return None

                id = None
                if sys.platform == "darwin":
                    id = execute(
                        "ioreg -d2 -c IOPlatformExpertDevice | awk -F\\\" '/IOPlatformUUID/{print $(NF-1)}'"
                    )
                elif sys.platform in ("win32", "cygwin", "msys"):
                    if WinRegistry is not None:
                        id = reg(
                            r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Cryptography",
                            "MachineGuid",
                        )
                    else:
                        id = execute(
                            "powershell.exe -ExecutionPolicy bypass -command (Get-CimInstance -Class Win32_ComputerSystemProduct).UUID"
                        )
                    if not id:
                        id = execute("wmic csproduct get uuid").split("\n")[2].strip()
                elif sys.platform.startswith("linux"):
                    id = read("/var/lib/dbus/machine-id")
                    if not id:
                        id = read("/etc/machine-id")
                    if not id:
                        cgroup = read("/proc/self/cgroup")
                    if cgroup and "docker" in cgroup:
                        id = execute("head -1 /proc/self/cgroup | cut -d/ -f3")
                    if not id:
                        mountinfo = read("/proc/self/mountinfo")
                    if mountinfo and "docker" in mountinfo:
                        id = execute(
                            "grep -oP '(?<=docker/containers/)([a-f0-9]+)(?=/hostname)' /proc/self/mountinfo"
                        )
                    if not id and "microsoft" in platform.uname().release:  # wsl
                        id = execute(
                            "powershell.exe -ExecutionPolicy bypass -command '(Get-CimInstance -Class Win32_ComputerSystemProduct).UUID'"
                        )
                elif sys.platform.startswith(("openbsd", "freebsd")):
                    id = read("/etc/hostid")
                    if not id:
                        id = execute("kenv -q smbios.system.uuid")

                assert id
                return sanitize(id)
            except:
                pass

            # Fourth attempt: Hash system properties
            try:
                import getpass

                system_data = [
                    getpass.getuser() or "",
                    socket.gethostname(),
                    platform.system(),
                    platform.machine(),
                ]
                unique_id_str = "-".join(filter(None, system_data))
                unique_id_hash = hashlib.sha256(
                    unique_id_str.encode("utf-8")
                ).hexdigest()
                if len(unique_id_hash) > 255:
                    unique_id_hash = unique_id_hash[:255]
                if (
                    len(unique_id_hash) > 10
                ):  # Only accept as valid if over a certain length
                    return unique_id_hash
            except:
                pass

            # Final fallback: Use the random uuid
            return str(uuid.getnode())
    except:
        raise ivy.exceptions.IvyException("Unable to verify device.") from None


def _is_frozen(obj):
    """
    Checks if a object is frozen
    """
    if is_dataclass(obj) and hasattr(obj, "__dataclass_params__"):
        return obj.__dataclass_params__.frozen
    return isinstance(obj, frozenset)


def _reload_args(args, kwargs, output_dir):
    """
    Reloads any objects within the args/kwargs which reference the transpiled outputs
    module to ensure they're referencing the latest version of this module.
    """
    args_list = list(args)

    for idx, arg in enumerate(args_list):
        args_list[idx] = _reload_obj(arg, output_dir)

    for key in kwargs:
        arg = kwargs[key]
        kwargs[key] = _reload_obj(arg, output_dir)

    return tuple(args_list), kwargs


def _reload_obj(obj, output_dir):
    """
    Reloads an object if it references the transpiled outputs module
    to ensure it's using the latest version of this module.
    """
    if not _is_frozen(obj):
        if output_dir in str(obj.__class__):
            reloaded_module = importlib.import_module(obj.__class__.__module__)
            obj.__class__ = getattr(reloaded_module, obj.__class__.__name__)
            setattr(obj, "__already_s2s", TARGET)
        elif hasattr(obj, "__class__") and obj.__class__ is not None and output_dir in obj.__class__.__module__:
            try:
                reloaded_module = importlib.import_module(obj.__class__.__module__)
                obj.__class__ = getattr(reloaded_module, obj.__class__.__name__)
                setattr(obj, "__already_s2s", TARGET)
            except:
                pass
        elif hasattr(obj, "__module__") and obj.__module__ is not None and output_dir in obj.__module__:
            try:
                reloaded_module = importlib.import_module(obj.__module__)
                obj = getattr(reloaded_module, obj.__name__)
                setattr(obj, "__already_s2s", TARGET)
            except:
                pass

    return obj


def _reload_variables(output_dir, frame=None):
    """
    Recursively reloads variables within the caller frames to reference the reloaded translated modules.
    """

    if frame is None:
        frame = inspect.currentframe()

    caller_frame = frame.f_back
    if caller_frame is None:
        return

    # reload globals
    for key, value in caller_frame.f_globals.items():
        caller_frame.f_globals[key] = _reload_obj(value, output_dir)

    # reload locals
    for key, value in caller_frame.f_locals.items():
        ctypes.pythonapi.PyFrame_LocalsToFast(
            ctypes.py_object(caller_frame), ctypes.c_int(1)
        )
        caller_frame.f_locals[key] = _reload_obj(value, output_dir)
        ctypes.pythonapi.PyFrame_LocalsToFast(
            ctypes.py_object(caller_frame), ctypes.c_int(0)
        )

    _reload_variables(output_dir, caller_frame)


def _set_debug_level(level):
    if level == 0:
        logging.getLogger("asyncio").setLevel(logging.ERROR)
        logging.getLogger().setLevel(logging.ERROR)
        ivy.LoggingMode().set_logging_mode("ERROR")
    elif level == 1:
        logging.getLogger("asyncio").setLevel(logging.WARNING)
        logging.getLogger().setLevel(logging.WARNING)
        ivy.LoggingMode().set_logging_mode("WARNING")
    elif level == 2:
        logging.getLogger("asyncio").setLevel(logging.INFO)
        logging.getLogger().setLevel(logging.INFO)
        ivy.LoggingMode().set_logging_mode("INFO")
    else:
        logging.getLogger("asyncio").setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
        ivy.LoggingMode().set_logging_mode("DEBUG")


def _transpile_required_args(args, kwargs, source, target, output_dir):
    """
    Transpile any args that need transpiling before a transpiled callable can be correctly called.
    """

    args_list = list(args)
    for i, arg in enumerate(args_list):
        if isinstance(arg, Enum) and not (
            hasattr(arg.__class__, "__already_s2s")
            or arg.__class__.__module__.startswith(output_dir)
        ):
            args_list[i] = transpile(arg.__class__, source=source, target=target)(
                arg.value
            )
    args = tuple(args_list)

    for key, arg in kwargs.items():
        if isinstance(arg, Enum) and not (
            hasattr(arg.__class__, "__already_s2s")
            or arg.__class__.__module__.startswith(output_dir)
        ):
            kwargs[key] = transpile(arg.__class__, source=source, target=target)(
                arg.value
            )

    return args, kwargs


class AssignmentFinder(gast.NodeVisitor):
    def __init__(self):
        self.assignments = []

    def visit_Assign(self, node: gast.Assign):
        # Capture the assignment node along with its line range
        self.assignments.append(node)
        self.generic_visit(node)


def find_containing_assignment(filename: str, target_line: int, cell_code: str = None):
    """
    Find the assignment statement containing the target line.
    Works with both regular files and IPython/Colab cells.

    Args:
        filename: Path to the source file or IPython input reference
        target_line: Line number we're interested in
        cell_code: The source code when running in IPython/Colab cell

    Returns:
        Tuple of (ast.Assign node, start_line, end_line) or None if not found
    """
    try:
        if cell_code is not None:
            # Use provided cell code for IPython/Colab
            source = cell_code
        else:
            # Regular file handling
            with open(filename, "r") as f:
                source = f.read()

        # Parse the source code
        tree = gast.parse(source)

        # Find all assignments
        finder = AssignmentFinder()
        finder.visit(tree)

        # Check each assignment's line range
        for assign_node in finder.assignments:
            start_line = assign_node.lineno
            end_line = assign_node.end_lineno

            # Check if our target line falls within this range
            if start_line <= target_line <= end_line:
                return assign_node, start_line, end_line

        return None
    except Exception as e:
        logging.info(f"Error finding assignment: {e}")
        return None


def extract_first_call_node(expression):
    """
    Extracts the first call node from a complex expression.
    Returns the extracted call node as a string.

    Examples:
        "foo()()' -> 'foo()'
        "foo().bar()" -> "foo()"
        "a.b.c().d()" -> "a.b.c()"
        "a.b(cd(), ef())()" -> "a.b(cd(), ef())"
    """

    def find_matching_parenthesis(s, start):
        """Find the position of matching closing parenthesis."""
        count = 1
        i = start + 1
        while i < len(s):
            if s[i] == "(":
                count += 1
            elif s[i] == ")":
                count -= 1
                if count == 0:
                    return i
            i += 1
        return -1

    def find_call_start(s, call_end):
        """
        Find the start position of the call node by walking backwards
        to include any attribute access (dots) and identifiers.
        """
        i = call_end - 1
        while i >= 0:
            # Skip whitespace
            while i >= 0 and s[i].isspace():
                i -= 1

            if i < 0:
                break

            # If we hit an operator or separator that's not a dot,
            # we've gone too far back
            if (
                s[i]
                not in ".abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
            ):
                i += 1
                break

            i -= 1

        return max(0, i + 1)

    # Find the first opening parenthesis
    start_paren = expression.find("(")
    if start_paren == -1:
        return expression.strip()

    # Find its matching closing parenthesis
    end_paren = find_matching_parenthesis(expression, start_paren)
    if end_paren == -1:
        raise ValueError("Unmatched parenthesis in expression")

    # Find the start of the call node by walking backwards
    call_start = find_call_start(expression, start_paren)

    # Extract the complete call node
    return expression[call_start : end_paren + 1]


def _wrap_callable(callable_obj, source, target, output_dir, module_name):
    """
    Wraps a callable such that it will be transpiled before being called.
    """

    # wrap any classmethods
    if inspect.isclass(callable_obj):
        for attr_name in dir(callable_obj):
            if not attr_name.startswith("__"):
                try:
                    attr = eval("callable_obj." + attr_name)
                    # Check for classmethod
                    if (
                        hasattr(attr, "__self__")
                        and callable_obj.__module__ == attr.__module__
                        and callable_obj.__module__ not in BUILTIN_LIKELY_MODULE_NAMES
                    ):
                        setattr(
                            callable_obj,
                            attr_name,
                            _wrap_classmethod(attr, source, target, output_dir),
                        )
                    # Check for staticmethod
                    elif (
                        attr.__module__.startswith(module_name)
                        and attr.__module__ not in BUILTIN_LIKELY_MODULE_NAMES
                        and isinstance(attr, (FunctionType, BuiltinFunctionType))
                        and not hasattr(attr, "__self__")
                    ):
                        # verify it's actually a staticmethod
                        if inspect.getsource(attr).strip().startswith("@staticmethod"):
                            setattr(
                                callable_obj,
                                attr_name,
                                _wrap_staticmethod(
                                    attr, callable_obj, source, target, output_dir
                                ),
                            )
                except Exception:
                    pass

    @functools.wraps(callable_obj)
    def wrapper(*args, **kwargs):
        nonlocal callable_obj

        if getattr(callable_obj, "__already_s2s", None) == target:
            args, kwargs = _transpile_required_args(
                args, kwargs, source, target, output_dir
            )
            args, kwargs = _reload_args(args, kwargs, output_dir)
            callable_obj = _reload_obj(callable_obj, output_dir)
            return callable_obj(*args, **kwargs)

        # transpile the callable
        callable_obj = transpile(
            callable_obj, source=source, target=target, output_dir=output_dir
        )

        args, kwargs = _transpile_required_args(
            args, kwargs, source, target, output_dir
        )
        args, kwargs = _reload_args(args, kwargs, output_dir)
        callable_obj = _reload_obj(callable_obj, output_dir)

        # Get caller's frame information
        caller_frame = inspect.currentframe().f_back

        # Find the actual calling frame (skip intermediate frames if any)
        while caller_frame:
            if caller_frame.f_code.co_name != wrapper.__name__:
                break
            caller_frame = caller_frame.f_back

        if caller_frame:
            try:
                # Get the source line
                calling_file = caller_frame.f_code.co_filename
                calling_line = caller_frame.f_lineno

                # Check if we're in IPython/Colab environment
                try:
                    from IPython import get_ipython

                    ipython = get_ipython()
                except ImportError:
                    logging.info(
                        "Could not import IPython. Assuming not in IPython/Colab environment."
                    )
                    ipython = None
                if ipython and calling_file.startswith("<ipython-input-"):
                    # Get the cell number from the filename
                    cell_num = calling_file.split("-")[2]

                    # Get the cell's source code
                    cell_code = ipython.user_ns["In"][int(cell_num)]
                    result = find_containing_assignment(
                        calling_file, calling_line, cell_code
                    )
                else:
                    result = find_containing_assignment(calling_file, calling_line)

                if result is None:
                    logging.info(
                        f"Could not find assignment containing the calling line {calling_line}"
                    )
                    return callable_obj(*args, **kwargs)

                # Get the actual source code
                assign_node, _, _ = result
                calling_statement = gast.unparse(assign_node.value).strip()
                calling_statement = extract_first_call_node(calling_statement)

                # Get the caller's globals and locals
                caller_globals = caller_frame.f_globals
                caller_locals = caller_frame.f_locals

                # Re-execute the calling statement in the original context
                eval(calling_statement, caller_globals, caller_locals)
            except Exception as e:
                logging.info(f"Error in wrapper: {e}")
                # If there's an error, just return the original result
                return callable_obj(*args, **kwargs)

        if not caller_frame:
            logging.info("No caller frame found.")
        return callable_obj(*args, **kwargs)

    return wrapper


def _wrap_classmethod(classmethod_obj, source, target, output_dir):
    """
    Wraps a classmethod such that the class it belongs to will be transpiled before calling.
    """

    @functools.wraps(classmethod_obj)
    def wrapper(*args, **kwargs):
        nonlocal classmethod_obj

        if getattr(classmethod_obj.__self__, "__already_s2s", None) == target:
            args, kwargs = _transpile_required_args(
                args, kwargs, source, target, output_dir
            )
            args, kwargs = _reload_args(args, kwargs, output_dir)
            classmethod_obj = _reload_obj(classmethod_obj, output_dir)
            return classmethod_obj(*args, **kwargs)

        # transpile the class this classmethod belongs to
        transpiled_cls = transpile(
            classmethod_obj.__self__,
            source=source,
            target=target,
            output_dir=output_dir,
        )

        args, kwargs = _transpile_required_args(
            args, kwargs, source, target, output_dir
        )
        args, kwargs = _reload_args(args, kwargs, output_dir)
        transpiled_cls = _reload_obj(transpiled_cls, output_dir)
        transpiled_cls_method = getattr(transpiled_cls, classmethod_obj.__name__)
        return transpiled_cls_method(*args, **kwargs)

    return wrapper


def _wrap_staticmethod(static_method, owning_class, source, target, output_dir):
    """
    Wraps a staticmethod such that the class it belongs to will be transpiled before calling.
    """

    @functools.wraps(static_method)
    def wrapper(*args, **kwargs):
        nonlocal static_method
        if getattr(owning_class, "__already_s2s", None) == target:
            args, kwargs = _transpile_required_args(
                args, kwargs, source, target, output_dir
            )
            args, kwargs = _reload_args(args, kwargs, output_dir)
            static_method = _reload_obj(static_method, output_dir)
            return static_method(*args, **kwargs)

        # transpile the class this staticmethod belongs to
        transpiled_cls = transpile(
            owning_class,
            source=source,
            target=target,
            output_dir=output_dir,
        )

        args, kwargs = _transpile_required_args(
            args, kwargs, source, target, output_dir
        )
        args, kwargs = _reload_args(args, kwargs, output_dir)
        transpiled_cls = _reload_obj(transpiled_cls, output_dir)
        transpiled_static_method = getattr(transpiled_cls, static_method.__name__)
        return transpiled_static_method(*args, **kwargs)

    return staticmethod(wrapper)  # Preserve staticmethod nature


def wrap_module(module, source, target, output_dir, inplace=False):
    """
    Wraps all callables in a module with a wrapper that will transpile them when called,
    ensuring that all module attributes pointing to the same callable are updated.
    """

    callable_map = {}
    visited_modules = {}
    name = "/".join(module.__name__.split("."))

    def collect_callables(module, inplace):
        if module in visited_modules:
            return visited_modules[module]

        module_copy = module if inplace else copy_module(module)
        visited_modules[module] = module_copy

        for m in dir(module):
            val = getattr(module, m)
            if callable(val):
                callable_map.setdefault(val, []).append((module_copy, m))
            elif (
                isinstance(val, ModuleType)
                and "__file__" in val.__dict__
                and name in val.__file__
            ):
                submodule_copy = collect_callables(val, False)
                setattr(module_copy, m, submodule_copy)

        return module_copy

    module_copy = collect_callables(module, inplace)

    for val, locations in callable_map.items():
        if inspect.isclass(val):
            try:
                # reconstruct the class
                val = type(val.__name__, val.__bases__, dict(val.__dict__))
            except Exception:
                pass

        if not hasattr(val, "__module__") or all(
            [x not in val.__module__ for x in ["ivy.", "source_to_source_translator."]]
        ):
            wrapped_val = _wrap_callable(
                val, source, target, output_dir, module_name=module.__name__
            )
            for mod, attr_name in locations:
                setattr(mod, attr_name, wrapped_val)

    return module_copy


# Main #
# ---- #


def translate(
    object,
    source: str = "torch",
    target: str = "tensorflow",
    inplace: bool = False,
    reuse_existing: bool = True,
    output_dir: str = "ivy_transpiled_outputs/",
) -> Union[MethodType, FunctionType, type]:
    """
    Converts a given object (class/function) from one framework to another.

    This function performs source-to-source translation of a given object from the source framework
    to the target framework.

    The object can be translated between two frameworks or between the Ivy IR as well
    e.g. (source="torch_frontend", target="ivy") or (source="torch_frontend", target="tensorflow") etc.

    Args:
        object: The object (class/function) to be translated.
        source (str, optional): The source framework. Defaults to 'torch'.
        target (str, optional): The target framework. Defaults to 'tensorflow'.
        inplace (bool, optional): Whether modules should be lazily transpiled inplace. Defaults to False.
        reuse_existing (bool, optional): If True, the function will check if `object`
                                         already exists in the translated directory and reuse it.
                                         If False, it will re-translate `object`,
                                         even if it already exists in the directory, and overwrite
                                         the old implementation. Defaults to 'True'.
        output_dir (str, optional): The path to the directory where translated files will be saved.
                                    Defaults to 'ivy_transpiled_outputs/' in the current working directory.

    Returns:
        The translated object.
    """

    # Configure the logger
    # the higher the debug level, the more information is printed, default to level 1
    DEBUG = int(os.getenv("DEBUG", 1))
    _set_debug_level(DEBUG)

    global EXISTING_TRANSPILATION, TARGET
    EXISTING_TRANSPILATION = False
    TARGET = target

    if isinstance(object, ModuleType):
        return wrap_module(object, source, target, output_dir.replace("/", ""), inplace=inplace)
    elif is_builtin_function(object):
        # immediately return builtin functions; these don't need to be transpiled
        return object

    infuser_injection = "S2S API KEY"  # needed for infuser

    # 0. Return directly if already translated
    if getattr(object, "__already_s2s", None) == target:
        EXISTING_TRANSPILATION = True
        return object

    # 1. Initialize cache preloading
    # PRELOAD_CACHE.start_cache_loading_async()

    # 1. Load the configurations container
    output_dir = sanitize_dir_name(output_dir)
    configurations_container: ConfigurationsContainer = ConfigurationsContainer(
        base_output_dir=output_dir
    )
    configurations_container.load_configurations(source=source, target=target)

    # 1a. Return directly if decorated with @not_translate and DO NOT Cache it
    options = getattr(
        object, configurations_container.configuration["CONVERSION_OPTIONS"], None
    )
    if options is not None and options.not_convert:
        return object.__func__ if inspect.ismethod(object) else object

    # 1b. Return if `reuse_existing=True` and object exists in the translated directory
    translated_object = get_object_from_translated_directory(
        object,
        translated_dir=get_new_output_dir_name(
            target=target,
            base_output_dir=output_dir,
        ),
        base_output_dir=output_dir,
        target=target,
        reuse_existing=reuse_existing,
    )
    if translated_object:
        EXISTING_TRANSPILATION = True
        logging.debug(
            f"Reusing existing object ... {translated_object.__name__} from translated directory. "
        )
        setattr(translated_object, "__already_s2s", target)
        return translated_object

    # 2a. Unset the backend it if it was set globally. This is because certain transformers
    # rely on ivy's global dict not being modified (eg: BaseGlobalsTransformer) to work correctly.
    backend_str = ivy.current_backend_str() if ivy.backend_stack else None
    ivy.unset_backend()

    if (
        DEBUG == 1
    ):  # Allows the animation to be turned off with DEBUG=0, and it won't appear with other logging levels (DEBUG > 1)
        warnings.filterwarnings(
            "ignore"
        )  # Disables all warnings that can appear from native frameworks, etc
        _set_debug_level(0)

        # Create an event to stop the animation
        stop_animation = threading.Event()

        # Start the animation in a separate thread
        obj_name = object.__name__ if hasattr(object, "__name__") else repr(object)
        object_type = (
            "function"
            if inspect.isfunction(object)
            else "class" if inspect.isclass(object) else "object"
        )
        animation_str = f"Transpiling {obj_name} from {source} to {target}. This could take a few minutes if you're transpiling this {object_type} for the first time"
        animation_thread = threading.Thread(
            target=_animate, args=(stop_animation, animation_str)
        )
        animation_thread.start()

    try:
        # 2b. Validate and retrieve the correct object from the provided source framework.
        object = _validate_object(object, source=source)

        # 2c. Initialize the translators container which internally initializes the cache
        # and sets up the translation process by initializing a chain of translators (if needed)
        # based on the values of `source` and `target`.
        translators_container: TranslatorsContainer = TranslatorsContainer(
            configurations_container=configurations_container
        )
        translated_object = translators_container.load_translators()

        # 2d. Run the chain of translators and return
        translated_object = translators_container.run_translators(
            object=object, reuse_existing=reuse_existing, base_output_dir=output_dir
        )

        # 3. Restore the globally set backend if needed
        if backend_str:
            ivy.set_backend(backend_str)

        _reload_variables(output_dir.replace("/", ""))
    except Exception as e:
        if DEBUG == 1:
            stop_animation.set()
            animation_thread.join()
            _set_debug_level(DEBUG)
            warnings.filters = warnings.filters[
                1:
            ]  # Restores warnings how it was originally
        raise e

    if DEBUG == 1:
        warnings.filters = warnings.filters[
            1:
        ]  # Restores warnings how it was originally
        _set_debug_level(DEBUG)

        # Stop the animation
        stop_animation.set()
        animation_thread.join()
        print(f"Transpilation of {obj_name} complete.")

    return translated_object


def transpile(
    object,
    source: str = "torch",
    target: str = "tensorflow",
    reuse_existing: bool = True,
    output_dir: str = "ivy_transpiled_outputs/",
) -> Union[MethodType, FunctionType, type]:
    """
    Converts a given object (class/function) from one framework to another.

    This function performs source-to-source translation of a given object from the source framework
    to the target framework.

    The object can be translated between two frameworks or between the Ivy IR as well
    e.g. (source="torch_frontend", target="ivy") or (source="torch_frontend", target="tensorflow") etc.

    Args:
        object: The object (class/function) to be translated.
        source (str, optional): The source framework. Defaults to 'torch'.
        target (str, optional): The target framework. Defaults to 'tensorflow'.
        reuse_existing (bool, optional): If True, the function will check if `object`
                                         already exists in the translated directory and reuse it.
                                         If False, it will re-translate `object`,
                                         even if it already exists in the directory, and overwrite
                                         the old implementation. Defaults to 'True'.
        output_dir (str, optional): The path to the directory where translated files will be saved.
                                    Defaults to 'ivy_transpiled_outputs/' in the current working directory.

    Returns:
        The translated object.
    """

    infuser_injection = "S2S OBFUSCATION"  # adds a try-except block around the translate call to obfuscate the internal source code if an error if thrown

    # only allow a transpilation when telemetry isn't included if the code is not compiled, or one of the allowed paths is within the cwd
    if __file__.endswith(".py") or any(
        [
            allow_str in os.getcwd()
            for allow_str in ["/ivy-integration-tests", "/ivy", "/tracer-transpiler"]
        ]
    ):
        return translate(
            object,
            source=source,
            target=target,
            reuse_existing=reuse_existing,
            output_dir=output_dir,
        )
    else:
        raise ivy.exceptions.IvyException(
            "The installed binaries are incompatible with Ivy."
        ) from None


def source_to_source(
    object,
    source: str = "torch",
    target: str = "tensorflow",
    reuse_existing: bool = True,
    output_dir: str = "ivy_transpiled_outputs/",
) -> Union[MethodType, FunctionType, type]:
    """
    Converts a given object (class/function) from one framework to another.

    This function performs source-to-source translation of a given object from the source framework
    to the target framework.

    The object can be translated between two frameworks or between the Ivy IR as well
    e.g. (source="torch_frontend", target="ivy") or (source="torch_frontend", target="tensorflow") etc.

    Args:
        object: The object (class/function) to be translated.
        source (str, optional): The source framework. Defaults to 'torch'.
        target (str, optional): The target framework. Defaults to 'tensorflow'.
        reuse_existing (bool, optional): If True, the function will check if `object`
                                         already exists in the translated directory and reuse it.
                                         If False, it will re-translate `object`,
                                         even if it already exists in the directory, and overwrite
                                         the old implementation. Defaults to 'True'.
        output_dir (str, optional): The path to the directory where translated files will be saved.
                                    Defaults to 'ivy_transpiled_outputs/' in the current working directory.

    Returns:
        The translated object.
    """
    return transpile(
        object,
        source=source,
        target=target,
        reuse_existing=reuse_existing,
        output_dir=output_dir,
    )
