# UTILITIES

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import List
from unittest import mock
from hypothesis import find, strategies as st, errors as hyp_errors
import numpy as np  # used by the (mock) test function
import re

# IVY

import ivy
import ivy_tests.test_ivy.conftest  # noqa
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.helpers.globals as test_globals

ivy.set_inplace_mode("strict")

DEVICES = ["cpu", "gpu:0", "tpu:0"]
BACKENDS_DIR = Path("ivy/functional/backends").resolve()
FRONTENDS_DIR = Path("ivy/functional/frontends").resolve()
BACKENDS_TESTS_DIR = Path("ivy_tests/test_ivy/test_functional").resolve()
FRONTENDS_TESTS_DIR = Path("ivy_tests/test_ivy/test_frontends").resolve()
IGNORE_FILES = ["__init__", "func_wrapper", "helpers"]
REMAP_FILES = {"linear_algebra": "test_linalg", "data_type": "test_dtype"}
NN_FILES = ["activations", "layers", "losses", "norms"]
ARRAY_OR_DTYPE_CLASSES = [
    "ivy.Array",
    "ivy.data_classes.array.array.Array",
    "ivy.NativeArray",
    "ivy.Dtype",
    "ivy.NativeDtype",
]
REGEX_DICT = {
    "function_names": re.compile(r"(?:^|\n)\s*def ([A-Za-z]\w*)\("),
    "dtype_decorators": re.compile(
        r"@(with_(?:un)?supported_(?:device_and_)?dtypes)\("
    ),
    "imported_wrappers": re.compile(
        r"from ivy\.func_wrapper import"
        r" (?:\(\s*((?:\w+,?\s*)+)\)|((?:\w+,?[^\S\r\n]*)+))\n"
    ),
}


DTYPE_CLASSES = {
    "numeric": set(ivy.all_numeric_dtypes),
    "integer": set(ivy.all_int_dtypes),
    "unsigned": set(ivy.all_uint_dtypes),
    "float": set(ivy.all_float_dtypes),
    "complex": set(ivy.all_complex_dtypes),
}


CURRENT_VERSIONS = {
    "jax": "0.4.14",
    "numpy": "1.25.2",
    "paddle": "2.5.1",
    "tensorflow": "2.13.0",
    "torch": "2.0.1",
}

DTYPE_DECORATORS = {
    "with_supported_dtypes",
    "with_unsupported_dtypes",
    "with_supported_device_and_dtypes",
    "with_unsupported_device_and_dtypes",
}
"""
TODO list (prioritised):

########## MUST HAVE ############
#. skip tests that don't use one of the functions I've mocked (many roll their own)
#. don't change the support for dtypes that I had to skip or devices I didn't test
    (requires refactoring the decorator generating/writing logic)
#. remove false positives from the backend is_dtype_err functions

########## SHOULD HAVE #######
#. get it working for frontends
#. get it working for methods
#. keep the device decorator if I didn't test that device

########## NICE TO HAVE ########
#. remove false negatives from the backend is_dtype_err functions
#. remove hard-coded version numbers
#. iterate over versions of a framework
#. prettify the output to make it easier to read
#. decide intelligently which functions to run on (e.g. based on git diffs)
#. hook into something to run automatically

########## FOR MAINTENANCE ########
#. remove hard-coded dtype classes
#. big ol' refactor and cleanup
"""


class NoTestException(Exception):
    pass


class FromFunctionException(Exception):
    def __init__(self, message, e):
        super().__init__(self, message)
        self.wrapped_error = e


class SuppressPrint:
    def __init__(self, out=True, err=False):
        self.out = out
        self.err = err

    def __enter__(self):
        if self.out:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")
        if self.err:
            self._original_stderr = sys.stderr
            sys.stderr = open(os.devnull, "w")

    def __exit__(self, *args):
        if self.out:
            sys.stdout.close()
            sys.stdout = self._original_stdout
        if self.err:
            sys.stderr.close()
            sys.stderr = self._original_stderr


def _is_dtype_err_jax(e, dtype):
    dtype_err_substrings = [
        f"does not accept dtype {dtype}. Accepted dtypes are",
        f"Unsupported dtype {dtype}",
        "Unsupported input type",
        f"does not accept dtype {dtype} at position",
        "cannot take arguments of type",
        f"must be a float dtype, got {dtype}",
        f"A.dtype={dtype} is not supported.",
        f"data type <class 'numpy.{dtype}'> not inexact",
        "data type <class 'numpy.bool_'> not inexact",
        "data type <class 'ml_dtypes.bfloat16'> not inexact",
        (
            "requires real- or complex-valued inputs (input dtype that is a sub-dtype"
            f" of np.inexact), but got {dtype}"
        ),
        f"must be a float or complex dtype, got {dtype}",
        "[[False False]] must be lesser than [[False False]]",
    ]
    only_if_complex = [
        (
            "requires real-valued outputs (output dtype that is a sub-dtype of"
            f" np.floating), but got {dtype}"
        ),
        "can't convert complex to int",
        "only real valued inputs supported for",
        "does not support complex input",
        "must be a float dtype, got complex",
    ]
    only_non_int = [
        f"must have integer or boolean type, got indexer with type {dtype}",
        "Invalid integer data type",
        "Arguments to jax.numpy.gcd must be integers",
        f"only accepts integer dtypes, got {dtype}",
        f" must have an integer type; got {dtype}",
    ]
    substrings_to_check = (
        dtype_err_substrings
        + (only_if_complex if dtype in DTYPE_CLASSES["complex"] else [])
        + (only_non_int if dtype not in DTYPE_CLASSES["integer"] else [])
    )
    return any(s in str(e) for s in substrings_to_check)


def _is_dtype_err_np(e):
    return "not supported for the input types" in str(e)


def _is_dtype_err_paddle(e):
    return "Selected wrong DataType" in str(e)


def _is_dtype_err_tf(e: ivy.exceptions.IvyException):
    return ("Value for attr 'T' of" in str(e)) or ("`features.dtype` must be" in str(e))


def _is_dtype_err_torch(e):
    return ("not implemented for" in str(e)) or ("inputs not supported for" in str(e))


is_dtype_err = {
    "jax": _is_dtype_err_jax,
    "mindspore": lambda _: False,  # TODO
    "mxnet": lambda _: False,  # TODO
    "numpy": _is_dtype_err_np,
    "onnx": lambda _: False,  # TODO
    "paddle": _is_dtype_err_paddle,
    "pandas": lambda _: False,  # TODO
    "scipy": lambda _: False,  # TODO
    "sklearn": lambda _: False,  # TODO
    "tensorflow": _is_dtype_err_tf,
    "torch": _is_dtype_err_torch,
    "xgboost": lambda _: False,  # TODO
}


def _path_to_test_path(file_path: Path):
    out = BACKENDS_TESTS_DIR
    if file_path.parent.stem == "experimental":
        out = out / "test_experimental"
    out = out / ("test_nn" if file_path.stem in NN_FILES else "test_core")
    if file_path.stem in REMAP_FILES:
        out = out / (REMAP_FILES[file_path.stem] + ".py")
    else:
        out = out / f"test_{file_path.stem}.py"
    return out


def _extract_fn_names(file_path: Path):
    with open(file_path, "r") as file:
        text = file.read()
    return REGEX_DICT["function_names"].findall(text)


def _import_module_from_path(module_path: Path, module_name="test_file"):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _type_list_to_tuple_string(dtype_list):
    return (
        "("
        + ",".join([f'"{t}"' for t in dtype_list])
        + ("," if len(dtype_list) == 1 else "")
        + ")"
    )


def _get_decorator_string(
    dtype_lists, is_supported, version_no, version_string="backend_version"
):
    out = "@with_"
    if not is_supported:
        out += "un"
    out += "supported_"
    if len(dtype_lists.keys()) > 1:
        out += "device_and_"
    out += 'dtypes({"' + version_no + ' and below": '
    if len(dtype_lists.keys()) > 1:
        out += "{"
        out += ", ".join(
            [
                f'"{k.split(":")[0]}": {_type_list_to_tuple_string(v)}'
                for k, v in dtype_lists.items()
            ]
        )
        out += "}"
    else:
        out += _type_list_to_tuple_string(list(dtype_lists.values())[0])
    out += "}, "
    out += version_string
    out += ")"
    return out


class BackendFileTester:
    def __init__(
        self,
        file_path: Path,
        devices=[],
        fn_names=[],
        verbosity=0,
        handle_unsure="supported",
        safety_mode=False,
    ):
        self.file_path = file_path
        if verbosity >= 1:
            print()
            print(self.file_path)
        for i in range(1, len(self.file_path.parents)):
            if self.file_path.parents[i].stem == "backends":
                self.backend = self.file_path.parents[i - 1].stem
                if verbosity >= 2:
                    print(f"Backend: {self.backend}")
                break
        else:
            raise Exception("No backend was identified.")

        self.test_path = _path_to_test_path(self.file_path)
        self.result = {}
        self.devices = devices or DEVICES
        self.dtypes = ()
        self.fn_names = fn_names
        self.skip_fns = set()

        self.verbosity = verbosity
        self.handle_unsure = handle_unsure
        self.safety_mode = safety_mode

        self.current_dtype = None
        self.current_fn_name = None
        self.current_device = None

        self.is_set_up = False

    def setup_test(self):
        ivy.set_backend(self.backend)

        if "gpu:0" in self.devices and not ivy.gpu_is_available():
            self.devices.remove("gpu:0")
        if "tpu:0" in self.devices and not ivy.tpu_is_available():
            self.devices.remove("tpu:0")

        self.dtypes = ivy.valid_dtypes

        discovered_fn_names = _extract_fn_names(self.file_path)
        if len(self.fn_names) == 0:
            self.fn_names = discovered_fn_names
        else:
            self.fn_names = list(set(self.fn_names) & set(discovered_fn_names))

        if not self.test_path.exists():
            if self.verbosity >= 1:
                print(f"Skipping; no file at {self.test_path}")
            raise NoTestException(f"No test file matching {self.file_path}")

        self.result = {
            fn_name: {
                device: {
                    "supported": set(),
                    "unsupported": set(),
                    "unsure": set(),
                    "skipped": set(),
                }
                for device in self.devices
            }
            for fn_name in self.fn_names
        }

        if self.verbosity >= 1:
            if self.verbosity >= 2:
                print(f"Testing with devices: {self.devices}")
                print(f"Testing with dtypes: {self.dtypes}")
                print(f"Test file path: {self.test_path}")
            print(f"Discovered {len(self.fn_names)} functions.")
            if self.verbosity >= 2:
                print("  " + ", ".join(self.fn_names))
            print()
            print()

        if self.verbosity == 1:
            print("Results: ", end="")

        self.decorators = {}

        self.is_set_up = True

    def iterate_dtypes(self):
        for dtype in self.dtypes:
            self.current_dtype = dtype
            if self.verbosity == 2:
                print()
                print(f"DType {dtype}: ", end="")
            yield dtype

    def iterate_fn_names(self):
        for fn_name in self.fn_names:
            if fn_name is None:
                if self.verbosity >= 1 and self.verbosity < 3:
                    print("X", end="")
                continue
            self.current_fn_name = fn_name
            yield fn_name

    def iterate_devices(self):
        for device in self.devices:
            self.current_device = device
            if self.verbosity == 3:
                print()
                print(
                    f"{self.file_path}::{self.current_fn_name}"
                    f"[DType={self.current_dtype},Device={self.current_device}]: ",
                    end="",
                )
            yield device

    def remove_fn(self, fn_name, reason):
        if fn_name in self.fn_names:
            i = self.fn_names.index(fn_name)
            self.fn_names[i] = None
            if self.verbosity == 3:
                print()
                print(f"{self.file_path}::{fn_name}: skipped")
            elif self.verbosity >= 1:
                print("X", end="")
            self.skip_fns.add((fn_name, reason))

    def set_result(self, result, err=None):
        if result == "unsure" and self.handle_unsure == "error":
            raise err
        value = self.current_dtype if err is None else (self.current_dtype, err)
        self.result[self.current_fn_name][self.current_device][result].add(value)
        if self.verbosity == 3:
            print(result)
        elif self.verbosity >= 1:
            char = (
                "S"
                if result == "supported"
                else (
                    "U"
                    if result == "unsupported"
                    else "X" if result == "skipped" else "?"
                )
            )
            print(char, end="")

    def resolve_unsure(self):
        self.fn_names = [func for func in self.fn_names if func is not None]

        first_skip = True
        for f in self.fn_names:
            if all(
                len(self.result[f][d]["skipped"]) == len(self.dtypes)
                for d in self.devices
            ):
                reasons = {
                    r for d in self.devices for _, r in self.result[f][d]["skipped"]
                }
                self.fn_names[self.fn_names.index(f)] = None
                self.skip_fns.add(
                    (f, "all functions skipped with reasons: " + "; ".join(reasons))
                )
                for d in self.devices:
                    del self.result[f][d]["skipped"]
                continue

            for d in self.devices:
                if len(self.result[f][d]["skipped"]) > 0 and self.verbosity >= 1:
                    if first_skip:
                        print()
                        print()
                        print("Unable to test:")
                        print()
                        first_skip = False
                    for dtype, reason in self.result[f][d]["skipped"]:
                        print(
                            f"{self.file_path}::{f}[DType={dtype},Device={d}] with"
                            f" error:\n {reason}"
                        )
                        print()
                self.result[f][d]["supported"].update(
                    {t for t, _ in self.result[f][d]["skipped"]}
                )
                del self.result[f][d]["skipped"]

        self.fn_names = [func for func in self.fn_names if func is not None]

        if self.verbosity >= 1:
            for func, reason in self.skip_fns:
                print()
                print()
                print(f"Skipped function {func}: {reason}")

        first_unsure = True
        for f in self.fn_names:
            for d in self.devices:
                if len(self.result[f][d]["unsure"]) == 0:
                    del self.result[f][d]["unsure"]

                elif self.handle_unsure in ["supported", "unsupported"]:
                    if self.verbosity >= 1:
                        for dtype, reason in self.result[f][d]["unsure"]:
                            if first_unsure:
                                print()
                                print()
                                print(f"Marking as {self.handle_unsure}:")
                                print()
                                first_unsure = False
                            print(
                                f"{self.file_path}::{f}[DType={dtype},Device={d}] with"
                                f" error:\n  {reason}"
                            )
                            print()
                    self.result[f][d][self.handle_unsure].update(
                        {t for t, _ in self.result[f][d]["unsure"]}
                    )
                    del self.result[f][d]["unsure"]

                elif self.handle_unsure == "interactive":
                    for dtype, reason in self.result[f][d]["unsure"]:
                        print(
                            f"{self.file_path}::{f}[DType={dtype},Device={d}] threw"
                            f" error {reason}"
                        )
                        mark_as = "?"
                        while mark_as not in ["S", "U"]:
                            raw = input(
                                "Please mark as (S)upported or (U)nsupported (or"
                                " (E)xit): "
                            )
                            if raw == "":
                                continue
                            mark_as = raw[0].upper()
                            if mark_as == "E":
                                raise SystemExit()
                        mark_as = "supported" if mark_as == "S" else "unsupported"
                        self.result[f][d][mark_as].add(dtype)
                    del self.result[f][d]["unsure"]

    def print_result(self):
        if self.verbosity < 2:
            return
        print()
        print(f"Printing results for {self.file_path}:\n")
        for f in self.fn_names:
            for d in self.devices:
                print(f, d)
                print("Supported:", self.result[f][d]["supported"])
                print("Unsupported:", self.result[f][d]["unsupported"])
                print("\n")

    def create_decorators(self):
        if self.verbosity >= 2:
            print()
            print("Generating decorators:")
        for f in self.fn_names:
            device_and_dtypes = {}
            for d in self.devices:
                supported = self.result[f][d]["supported"]
                unsupported = self.result[f][d]["unsupported"]

                # reduce to classes where possible
                for cls, types in DTYPE_CLASSES.items():
                    if types.issubset(supported):
                        supported = supported - types
                        supported.add(cls)
                    elif types.issubset(unsupported):
                        unsupported = unsupported - types
                        unsupported.add(cls)

                device_and_dtypes[d] = (supported, unsupported)
            total_supported = sum(len(device_and_dtypes[d][0]) for d in self.devices)
            total_unsupported = sum(len(device_and_dtypes[d][1]) for d in self.devices)
            use_supported = total_supported <= total_unsupported
            if total_unsupported == 0:
                self.decorators[f] = None
            else:
                for d in self.devices:
                    device_and_dtypes[d] = device_and_dtypes[d][
                        0 if use_supported else 1
                    ]
                self.decorators[f] = _get_decorator_string(
                    device_and_dtypes, use_supported, CURRENT_VERSIONS[self.backend]
                )
            if self.verbosity >= 2:
                print(f"  {f}: {self.decorators[f]}")
        if self.verbosity >= 2:
            print()

    def write_decorators(self):
        # read in the code file
        with open(self.file_path, "r") as f:
            code_text = f.readlines()

        # find each function in it
        for func in self.fn_names:
            for i, v in enumerate(code_text):
                if f"def {func}(" in v:
                    def_index = i
                    break
            if self.verbosity >= 3:
                print(f"{func} found at line no. {def_index+1}")

            # read all the lines that could be decorators (i.e. they're immediately
            # above the `def` line)
            decorator_start_idx = def_index
            decorator_lines = []
            while (
                decorator_start_idx >= 1
                and code_text[decorator_start_idx - 1].strip() != ""
            ):
                decorator_lines.insert(0, code_text[decorator_start_idx - 1])
                decorator_start_idx -= 1

            # find any dtype decorators in those lines
            indices_to_remove = []
            start_of_decorator = None
            dec_string = ""
            for i in range(len(decorator_lines)):
                line = decorator_lines[i]

                # check for the start of a dtype decorator
                if start_of_decorator is None:
                    if REGEX_DICT["dtype_decorators"].search(line):
                        start_of_decorator = decorator_start_idx + i

                # if we're in a decorator, check if we've seen the end of it yet, and
                # if we have mark the whole segment for deletion
                if start_of_decorator is not None:
                    dec_string += line
                    if dec_string.count("(") == dec_string.count(")"):
                        indices_to_remove.append(
                            (start_of_decorator, decorator_start_idx + i + 1)
                        )
                        start_of_decorator = None
                        dec_string = ""

            # remove dtype decorators that already exist
            for i, j in indices_to_remove:
                if self.verbosity >= 2:
                    print(f"Removing existing decorator from {func}")
                del code_text[i:j]
                def_index -= j - i

            # add the new decorator
            if self.decorators[func] is not None:
                if self.verbosity >= 2:
                    print(f"Adding decorator to {func}")
                code_text.insert(def_index, self.decorators[func] + "\n")

        # combine the lines into a single string
        code_text = "".join(code_text)

        # get all the decorators that are used in it
        all_decorators = set(REGEX_DICT["dtype_decorators"].findall(code_text))

        # get all the decorators that are imported
        matches = [
            g1 + "," + g2
            for g1, g2 in REGEX_DICT["imported_wrappers"].findall(code_text)
        ]
        imported_decorators = {
            item.strip() for match in matches for item in match.split(",")
        }
        imported_decorators.discard("")

        # just in case any other functions are imported
        desired_decorators = (imported_decorators - DTYPE_DECORATORS) | all_decorators

        # write the import statement to the file
        if self.verbosity >= 2:
            print()
            print(f"Previously imported wrappers: {list(imported_decorators)}")
            print(f"Updating to import {desired_decorators}")

        if desired_decorators != imported_decorators:
            if len(desired_decorators) == 0:
                code_text = REGEX_DICT["imported_wrappers"].sub("", code_text)
                code_text = code_text.replace("from . import backend_version\n", "", 1)
            else:
                decorator_import_string = (
                    "from ivy.func_wrapper import"
                    f" {', '.join(list(desired_decorators))}\n"
                )
                if len(imported_decorators) == 0:
                    code_text = code_text.replace(
                        "import ivy\n",
                        "import ivy\n"
                        + decorator_import_string
                        + "from . import backend_version\n",
                        1,
                    )
                else:
                    code_text = REGEX_DICT["imported_wrappers"].sub(
                        decorator_import_string,
                        code_text,
                    )

        if self.safety_mode:
            print("File write suppressed due to safety mode.")
        else:
            with open(self.file_path, "w") as f:
                f.write(code_text)


def _kwargs_to_args_n_kwargs(num_positional_args, kwargs):
    args = [v for v in list(kwargs.values())[:num_positional_args]]
    kwargs = {k: kwargs[k] for k in list(kwargs.keys())[num_positional_args:]}
    return args, kwargs


def _get_nested_np_arrays(nest):
    indices = ivy.nested_argwhere(nest, lambda x: isinstance(x, np.ndarray))
    ret = ivy.multi_index_nest(nest, indices)
    return ret, indices, len(ret)


def mock_test_function(
    *,
    input_dtypes,
    test_flags,
    fn_name,
    rtol_=None,
    atol_=1e-06,
    tolerance_dict=None,
    test_values=True,
    xs_grad_idxs=None,
    ret_grad_idxs=None,
    backend_to_test,
    on_device,
    return_flat_np_arrays=False,
    **all_as_kwargs_np,
):
    # split the arguments into their positional and keyword components
    args_np, kwargs_np = _kwargs_to_args_n_kwargs(
        num_positional_args=test_flags.num_positional_args, kwargs=all_as_kwargs_np
    )

    # Extract all arrays from the arguments and keyword arguments
    arg_np_arrays, arrays_args_indices, n_args_arrays = _get_nested_np_arrays(args_np)
    kwarg_np_arrays, arrays_kwargs_indices, n_kwargs_arrays = _get_nested_np_arrays(
        kwargs_np
    )

    # Make all array-specific test flags and dtypes equal in length
    total_num_arrays = n_args_arrays + n_kwargs_arrays
    if len(input_dtypes) < total_num_arrays:
        input_dtypes = [input_dtypes[0] for _ in range(total_num_arrays)]
    if len(test_flags.as_variable) < total_num_arrays:
        test_flags.as_variable = [
            test_flags.as_variable[0] for _ in range(total_num_arrays)
        ]
    if len(test_flags.native_arrays) < total_num_arrays:
        test_flags.native_arrays = [
            test_flags.native_arrays[0] for _ in range(total_num_arrays)
        ]
    if len(test_flags.container) < total_num_arrays:
        test_flags.container = [
            test_flags.container[0] for _ in range(total_num_arrays)
        ]

    with helpers.BackendHandler.update_backend(backend_to_test) as ivy_backend:
        # Update variable flags to be compatible with float dtype and with_out args
        test_flags.as_variable = [
            v if ivy_backend.is_float_dtype(d) else False
            for v, d in zip(test_flags.as_variable, input_dtypes)
        ]

        # create args
        args = ivy_backend.copy_nest(args_np, to_mutable=False)
        ivy_backend.set_nest_at_indices(
            args,
            arrays_args_indices,
            test_flags.apply_flags(
                arg_np_arrays,
                input_dtypes,
                0,
                backend=backend_to_test,
                on_device=on_device,
            ),
        )

        # create kwargs
        kwargs = ivy_backend.copy_nest(kwargs_np, to_mutable=False)
        ivy_backend.set_nest_at_indices(
            kwargs,
            arrays_kwargs_indices,
            test_flags.apply_flags(
                kwarg_np_arrays,
                input_dtypes,
                len(arg_np_arrays),
                backend=backend_to_test,
                on_device=on_device,
            ),
        )

        # Ignore the instance method flag and run it directly from the API
        target_fn = ivy_backend.__dict__[fn_name]

        try:
            target_fn(*args, **kwargs)
        except Exception as e:
            raise FromFunctionException(str(e), e)


def run_dtype_setter(
    files_list: List[Path],
    devices=DEVICES,
    fn_names=[],
    options={"verbosity": 0, "handle_unsure": "supported", "safety_mode": False},
):
    helpers.test_function = mock.Mock(wraps=mock_test_function)
    sys.modules["ivy_tests.test_ivy.helpers"] = helpers

    test_globals._set_backend("tensorflow")

    if options["verbosity"] >= 1:
        print("============== RUNNING =============")

    for file in files_list:
        test_handler = BackendFileTester(file, devices, fn_names, **options)
        try:
            test_handler.setup_test()
        except NoTestException:
            continue

        for dtype in test_handler.iterate_dtypes():
            helpers.get_dtypes = mock.Mock(return_value=st.just([dtype]))
            sys.modules["ivy_tests.test_ivy.helpers"] = helpers

            test_file = _import_module_from_path(test_handler.test_path)

            for fn_name in test_handler.iterate_fn_names():
                test_fn = f"test_{fn_name}"

                # we only need to check these once for each function
                if dtype == test_handler.dtypes[0]:
                    if test_fn not in test_file.__dict__:
                        test_handler.remove_fn(fn_name, "no test function")
                        continue
                    if "hypothesis" not in test_file.__dict__[test_fn].__dict__:
                        test_handler.remove_fn(fn_name, "test does not use hypothesis")
                        continue

                    # Ideally we'd be doing this with something like
                    # typing.get_type_hints or inspect.signature but there are weird
                    # edge cases that those don't work for (e.g. ivy.asarray breaks
                    # typing.get_type_hints)
                    if not any(
                        t in str(p)
                        for t in ARRAY_OR_DTYPE_CLASSES
                        for p in ivy.__dict__[fn_name].__annotations__.values()
                    ):
                        test_handler.remove_fn(
                            fn_name, "function does not take array or dtype inputs"
                        )
                        continue

                kwargs = (
                    test_file.__dict__[test_fn].__dict__["hypothesis"]._given_kwargs
                )
                min_example = {}
                try:
                    with SuppressPrint():
                        for k, v in kwargs.items():
                            min_example[k] = find(specifier=v, condition=lambda _: True)
                except Exception as e:
                    for device in test_handler.iterate_devices():
                        test_handler.set_result("skipped", "In hypothesis: " + str(e))
                    continue

                for device in test_handler.iterate_devices():
                    try:
                        with SuppressPrint():
                            test_file.__dict__[test_fn].original(
                                **min_example,
                                backend_fw=test_handler.backend,
                                on_device=device,
                            )
                        test_handler.set_result("supported")
                    except FromFunctionException as ffe:
                        e = ffe.wrapped_error
                        if is_dtype_err[test_handler.backend](e, dtype):
                            test_handler.set_result("unsupported")
                        else:
                            test_handler.set_result("unsure", str(e))
                    except hyp_errors.UnsatisfiedAssumption:
                        test_handler.set_result("skipped", "Unsatisfied assumption")
                    except Exception as e:
                        test_handler.set_result(
                            "skipped", "In test function: " + str(e)
                        )

        test_handler.resolve_unsure()
        test_handler.print_result()
        test_handler.create_decorators()
        test_handler.write_decorators()
    test_globals._unset_backend()


def _is_same_or_child(a: Path, b: Path):
    return a.samefile(b) or b in a.parents


def main():
    parser = argparse.ArgumentParser(
        "DType Setter",
        description=(
            "Automatically identifies and sets (un)supported dtypes for a given set of"
            " functions."
        ),
    )
    parser.add_argument(
        "PATH",
        nargs="*",
        type=Path,
        default=["."],
        help=(
            "path(s) to the files and/or directories containing the functions to add"
            " dtype decorators to. Note that decorators are added to frontend and"
            " backend functions, not to ivy functions or to test functions."
        ),
    )
    parser.add_argument(
        "-f",
        "--functions",
        nargs="+",
        default=[],
        help=(
            "specify functions to check. Discovered functions that don't match these"
            " names will be skipped."
        ),
    )
    parser.add_argument(
        "-d",
        "--devices",
        nargs="+",
        choices=["cpu", "gpu", "tpu"],
        default=[],
        help="specify devices to check. Others will be skipped.",
    )
    parser.add_argument(
        "-u",
        "--unsure",
        choices=["supported", "unsupported", "interactive", "error"],
        default="supported",
        help=(
            "specify how to classify unknown error messages. Can be"
            " supported/unsupported, interactive (you will be shown the error message"
            " and asked to decide), or error (an error will be raised when an unknown"
            " message is found)."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbosity",
        action="count",
        default=0,
        help="set the verbosity level.",
    )
    parser.add_argument(
        "--safety-mode",
        dest="safety_mode",
        action="store_true",
        help="activate a debug mode which prevents actually writing to files.",
    )
    args = parser.parse_args()

    if args.safety_mode:
        print("Safety mode on.")

    if args.verbosity >= 1:
        print("========= COLLECTING FILES =========")

    reduced_paths: List[Path] = []

    for p in args.PATH:
        assert p.exists()
        p = p.resolve()
        if p in BACKENDS_DIR.parents:
            reduced_paths = [BACKENDS_DIR, FRONTENDS_DIR]
            break

        assert _is_same_or_child(p, BACKENDS_DIR) or _is_same_or_child(p, FRONTENDS_DIR)
        reduced_paths = [q for q in reduced_paths if p not in q.parents]
        if not any(q.samefile(p) or q in p.parents for q in reduced_paths):
            reduced_paths.append(p)

    files_list = []
    for p in reduced_paths:
        if p.is_dir():
            files = p.rglob("*.py")
            files = [f for f in files if f.stem not in IGNORE_FILES]
            files_list.extend(files)
        if p.is_file():
            files_list.append(p)

    if args.verbosity >= 1:
        print(f"Discovered {len(files_list)} files.")
        if args.verbosity >= 3:
            print("  " + "\n  ".join([str(file) for file in files_list]))

    if args.devices is not None:
        devices = [d + ":0" if d in ["gpu", "tpu"] else d for d in args.devices]
    else:
        devices = DEVICES

    run_dtype_setter(
        files_list,
        devices,
        args.functions,
        {
            "verbosity": args.verbosity,
            "handle_unsure": args.unsure,
            "safety_mode": args.safety_mode,
        },
    )


if __name__ == "__main__":
    main()
