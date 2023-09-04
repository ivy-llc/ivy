# UTILITIES

import argparse
import importlib
import inspect
import sys
from pathlib import Path
from typing import List
from unittest import mock
from hypothesis import find
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
NN_FILES = ["activations", "layers", "losses", "norms"]


# TODO: don't hardcode these
DTYPE_CLASSES = [
    ("numeric", set(ivy.all_numeric_dtypes)),
    ("integer", set(ivy.all_int_dtypes)),
    ("unsigned", set(ivy.all_uint_dtypes)),
    ("float", set(ivy.all_float_dtypes)),
    ("complex", set(ivy.all_complex_dtypes)),
]


# TODO: don't hardcode these
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


# TODO list:
# - decide how to deal with uncertain cases (interactive mode? just write it and ask the user to check? flag to switch between these? send a warning?)  # noqa
# - make sure I don't overgeneralise (maybe gpu doesn't seem usable, but that's just because of the machine I'm on)  # noqa
# - related to that, make sure that if there's an existing dtype decorator referencing a version/device I didn't test, make sure I include that data when I write the new version (maybe I could read that in before I "compile" the decorator and use it to modify the data structure)  # noqa
# - add new function types: method, frontend, frontend method (for now, I assume its a backend function)  # noqa
# - add command line args (to specify certains folders/files/functions/backends/devices/versions/types) (partially done)  # noqa
# - add ability to iterate over versions of a framework (talk to multiversion support people)  # noqa
# - hook the script into *something* so it runs automatically
# - have some smart way of deciding which functions to run on (maybe a flag that tells it to look at the most recent `git diff`s and only check the functions that have been changed) # noqa
# - fix other TODO comments left in the file


# TODO: make sure these functions are absolutely right; convert them to regex and give them awareness of the input dtype if need be  # noqa
def is_dtype_err_jax(e):
    return "does not accept dtype" in str(e)


def is_dtype_err_np(e):
    return "not supported for the input types" in str(e)


def is_dtype_err_paddle(e):
    return "Selected wrong DataType" in str(e)


def is_dtype_err_tf(e: ivy.exceptions.IvyException):
    return ("Value for attr 'T' of" in str(e)) or ("`features.dtype` must be" in str(e))


def is_dtype_err_torch(e):
    return ("not implemented for" in str(e)) or ("inputs not supported for" in str(e))


is_dtype_err = {
    "jax": is_dtype_err_jax,
    "mindspore": lambda _: False,  # TODO
    "mxnet": lambda _: False,  # TODO
    "numpy": is_dtype_err_np,
    "onnx": lambda _: False,  # TODO
    "paddle": is_dtype_err_paddle,
    "pandas": lambda _: False,  # TODO
    "scipy": lambda _: False,  # TODO
    "sklearn": lambda _: False,  # TODO
    "tensorflow": is_dtype_err_tf,
    "torch": is_dtype_err_torch,
    "xgboost": lambda _: False,  # TODO
}


def _path_to_test_path(file_path: Path):
    out = BACKENDS_TESTS_DIR
    if file_path.parent.stem == "experimental":
        out = out / "test_experimental"
    out = out / ("test_nn" if file_path.stem in NN_FILES else "test_core")
    out = out / f"test_{file_path.stem}.py"
    return out


def _extract_fn_names(file_path: Path):
    fn_name_regex = r"def ([A-Za-z]\w*)\("
    with open(file_path, "r") as file:
        text = file.read()
    return re.findall(fn_name_regex, text)


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


# TODO: clean this up, check whether removing empty tuples is needed
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
    def __init__(self, file_path: Path, devices=[], fn_names=[]):
        self.file_path = file_path
        for i in range(1, len(self.file_path.parents)):
            if self.file_path.parents[i].stem == "backends":
                self.backend = self.file_path.parents[i - 1].stem
                break
        else:
            raise Exception("No backend was identified.")

        self.test_path = _path_to_test_path(self.file_path)
        self.result = {}
        self.devices = devices or DEVICES
        self.dtypes = ()
        self.fn_names = fn_names
        self.skip_fns = set()

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
            raise Exception(f"No test file matching {self.file_path}")

        self.result = {
            fn_name: {
                device: {"supported": set(), "unsupported": set(), "unsure": set()}
                for device in self.devices
            }
            for fn_name in self.fn_names
        }

        self.decorators = {}

        self.is_set_up = True

    def iterate_dtypes(self):
        for dtype in self.dtypes:
            self.current_dtype = dtype
            yield dtype

    def iterate_fn_names(self):
        for fn_name in self.fn_names:
            if fn_name in self.skip_fns:
                continue
            self.current_fn_name = fn_name
            yield fn_name
        for f in self.skip_fns:
            self.fn_names.remove(f)
        self.skip_fns = set()

    def iterate_devices(self):
        for device in self.devices:
            self.current_device = device
            yield device

    def remove_fn(self, fn_name):
        if fn_name in self.fn_names:
            self.skip_fns.add(fn_name)

    def set_result(self, result, err=None):
        value = self.current_dtype if err is None else (self.current_dtype, err)
        self.result[self.current_fn_name][self.current_device][result].add(value)

    def print_result(self):
        print(f"Printing results for {self.file_path}:\n")
        for f in self.fn_names:
            for d in self.devices:
                print(f, d)
                print("Supported:", self.result[f][d]["supported"])
                print("Unsupported:", self.result[f][d]["unsupported"])
                print("Unsure:", self.result[f][d]["unsure"])
                print("\n")

    def create_decorators(self):
        for f in self.fn_names:
            device_and_dtypes = {}
            for d in self.devices:
                # assume remaining "unsure" should be marked as supported
                supported = self.result[f][d]["supported"] | {
                    t for t, _ in self.result[f][d]["unsure"]
                }
                unsupported = self.result[f][d]["unsupported"]

                # reduce to classes where possible
                for cls, types in DTYPE_CLASSES:
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
            for d in self.devices:
                device_and_dtypes[d] = device_and_dtypes[d][0 if use_supported else 1]
            self.decorators[f] = _get_decorator_string(
                device_and_dtypes, use_supported, CURRENT_VERSIONS[self.backend]
            )
            print(f, self.decorators[f])

    def write_decorators(self):
        is_dtype_dec_regex = r"@with_(un)?supported_(device_and_)?dtypes\("
        get_dtype_decs_regex = r"@(with_(?:un)?supported_(?:device_and_)?dtypes)\("
        get_imported_decs_regex = (
            r"from ivy\.func_wrapper import"
            r" (?:\(\s*((?:\w+,?\s*)+)\)|((?:\w+,?[^\S\r\n]*)+))\n"
        )
        replace_imported_decs_regex = (
            r"from ivy\.func_wrapper import"
            r" (?:\(\s*(?:\w+,?\s*)+\)|(?:\w+,?[^\S\r\n]*)+)\n"
        )

        # read in the code file
        with open(self.file_path, "r") as f:
            code_text = f.readlines()
        print(code_text[15])

        # find each function in it
        for func in self.fn_names:
            for i, v in enumerate(code_text):
                if f"def {func}(" in v:
                    def_index = i
                    break
            else:
                print(
                    f"WARNING: no definition of {func} found in {self.file_path}"
                )  # TODO: make it an actual warning

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
                    if re.search(is_dtype_dec_regex, line):
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
                del code_text[i:j]
                def_index -= j - i

            # add the new decorator
            code_text.insert(def_index, self.decorators[func] + "\n")

        # combine the lines into a single string
        code_text = "".join(code_text)

        # get all the decorators that are used in it
        all_decorators = set(re.findall(get_dtype_decs_regex, code_text))

        # get all the decorators that are imported
        matches = [
            g1 + "," + g2 for g1, g2 in re.findall(get_imported_decs_regex, code_text)
        ]
        imported_decorators = {
            item.strip() for match in matches for item in match.split(",")
        }
        imported_decorators.discard("")

        # just in case any other functions are imported
        desired_decorators = list(
            (imported_decorators - DTYPE_DECORATORS) | all_decorators
        )

        # write the import statement to the file
        code_text = re.sub(
            replace_imported_decs_regex,
            f"from ivy.func_wrapper import {', '.join(desired_decorators)}\n",
            code_text,
        )

        with open(self.file_path, "w") as f:
            f.write(code_text)


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
    args_np, kwargs_np = helpers.kwargs_to_args_n_kwargs(
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
            v if ivy_backend.is_float_dtype(d) and not test_flags.with_out else False
            for v, d in zip(test_flags.as_variable, input_dtypes)
        ]

    # update instance_method flag to only be considered if the
    # first term is either an ivy.Array or ivy.Container
    instance_method = test_flags.instance_method and (
        not test_flags.native_arrays[0] or test_flags.container[0]
    )

    args, kwargs = helpers.create_args_kwargs(
        backend=backend_to_test,
        args_np=args_np,
        arg_np_vals=arg_np_arrays,
        args_idxs=arrays_args_indices,
        kwargs_np=kwargs_np,
        kwarg_np_vals=kwarg_np_arrays,
        kwargs_idxs=arrays_kwargs_indices,
        input_dtypes=input_dtypes,
        test_flags=test_flags,
        on_device=on_device,
    )

    with helpers.BackendHandler.update_backend(backend_to_test) as ivy_backend:
        # If function doesn't have an out argument but an out argument is given
        # or a test with out flag is True
        if ("out" in kwargs or test_flags.with_out) and "out" not in inspect.signature(
            getattr(ivy_backend, fn_name)
        ).parameters:
            raise Exception(f"Function {fn_name} does not have an out parameter")

        # Run either as an instance method or from the API directly
        instance = None
        if instance_method:
            array_or_container_mask = [
                (not native_flag) or container_flag
                for native_flag, container_flag in zip(
                    test_flags.native_arrays, test_flags.container
                )
            ]

            # Boolean mask for args and kwargs True if an entry's
            # test Array flag is True or test Container flag is true
            args_instance_mask = array_or_container_mask[
                : test_flags.num_positional_args
            ]
            kwargs_instance_mask = array_or_container_mask[
                test_flags.num_positional_args :
            ]

            if any(args_instance_mask):
                instance, args = helpers._find_instance_in_args(
                    backend_to_test, args, arrays_args_indices, args_instance_mask
                )
            else:
                instance, kwargs = helpers._find_instance_in_args(
                    backend_to_test, kwargs, arrays_kwargs_indices, kwargs_instance_mask
                )

            if test_flags.test_compile:
                target_fn = lambda instance, *args, **kwargs: instance.__getattribute__(
                    fn_name
                )(*args, **kwargs)
                args = [instance, *args]
            else:
                target_fn = instance.__getattribute__(fn_name)
        else:
            target_fn = ivy_backend.__dict__[fn_name]

        helpers.get_ret_and_flattened_np_array(
            backend_to_test,
            target_fn,
            *args,
            test_compile=test_flags.test_compile,
            precision_mode=test_flags.precision_mode,
            **kwargs,
        )


def run_dtype_setter(files_list: List[Path], devices=DEVICES, fn_names=[]):
    helpers.test_function = mock.Mock(wraps=mock_test_function)
    sys.modules["ivy_tests.test_ivy.helpers"] = helpers

    test_globals._set_backend("tensorflow")

    for file in files_list:
        test_handler = BackendFileTester(file, devices, fn_names)
        test_handler.setup_test()

        for dtype in test_handler.iterate_dtypes():
            helpers.get_dtypes = mock.Mock(return_value=[dtype])
            sys.modules["ivy_tests.test_ivy.helpers"] = helpers

            test_file = _import_module_from_path(test_handler.test_path)

            for fn_name in test_handler.iterate_fn_names():
                test_fn = f"test_{fn_name}"
                if test_fn not in test_file.__dict__:
                    # TODO: turn this into a real warning
                    print(f"Warning: no {test_fn} to match {fn_name} in {file}")
                    test_handler.remove_fn(fn_name)
                    continue

                kwargs = (
                    test_file.__dict__[test_fn].__dict__["hypothesis"]._given_kwargs
                )
                min_example = {
                    k: find(specifier=v, condition=lambda _: True)
                    for k, v in kwargs.items()
                }

                for device in test_handler.iterate_devices():
                    try:
                        test_file.__dict__[test_fn].original(
                            **min_example,
                            backend_fw=test_handler.backend,
                            on_device=device,
                        )
                        test_handler.set_result("supported")
                    except Exception as e:
                        if is_dtype_err[test_handler.backend](e):
                            test_handler.set_result("unsupported")
                        else:
                            test_handler.set_result("unsure", e)
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
    args = parser.parse_args()

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

    if args.devices is not None:
        devices = [d + ":0" if d in ["gpu", "tpu"] else d for d in args.devices]
    else:
        devices = DEVICES

    run_dtype_setter(files_list, devices, args.functions)


if __name__ == "__main__":
    main()
