# UTILITIES

import argparse
import importlib
from importlib.metadata import version
import os
import sys
from pathlib import Path
from typing import List
from unittest import mock
from hypothesis import find, strategies as st, errors as hyp_errors
import numpy as np  # used by the (mock) test function
import re
from abc import ABC, abstractmethod

# used to implement a timeout on hypothesis.find
import threading
import _thread

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


PYPI_NAMES = {
    "jax": "jax",
    "numpy": "numpy",
    "paddle": "paddlepaddle",
    "tensorflow": "tensorflow",
    "torch": "torch",
}


DTYPE_DECORATORS = {
    "with_supported_dtypes",
    "with_unsupported_dtypes",
    "with_supported_device_and_dtypes",
    "with_unsupported_device_and_dtypes",
}


class NoTestException(Exception):
    pass


class NoTestFunctionException(Exception):
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


def _remove_numbers(dtype, bfloat_to_float=False):
    if dtype == "bfloat16" and bfloat_to_float:
        return "float"
    return "".join(c for c in dtype if not c.isnumeric())


# TODO: check lax.conv et al. (gets given ivy arrays instead of jax arrays)
def _is_dtype_err_jax(e, dtype):
    short_type = (
        "PRED"
        if dtype == "bool"
        else dtype[0].upper() + "".join(c for c in dtype if c.isnumeric())
    )
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
        (
            "JAX encountered invalid PRNG key data: expected key_data.dtype = uint32;"
            f" got dtype={dtype}"
        ),
        "[[False False]] must be lesser than [[False False]]",
        f"must be inexact; got {dtype}",
        "Scanned function carry input and carry output must have equal types",
        "output and input must have identical types, got",
        f"unimplemented for {short_type}",
        f"pad operand and padding_value must be same dtype: got {dtype} and",
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
        "does not support complex-valued inputs",
        "not well defined for complex",
        "is not compatible with complex inputs.",
    ]
    only_non_int = [
        f"must have integer or boolean type, got indexer with type {dtype}",
        "Invalid integer data type",
        "Arguments to jax.numpy.gcd must be integers",
        "Arguments to jax.numpy.lcm must be integers.",
        f"only accepts integer dtypes, got {dtype}",
        f" must have an integer type; got {dtype}",
        "concrete values of integer type",
        f"must be an integer type, but got {dtype}",
    ]
    substrings_to_check = (
        (only_if_complex if dtype in DTYPE_CLASSES["complex"] else [])
        + (only_non_int if dtype not in DTYPE_CLASSES["integer"] else [])
        + dtype_err_substrings
    )
    return any(s in str(e) for s in substrings_to_check)


MAP_HIGHER = {
    "float16": "float64",
    "float32": "float64",
    "complex64": "complex128",
}


def _is_dtype_err_np(e, dtype):
    dtype_err_substrings = [
        f"to dtype('{dtype}') with casting rule 'same_kind'",
        f"Cannot cast array data from dtype('{dtype}') to dtype",
        f"array type {dtype} is unsupported in linalg",
        f"data type <class 'numpy.{dtype}'> not inexact",
        "Invalid integer data type",
        "arrays used as indices must be of integer (or boolean) type",
        f"'numpy.{dtype}' object cannot be interpreted as an integer",
        (
            "not supported for the input types, and the inputs could not be safely"
            " coerced to any supported types according to the casting rule ''safe''"
        ),
        (
            "did not contain a loop with signature matching types <class"
            f" 'numpy.dtypes.{dtype.capitalize()}DType'> ->"
        ),
        (
            "did not contain a loop with signature matching types (<class"
            f" 'numpy.dtypes.{dtype.capitalize()}DType'>,"
        ),
        f"Unsupported dtype dtype('{dtype}')",
        f"cannot take arguments of type {_remove_numbers(dtype)}",
        f"to dtype('{dtype}') according to the rule 'safe'",
        f"Cannot cast scalar from dtype('{dtype}') to dtype",
        "a must be an array of real numbers",
        "must be real number, not complex",
        "i0 not supported for complex values",
        (
            "dtype of `to_begin` must be compatible with input `ary` under the"
            " `same_kind` rule."
        ),
        f"This operation is not supported for {dtype} values",
        "only int indices permitted",
    ] + (
        [
            f"Cannot cast array data from dtype('{MAP_HIGHER[dtype]}') to dtype",
            f"'numpy.{MAP_HIGHER[dtype]}' object cannot be interpreted as an integer",
            f"Cannot cast scalar from dtype('{MAP_HIGHER[dtype]}') to dtype",
        ]
        if dtype in MAP_HIGHER
        else []
    )
    only_if_bool = [
        (
            "The numpy boolean negative, the `-` operator, is not supported, use the"
            " `~` operator or the logical_not function instead."
        ),
        (
            "numpy boolean subtract, the `-` operator, is not supported, use the"
            " bitwise_xor, the `^` operator, or the logical_xor function instead."
        ),
        "data type <class 'numpy.bool_'> not inexact",
        "[[False False]] must be lesser than [[False False]]",
        "math domain error",
        "dirichlet: alpha <= 0",
        "beta: a <= 0",
    ]
    substrings_to_check = (
        only_if_bool if dtype == "bool" else []
    ) + dtype_err_substrings
    return any(s in str(e) for s in substrings_to_check)


def _is_dtype_err_paddle(e, dtype):
    dtype_err_substrings = [
        f"Selected wrong DataType `{dtype}`. Paddle support following DataTypes:",
        f"(Unimplemented) Data type ({dtype}) is not supported when casting data type.",
        f"'{dtype}'",
        (
            f"The type of data we are trying to retrieve ({dtype}) does not match the"
            " type of data"
        ),
        (
            f"does not match the type of data ({dtype}) currently contained in the"
            " container."
        ),
        f"data type <class 'numpy.{dtype}'> not inexact",
        "Invalid integer data type",
        f"must be list of int, but got {_remove_numbers(dtype)}",
        f"randint cannot take arguments of type {_remove_numbers(dtype, True)}",
        f"as_ivy_dtype: dtype({dtype})",
        "(NotFound) There are no kernels which are registered in the kron operator.",
        f"must be int, but got {_remove_numbers(dtype, True)}",
        "Elements in Input(x) should all be integers",
        f"Invalid dtype 'paddle.{dtype}'. Valid dtypes are",
    ]
    only_int32 = (
        ["Selected wrong DataType `int64`. Paddle support following DataTypes:"]
        if dtype == "int32"
        else []
    )
    only_complex128 = (
        ["Selected wrong DataType `complex64`. Paddle support following DataTypes:"]
        if dtype == "complex128"
        else []
    )
    only_complex = (
        ["The filled value is out of range for target type"]
        if dtype in DTYPE_CLASSES["complex"]
        else []
    )
    only_bool = (
        [
            (
                "Tensor(shape=[1, 2], dtype=bool, place=Place(cpu),"
                " stop_gradient=True,\n       [[False, False]]) must be lesser than"
                " Tensor(shape=[1, 2], dtype=bool, place=Place(cpu),"
                " stop_gradient=True,\n       [[False, False]])"
            ),
            "data type <class 'numpy.bool_'> not inexact",
        ]
        if dtype == "bool"
        else []
    )
    substrings_to_check = (
        dtype_err_substrings + only_int32 + only_complex128 + only_complex + only_bool
    )
    return any(s in str(e) for s in substrings_to_check)


# These aren't equivalences, per se, they're just other dtypes tf might complain about
# when the actual dtype is the one on the left. complex64: complex128 is because it
# sometimes upcasts first because of a multiplication with a float64
TF_DTYPES_MAP = {
    "float16": "half",
    "float32": "float",
    "float64": "double",
    "complex64": "complex128",
}


def _is_dtype_err_tf(e, dtype):
    dtype_err_substrings = [
        (
            "`features.dtype` must be a floating point"
            f" tensor.Received:features.dtype=<dtype: '{dtype}'>"
        ),
        "Could not find device for node: ",
        f" to EagerTensor of dtype {dtype}",
        f"Input `a` must have `float`-like `dtype` (saw {dtype}).",
        f"data type <class 'numpy.{dtype}'> not inexact",
        "data type <class 'numpy.bool_'> not inexact",
        "data type <class 'ml_dtypes.bfloat16'> not inexact",
        "Input must be either real or complex. Received ",
        f"Argument `dtype` got invalid value <dtype: '{dtype}'>. Accepted dtypes are",
        f"dtype must be a floating point type. Found <dtype: '{dtype}'>",
        "Arguments to gcd must be integers.",
        "randint cannot take arguments of type ",
        "Invalid integer data type ",
        f"does not support matrices of type <dtype: '{dtype}'>",
        "RFFT requires tf.float32 or tf.float64 inputs, got:",
        f"`x` must be of a real dtype. Received: <dtype: '{dtype}'>",
        f"`x` must be of a complex dtype. Received: <dtype: '{dtype}'>",
        f" midpoint interpolation not allowed with dtype <dtype: '{dtype}'>",
        f"Expected numeric or variant tensor, got dtype tf.{dtype}.",
        f"The input data type is not supported, DataType : {dtype}",
        "Argument `concentration` must be float type.",
        "Argument `probs` must having floating type.",
        f"Tensor conversion requested dtype int32 for Tensor with dtype {dtype}",
        f"Invalid dtype tf.{dtype} in",
        "Expected x to be numeric,",
        f"was expected to be a bool tensor but is a {dtype} tensor",
        f"was expected to be a {dtype} tensor but is a ",
        "`x.dtype` must be a floating point tensor",
        "AttrValue has invalid DataType",
        "`x` and `y` must have the same dtype, got",
    ] + (
        [
            f"was expected to be a bool tensor but is a {TF_DTYPES_MAP[dtype]} tensor",
        ]
        if dtype in TF_DTYPES_MAP
        else []
    )
    value_for_attr = [
        f"Value for attr '{attr}' of {t} is not in the list of allowed values:"
        for attr in [
            "T",
            "Tidx",
            "TI",
            "out_type",
            "Tmultiples",
            "dtype",
            "Tcomplex",
            "Tb",
            "Ta",
        ]
        for t in [dtype] + ([TF_DTYPES_MAP[dtype]] if dtype in TF_DTYPES_MAP else [])
    ]
    value_to_param = [
        f"Value passed to parameter '{param}' has DataType {dtype} not in list of"
        " allowed values:"
        for param in ["alpha", "x"]
    ]
    substrings_to_check = value_for_attr + value_to_param + dtype_err_substrings
    return any(s in str(e) for s in substrings_to_check)


TORCH_DTYPES_MAP = {
    "bool": "Bool",
    "int8": "Char",
    "int16": "Short",
    "int32": "Int",
    "int64": "Long",
    "uint8": "Byte",
    "bfloat16": "BFloat16",
    "float16": "Half",
    "float32": "Float",
    "float64": "Double",
    "complex64": "ComplexFloat",
    "complex128": "ComplexDouble",
}
TORCH_SECONDARY_DTPYES_MAP = {
    "int8": "signed char",
    "int16": "short int",
    "int32": "int",
    "int64": "long int",
    "uint8": "unsigned char",
}


# TODO: check on is_floating_point (gets given tf tensor)
# TODO: from_numpy (obviously doesn't want tensor input)
# TODO: check can_cast (gets DType, not torch.dtype)
# TODO: check on triplet_margin_with_distance_loss (it's talking about tf dtypes)
def _is_dtype_err_torch(e, dtype):
    torch_dtype = TORCH_DTYPES_MAP[dtype]
    secondary_dtype = (
        TORCH_SECONDARY_DTPYES_MAP[dtype]
        if dtype in TORCH_SECONDARY_DTPYES_MAP
        else dtype
    )
    dtype_err_substrings = [
        f"\" not implemented for '{torch_dtype}'",
        f"not implemented for {torch_dtype}",
        f"{torch_dtype} type is not supported by",
        f"{torch_dtype} inputs not supported for",
        f"not supported for {torch_dtype}",
        f"Expected a floating point or complex tensor as input. Got {torch_dtype}",
        f"expected scalar type Long but found {torch_dtype}",
        f"Low precision dtypes not supported. Got {torch_dtype}",
        (
            "expected a tensor with 2 or more dimensions of float, double, cfloat or"
            " cdouble types"
        ),
        (
            "torch.finfo() requires a floating point input type. Use torch.iinfo to"
            " handle 'torch.finfo'"
        ),
        (
            "torch.iinfo() requires an integer input type. Use torch.finfo to handle"
            " 'torch.iinfo'"
        ),
        f"torch.{dtype} is not supported by",
        "only Tensors of floating point dtype can require gradients",
        "Unsupported input type encountered",
        "only support floating point and complex dtypes",
        f"Got unsupported ScalarType {torch_dtype}",
        f"received a {dtype} input for `y`, but {dtype} is not supported",
        "cannot take arguments of type float",
        "cannot take arguments of type uint",
        f"expects floating point dtypes, got: TensorOptions(dtype={secondary_dtype},",
        f"Unsupported dtype {torch_dtype}",
        (
            "Input dtype must be either a floating point or complex dtype. Got:"
            f" {torch_dtype}"
        ),
        "input tensor must be either float or double dtype",
        f"expected input to have floating point dtype but got {torch_dtype}",
        "only supports double, float and half tensors",
        f"expected probabilities tensor to have floating type, got {torch_dtype}",
        f"expected scalar type Double but found {torch_dtype}",
        f"only supports floating-point dtypes for input, got: {torch_dtype}",
        f"is not implemented for {torch_dtype.lower()}",
        "only supports floating-point dtypes",
        f"does not support {dtype} input",
        "is not implemented for tensors with non-complex dtypes.",
        (
            "input dtype should be either floating point or complex. Got"
            f" {torch_dtype} instead."
        ),
        f"Expected floating point type for result tensor, but got: {torch_dtype}",
        (
            "Optional dtype must be either a floating point or complex dtype. Got:"
            f" {torch_dtype}"
        ),
        (
            "only supported for half, float and double tensors, but got a tensor of"
            f" scalar type: {torch_dtype}"
        ),
        (
            "For integral input tensors, argument beta must not be a floating point"
            " number."
        ),
        f"Unknown Complex ScalarType for {torch_dtype}",
        f"supports floating point, complex, and integer tensors, but got {torch_dtype}",
        f"result type Float can't be cast to the desired output type {secondary_dtype}",
        "Integral inputs not supported.",
        f"result type Float can't be cast to the desired output type {torch_dtype}",
        f"to have floating point type, but got torch.{dtype}",
        (
            "expected common dtype to be floating point, yet common dtype is"
            f" {torch_dtype}"
        ),
        f"dtype must be a floating point but you specified {torch_dtype}",
        f"expected scalar type Float but found {torch_dtype}",
    ]
    only_if_bool = [
        "currently does not support bool dtype on CUDA.",
        "operator, on a bool tensor is not supported.",
        "operator, with two bool tensors is not supported.",
        "operator, with a bool tensor is not supported.",
        "tensor([[False, False]]) must be lesser than tensor([[False, False]])",
        (
            "tensor([[False, False]], device='cuda:0') must be lesser than"
            " tensor([[False, False]], device='cuda:0')"
        ),
        "Boolean inputs not supported for",
        "Expected parameter concentration (Tensor of shape (",
        "The input tensor may not be a boolean tensor.",
    ]
    only_if_complex = [
        "currently does not support complex dtypes on CUDA.",
        "is not supported for complex",
        "not implemented for complex tensors",
        f"expects a real-valued input tensor, but got {torch_dtype}",
        "is not yet implemented for complex tensors.",
        "expects standard deviation to be non-complex",
        "is not intended to support complex numbers.",
        "does not support complex input",
        "Complex inputs not supported.",
    ]
    substrings_to_check = (
        (only_if_bool if dtype == "bool" else [])
        + (only_if_complex if dtype in DTYPE_CLASSES["complex"] else [])
        + dtype_err_substrings
    )
    return any(s in str(e) for s in substrings_to_check)


dtype_err_fns_map = {
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
    if BACKENDS_DIR in file_path.parents:
        out = BACKENDS_TESTS_DIR
        if file_path.parent.stem == "experimental":
            out = out / "test_experimental"
        out = out / ("test_nn" if file_path.stem in NN_FILES else "test_core")
        if file_path.stem in REMAP_FILES:
            out = out / (REMAP_FILES[file_path.stem] + ".py")
        else:
            out = out / f"test_{file_path.stem}.py"
        return out
    else:
        out = FRONTENDS_TESTS_DIR
        started = False
        for p in file_path.parents[::-1]:
            if started:
                out = out / f"test_{p.name}"
            if p == FRONTENDS_DIR:
                started = True
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


def _tuple_to_dict(dtypes: tuple, is_unsupported: bool, all_dtypes: set):
    t = {dtype for dtype in dtypes if dtype not in DTYPE_CLASSES}.union(
        *[DTYPE_CLASSES[dtype] for dtype in dtypes if dtype in DTYPE_CLASSES]
    )
    t_complement = all_dtypes - t
    return {
        "supported": t_complement if is_unsupported else t,
        "unsupported": t if is_unsupported else t_complement,
    }


def _version_str_to_ints(v):
    parts = v.split(" ")
    upper = None
    if len(parts) == 1:
        upper = v
    elif "to" in parts:
        upper = parts[2]
    elif "below" in parts:
        upper = parts[0]
    # 'above' is not supported yet
    upper = tuple(int(x) for x in upper.split("."))
    return upper


def _parse_decorator(dec_string, device_list, dtype_set):
    left, _, right = dec_string.partition("{")
    inner_string = right.rpartition("}")[0]
    original_decorator_dict = eval("{" + inner_string + "}")
    mark_unsupp = "unsupp" in left
    has_device = "device" in left
    prev_result = {}
    latest = None
    latest_str = None
    for v in original_decorator_dict.keys():
        prev_result[v] = {}
        if has_device:
            for d, t in original_decorator_dict[v].items():
                if d in ["gpu", "tpu"]:
                    d = d + ":0"
                prev_result[v][d] = _tuple_to_dict(t, mark_unsupp, dtype_set)
            for d in device_list:
                if d not in prev_result[v]:
                    prev_result[v][d] = {
                        "supported": dtype_set,
                        "unsupported": set(),
                    }
        else:
            types_dict = _tuple_to_dict(
                original_decorator_dict[v], mark_unsupp, dtype_set
            )
            for d in device_list:
                prev_result[v][d] = {**types_dict}
        if latest is None:
            latest = _version_str_to_ints(v)
            latest_str = v
        else:
            this_version = _version_str_to_ints(v)
            if this_version > latest:
                latest = this_version
                latest_str = v
    return prev_result, latest_str, not mark_unsupp


def _type_list_to_tuple_string(dtype_list):
    return (
        "("
        + ", ".join([f'"{t}"' for t in dtype_list])
        + ("," if len(dtype_list) == 1 else "")
        + ")"
    )


def _get_decorator_string(dtype_lists, is_supported, version_string="backend_version"):
    d0 = "cpu"
    is_multi_device = any(
        d != d0 and dtype_lists[v][d] != dtype_lists[v][d0]
        for v in dtype_lists.keys()
        for d in dtype_lists[v].keys()
    )
    ret = "@with_"
    if not is_supported:
        ret += "un"
    ret += "supported_"
    if is_multi_device:
        ret += "device_and_"
    ret += "dtypes({"

    for v in dtype_lists.keys():
        mid = '"' + v + '": '
        if is_multi_device:
            mid += "{"
            for d, t in dtype_lists[v].items():
                if len(t) == 0:
                    continue
                mid += '"' + d.partition(":")[0] + '": '
                mid += _type_list_to_tuple_string(t)
                mid += ", "
            mid += "}"
        else:
            mid += _type_list_to_tuple_string(dtype_lists[v][d0])
        mid += ", "
        ret += mid

    ret += "}, " + version_string + ")"
    return ret


class BaseFileTester(ABC):
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

        self.test_path = _path_to_test_path(self.file_path)
        self.result = {}
        self.decorators = {}
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
            print("Results: ", end="", flush=True)

        self.is_set_up = True

    def iterate_dtypes(self):
        for dtype in self.dtypes:
            self.current_dtype = dtype
            if self.verbosity == 2:
                print()
                print(f"DType {dtype}: ", end="", flush=True)
            yield dtype

    @abstractmethod
    def get_test_name(self, fn_name):
        pass

    def iterate_fn_names(self):
        for fn_name in self.fn_names:
            if fn_name is None:
                if self.verbosity >= 1 and self.verbosity < 3:
                    print("-", end="", flush=True)
                continue
            self.current_fn_name = fn_name
            yield fn_name, self.get_test_name(fn_name)

    def iterate_devices(self):
        for device in self.devices:
            self.current_device = device
            if self.verbosity == 3:
                print()
                print(
                    f"{self.file_path}::{self.current_fn_name}"
                    f"[DType={self.current_dtype},Device={self.current_device}]: ",
                    end="",
                    flush=True,
                )
            yield device

    def remove_fn(self, fn_name, reason, in_process=True):
        if fn_name in self.fn_names:
            i = self.fn_names.index(fn_name)
            self.fn_names[i] = None
            if in_process and self.verbosity == 3:
                print()
                print(f"{self.file_path}::{fn_name}: skipped")
            elif in_process and self.verbosity >= 1:
                print("-", end="", flush=True)
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
                    else "-" if result == "skipped" else "?"
                )
            )
            print(char, end="", flush=True)

    @abstractmethod
    def get_decorator_string(self, *args):
        pass

    @abstractmethod
    def get_fresh_import_strings(self, decorator_import_string):
        pass

    @abstractmethod
    def run_test_function(self, the_test_func, min_example):
        pass

    @abstractmethod
    def is_dtype_err(self, e):
        pass

    def complete_test(self):
        # Skip functions for which all dtypes are skipped
        for f in self.fn_names:
            if f is not None and all(
                len(self.result[f][d]["skipped"]) == len(self.dtypes)
                for d in self.devices
            ):
                reasons = {
                    r for d in self.devices for _, r in self.result[f][d]["skipped"]
                }
                self.remove_fn(
                    f,
                    "all dtypes skipped with reasons: " + "; ".join(reasons),
                    in_process=False,
                )

        # Drop the skipped functions from the list
        self.fn_names = [f for f in self.fn_names if f is not None]
        if self.verbosity >= 1:
            for f, reason in self.skip_fns:
                print()
                print(f"Skipped function {f}: {reason}")
            print()

        # Read in the code file
        with open(self.file_path, "r") as f:
            code_text = f.readlines()

        for f in self.fn_names:
            existing_decorator_data = None
            latest_known_version = None
            existing_is_supported = None

            # Find function position in the code
            for i, v in enumerate(code_text):
                if f"def {f}(" in v:
                    def_index = i
                    break
            if self.verbosity >= 3:
                print(f"{f} found at line no. {def_index+1}")

            # Find and read existing dtype decorator(s)
            decorator_start_idx = def_index
            decorator_lines = []
            while (
                decorator_start_idx >= 1
                and code_text[decorator_start_idx - 1].strip() != ""
            ):
                decorator_lines.insert(0, code_text[decorator_start_idx - 1])
                decorator_start_idx -= 1

            indices_of_dtype_decs = []
            start_of_decorator = None
            dec_string = ""
            for i in range(len(decorator_lines)):
                line = decorator_lines[i]

                if start_of_decorator is None:
                    if REGEX_DICT["dtype_decorators"].search(line):
                        start_of_decorator = decorator_start_idx + i

                if start_of_decorator is not None:
                    dec_string += line
                    if dec_string.count("(") == dec_string.count(")"):
                        indices_of_dtype_decs.append(
                            (start_of_decorator, decorator_start_idx + i + 1)
                        )
                        (
                            existing_decorator_data,
                            latest_known_version,
                            existing_is_supported,
                        ) = _parse_decorator(dec_string, self.devices, set(self.dtypes))
                        start_of_decorator = None
                        dec_string = ""

            # Remove existing decorator if one was found
            for i, j in indices_of_dtype_decs:
                if self.verbosity >= 2:
                    print(f"Removing existing decorator from {f}")
                del code_text[i:j]
                def_index -= j - i

            for d in self.devices:
                # Report skipped dtypes (handling is later)
                if len(self.result[f][d]["skipped"]) > 0 and self.verbosity >= 1:
                    print("Unable to test:")
                    for dtype, reason in self.result[f][d]["skipped"]:
                        print(
                            f"{self.file_path}::{f}[DType={dtype},Device={d}] with"
                            f" error:\n {reason}"
                        )
                        print()

                # Handle unsure dtypes
                if self.handle_unsure in ["supported", "unsupported"]:
                    if len(self.result[f][d]["unsure"]) > 0 and self.verbosity >= 1:
                        print(f"Marking as {self.handle_unsure}:")
                        for dtype, reason in self.result[f][d]["unsure"]:
                            print(
                                f"{self.file_path}::{f}[DType={dtype},Device={d}] with"
                                f" error:\n  {reason}"
                            )
                            print()
                    self.result[f][d][self.handle_unsure].update(
                        {t for t, _ in self.result[f][d]["unsure"]}
                    )
                elif self.handle_unsure == "as_original":
                    self.result[f][d]["skipped"].update(
                        {t for t, _ in self.result[f][d]["unsure"]}
                    )
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

                # Handle skipped dtypes (and unsure if using as_original)
                existing_and_is_supported = (
                    existing_decorator_data is not None and existing_is_supported
                )
                existing_and_is_unsupported = (
                    existing_decorator_data is not None and not existing_is_supported
                )
                for t, _ in self.result[f][d]["skipped"]:
                    if (
                        existing_and_is_supported
                        and t
                        not in existing_decorator_data[latest_known_version][d][
                            "supported"
                        ]
                    ) or (
                        existing_and_is_unsupported
                        and t
                        in existing_decorator_data[latest_known_version][d][
                            "unsupported"
                        ]
                    ):
                        self.result[f][d]["unsupported"].add(t)
                    else:
                        self.result[f][d]["supported"].add(t)

            if existing_decorator_data is not None:
                versions_and_dtypes = existing_decorator_data
                current_version = _version_str_to_ints(self.current_version)
                latest_version_tuple = _version_str_to_ints(latest_known_version)
                if current_version == latest_version_tuple:
                    for d in self.devices:
                        versions_and_dtypes[latest_known_version][d] = {
                            "supported": self.result[f][d]["supported"],
                            "unsupported": self.result[f][d]["unsupported"],
                        }
                elif current_version > latest_version_tuple:
                    if all(
                        self.result[f][d]["supported"]
                        == versions_and_dtypes[latest_known_version][d]["supported"]
                        for d in self.devices
                    ) and all(
                        self.result[f][d]["unsupported"]
                        == versions_and_dtypes[latest_known_version][d]["unsupported"]
                        for d in self.devices
                    ):
                        new_version_string = ""
                        parts = latest_known_version.split(" ")
                        if len(parts) == 1:
                            new_version_string = (
                                latest_known_version + " to " + self.current_version
                            )
                        elif "to" in latest_known_version:
                            new_version_string = (
                                parts[0] + " to " + self.current_version
                            )
                        elif "below" in latest_known_version:
                            new_version_string = self.current_version + " and below"
                        versions_and_dtypes[new_version_string] = {}
                        for d in versions_and_dtypes[latest_known_version].keys():
                            versions_and_dtypes[new_version_string][d] = (
                                {
                                    "supported": self.result[f][d]["supported"],
                                    "unsupported": self.result[f][d]["unsupported"],
                                }
                                if d in self.devices
                                else versions_and_dtypes[latest_known_version][d]
                            )
                        del versions_and_dtypes[latest_known_version]
                    else:
                        versions_and_dtypes[self.current_version] = {}
                        for d in self.devices:
                            versions_and_dtypes[self.current_version][d] = {
                                "supported": self.result[f][d]["supported"],
                                "unsupported": self.result[f][d]["unsupported"],
                            }
            else:
                versions_and_dtypes = {
                    self.current_version: {
                        d: {
                            "supported": self.result[f][d]["supported"],
                            "unsupported": self.result[f][d]["unsupported"],
                        }
                        for d in self.devices
                    }
                }

            reduced_versions_and_dtypes = {}
            for v in versions_and_dtypes.keys():
                reduced_versions_and_dtypes[v] = {}
                for d in versions_and_dtypes[v].keys():
                    # Reduce (un)supported dtype lists to type classes where possible
                    supported = versions_and_dtypes[v][d]["supported"]
                    unsupported = versions_and_dtypes[v][d]["unsupported"]
                    for cls, types in DTYPE_CLASSES.items():
                        if types.issubset(supported):
                            supported = supported - types
                            supported.add(cls)
                        elif types.issubset(unsupported):
                            unsupported = unsupported - types
                            unsupported.add(cls)
                    reduced_versions_and_dtypes[v][d] = (supported, unsupported)

            # Generate updated decorator
            total_supported = sum(
                len(s)
                for d in reduced_versions_and_dtypes.values()
                for s, _ in d.values()
            )
            total_unsupported = sum(
                len(u)
                for d in reduced_versions_and_dtypes.values()
                for _, u in d.values()
            )
            use_supported = total_supported <= total_unsupported
            if total_unsupported == 0:
                self.decorators[f] = None
            else:
                for v in reduced_versions_and_dtypes.keys():
                    for d in reduced_versions_and_dtypes[v].keys():
                        reduced_versions_and_dtypes[v][d] = reduced_versions_and_dtypes[
                            v
                        ][d][0 if use_supported else 1]
                self.decorators[f] = self.get_decorator_string(
                    reduced_versions_and_dtypes, use_supported
                )
            if self.verbosity >= 2:
                print(f"  {f}: {self.decorators[f]}")

            # Write new decorator into the file text
            if self.decorators[f] is not None:
                if self.verbosity >= 2:
                    print(f"Adding decorator to {f}")
                code_text.insert(def_index, self.decorators[f] + "\n")

        code_text = "".join(code_text)

        # Change imports for the updated decorators
        all_decorators = set(REGEX_DICT["dtype_decorators"].findall(code_text))
        matches = [
            g1 + "," + g2
            for g1, g2 in REGEX_DICT["imported_wrappers"].findall(code_text)
        ]
        imported_decorators = {
            item.strip() for match in matches for item in match.split(",")
        }
        imported_decorators.discard("")
        desired_decorators = (imported_decorators - DTYPE_DECORATORS) | all_decorators

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
                    pattern, replacement = self.get_fresh_import_strings(
                        decorator_import_string
                    )
                    code_text = code_text.replace(pattern, replacement, 1)
                else:
                    code_text = REGEX_DICT["imported_wrappers"].sub(
                        decorator_import_string,
                        code_text,
                    )

        # Write the updated text to disk
        if self.safety_mode:
            print("File write suppressed due to safety mode.")
        else:
            with open(self.file_path, "w") as f:
                f.write(code_text)


class BackendFileTester(BaseFileTester):
    def __init__(
        self,
        file_path: Path,
        devices=[],
        fn_names=[],
        verbosity=0,
        handle_unsure="supported",
        safety_mode=False,
    ):
        super().__init__(
            file_path, devices, fn_names, verbosity, handle_unsure, safety_mode
        )
        for i in range(1, len(file_path.parents)):
            if file_path.parents[i].stem == "backends":
                self.backend = file_path.parents[i - 1].stem
                if verbosity >= 2:
                    print(f"Backend: {self.backend}")
                break
        else:
            raise Exception("No backend was identified.")
        self.current_version = version(PYPI_NAMES[self.backend])
        self._is_dtype_err = dtype_err_fns_map[self.backend]

    def setup_test(self):
        ivy.set_backend(self.backend)

        if "gpu:0" in self.devices and not ivy.gpu_is_available():
            self.devices.remove("gpu:0")
        if "tpu:0" in self.devices and not ivy.tpu_is_available():
            self.devices.remove("tpu:0")

        self.dtypes = ivy.valid_dtypes

        return super().setup_test()

    def get_test_name(self, fn_name):
        return f"test_{fn_name}"

    def get_decorator_string(self, *args):
        return _get_decorator_string(*args, version_string="backend_version")

    def get_fresh_import_strings(self, decorator_import_string):
        return (
            "import ivy\n",
            f"import ivy\n{decorator_import_string}from . import backend_version\n",
        )

    def run_test_function(self, the_test_func, min_example):
        return the_test_func(
            **min_example,
            backend_fw=self.backend,
            on_device=self.current_device,
        )

    def is_dtype_err(self, e):
        return self._is_dtype_err(e, self.current_dtype)


class FrontendFileTester(BaseFileTester):
    def __init__(
        self,
        file_path: Path,
        devices=[],
        fn_names=[],
        verbosity=0,
        handle_unsure="supported",
        safety_mode=False,
    ):
        super().__init__(
            file_path, devices, fn_names, verbosity, handle_unsure, safety_mode
        )
        for i in range(1, len(file_path.parents)):
            if file_path.parents[i].stem == "frontends":
                self.frontend = file_path.parents[i - 1].stem
                if verbosity >= 2:
                    print(f"Frontend: {self.frontend}")
                break
        else:
            raise Exception("No frontend was identified.")
        self.current_version = version(PYPI_NAMES[self.frontend])
        self._is_dtype_err = dtype_err_fns_map[self.frontend]

    def setup_test(self):
        frontend_config = importlib.import_module(
            f"ivy_tests.test_ivy.test_frontends.config.{self.frontend}"
        ).get_config()
        valid_devices = frontend_config.supported_devices.valid_devices
        self.devices = [d for d in self.devices if d.partition(":")[0] in valid_devices]
        self.dtypes = frontend_config.supported_dtypes.valid_dtypes
        return super().setup_test()

    def get_test_name(self, fn_name):
        return f"test_{self.frontend}_{fn_name}"

    def get_decorator_string(self, *args):
        return _get_decorator_string(*args, version_string=f"'{self.frontend}'")

    def get_fresh_import_strings(self, decorator_import_string):
        return (
            "import ivy_tests.test_ivy.helpers as helpers\n",
            f"import ivy_tests.test_ivy.helpers as helpers\n{decorator_import_string}",
        )

    def run_test_function(self, the_test_func, min_example):
        return the_test_func(
            **min_example,
            frontend=self.frontend,
            on_device=self.current_device,
            backend_fw="tensorflow",  # chosen for full dtype support
        )

    def is_dtype_err(self, e):
        return self._is_dtype_err(e, self.current_dtype)


def _kwargs_to_args_n_kwargs(num_positional_args, kwargs):
    args = [v for v in list(kwargs.values())[:num_positional_args]]
    kwargs = {k: kwargs[k] for k in list(kwargs.keys())[num_positional_args:]}
    return args, kwargs


def _get_nested_np_arrays(nest):
    indices = ivy.nested_argwhere(nest, lambda x: isinstance(x, np.ndarray))
    ret = ivy.multi_index_nest(nest, indices)
    return ret, indices, len(ret)


# global variable. I don't like using these but I don't see another option
_test_function_called = False


def create_args_kwargs(
    *,
    backend: str,
    args_np,
    arg_np_vals,
    args_idxs,
    kwargs_np,
    kwarg_np_vals,
    kwargs_idxs,
    input_dtypes,
    test_flags,
    on_device,
):
    # create args
    with helpers.BackendHandler.update_backend(backend) as ivy_backend:
        args = ivy_backend.copy_nest(args_np, to_mutable=False)
        ivy_backend.set_nest_at_indices(
            args,
            args_idxs,
            test_flags.apply_flags(
                arg_np_vals,
                input_dtypes,
                0,
                backend=backend,
                on_device=on_device,
            ),
        )

        # create kwargs
        kwargs = ivy_backend.copy_nest(kwargs_np, to_mutable=False)
        ivy_backend.set_nest_at_indices(
            kwargs,
            kwargs_idxs,
            test_flags.apply_flags(
                kwarg_np_vals,
                input_dtypes,
                len(arg_np_vals),
                backend=backend,
                on_device=on_device,
            ),
        )
    return args, kwargs


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
    global _test_function_called
    _test_function_called = True
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

        args, kwargs = create_args_kwargs(
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

        # Ignore the instance method flag and run it directly from the API
        target_fn = ivy_backend.__dict__[fn_name]

        try:
            target_fn(*args, **kwargs)
        except Exception as e:
            raise FromFunctionException(str(e), e)


def mock_test_frontend_function(
    *,
    input_dtypes,
    test_flags,
    backend_to_test,
    on_device="cpu",
    frontend,
    fn_tree,
    gt_fn_tree=None,
    rtol=None,
    atol=1e-06,
    tolerance_dict=None,
    test_values=True,
    **all_as_kwargs_np,
):
    global _test_function_called
    _test_function_called = True

    # split the arguments into their positional and keyword components
    args_np, kwargs_np = _kwargs_to_args_n_kwargs(
        num_positional_args=test_flags.num_positional_args, kwargs=all_as_kwargs_np
    )

    # extract all arrays from the arguments and keyword arguments
    arg_np_vals, args_idxs, c_arg_vals = _get_nested_np_arrays(args_np)
    kwarg_np_vals, kwargs_idxs, c_kwarg_vals = _get_nested_np_arrays(kwargs_np)

    # make all lists equal in length
    num_arrays = c_arg_vals + c_kwarg_vals
    if len(input_dtypes) < num_arrays:
        input_dtypes = [input_dtypes[0] for _ in range(num_arrays)]
    if len(test_flags.as_variable) < num_arrays:
        test_flags.as_variable = [test_flags.as_variable[0] for _ in range(num_arrays)]
    if len(test_flags.native_arrays) < num_arrays:
        test_flags.native_arrays = [
            test_flags.native_arrays[0] for _ in range(num_arrays)
        ]

    with helpers.BackendHandler.update_backend(backend_to_test) as ivy_backend:
        # update var flags to be compatible with float dtype and with_out args
        test_flags.as_variable = [
            v if ivy_backend.is_float_dtype(d) and not test_flags.with_out else False
            for v, d in zip(test_flags.as_variable, input_dtypes)
        ]

        local_importer = ivy_backend.utils.dynamic_import

        # strip the decorator to get an Ivy array
        # TODO, fix testing for jax frontend for x32
        if frontend == "jax":
            local_importer.import_module("ivy.functional.frontends.jax").config.update(
                "jax_enable_x64", True
            )

        _, _, fn_name = fn_tree.rpartition(".")

        # if gt_fn_tree and gt_fn_name are different from our frontend structure
        if gt_fn_tree is not None:
            split_index = gt_fn_tree.rfind(".")
            gt_frontend_submods, gt_fn_name = (
                gt_fn_tree[:split_index],
                gt_fn_tree[split_index + 1 :],
            )
        else:
            gt_frontend_submods, gt_fn_name = fn_tree[25 : fn_tree.rfind(".")], fn_name

        # apply test flags etc.
        args, kwargs = create_args_kwargs(
            backend=backend_to_test,
            args_np=args_np,
            arg_np_vals=arg_np_vals,
            args_idxs=args_idxs,
            kwargs_np=kwargs_np,
            kwarg_np_vals=kwarg_np_vals,
            kwargs_idxs=kwargs_idxs,
            input_dtypes=input_dtypes,
            test_flags=test_flags,
            on_device=on_device,
        )

        # strip the decorator to get an Ivy array
        # ToDo, fix testing for jax frontend for x32
        if frontend == "jax":
            importlib.import_module("ivy.functional.frontends.jax").config.update(
                "jax_enable_x64", True
            )

        args_for_test, kwargs_for_test = ivy_backend.args_to_ivy(*args, **kwargs)

        def arrays_to_numpy(x):
            return (
                ivy_backend.to_numpy(x._data) if isinstance(x, ivy_backend.Array) else x
            )

        gt_args_np = ivy.nested_map(
            args_for_test,
            arrays_to_numpy,
            shallow=False,
        )
        gt_kwargs_np = ivy.nested_map(
            kwargs_for_test,
            arrays_to_numpy,
            shallow=False,
        )

    # create frontend framework args
    frontend_config = helpers.get_frontend_config(frontend)
    args_frontend = ivy.nested_map(
        gt_args_np,
        lambda x: (
            frontend_config.native_array(x)
            if isinstance(x, np.ndarray)
            else (
                frontend_config.as_native_dtype(x)
                if isinstance(x, frontend_config.Dtype)
                else x
            )
        ),
        shallow=False,
    )
    kwargs_frontend = ivy.nested_map(
        gt_kwargs_np,
        lambda x: frontend_config.native_array(x) if isinstance(x, np.ndarray) else x,
        shallow=False,
    )

    # change ivy dtypes to native dtypes
    if "dtype" in kwargs_frontend and kwargs_frontend["dtype"] is not None:
        kwargs_frontend["dtype"] = frontend_config.as_native_dtype(
            kwargs_frontend["dtype"]
        )

    # change ivy device to native devices
    if "device" in kwargs_frontend:
        kwargs_frontend["device"] = frontend_config.as_native_device(
            kwargs_frontend["device"]
        )

    # compute the return via the frontend framework
    frontend_fw = importlib.import_module(gt_frontend_submods)
    frontend_fw_fn = frontend_fw.__dict__[gt_fn_name]

    try:
        frontend_fw_fn(*args_frontend, **kwargs_frontend)
    except Exception as e:
        raise FromFunctionException(str(e), e)


def run_dtype_setter(
    files_list: List[Path],
    devices=DEVICES,
    fn_names=[],
    options={"verbosity": 0, "handle_unsure": "supported", "safety_mode": False},
):
    global _test_function_called
    helpers.test_function = mock.Mock(wraps=mock_test_function)
    helpers.test_frontend_function = mock.Mock(wraps=mock_test_frontend_function)
    sys.modules["ivy_tests.test_ivy.helpers"] = helpers

    test_globals._set_backend("tensorflow")

    if options["verbosity"] >= 1:
        print("============== RUNNING =============")

    for file in files_list:
        if BACKENDS_DIR in file.parents:
            test_handler = BackendFileTester(file, devices, fn_names, **options)
        else:
            test_handler = FrontendFileTester(file, devices, fn_names, **options)
        try:
            test_handler.setup_test()
        except NoTestException:
            continue

        for dtype in test_handler.iterate_dtypes():
            helpers.get_dtypes = mock.Mock(return_value=st.just([dtype]))
            sys.modules["ivy_tests.test_ivy.helpers"] = helpers

            test_file = _import_module_from_path(test_handler.test_path)

            for fn_name, test_fn in test_handler.iterate_fn_names():
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
                    if isinstance(test_handler, BackendFileTester) and not any(
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
                            timer = threading.Timer(10, _thread.interrupt_main)
                            timer.start()
                            try:
                                min_example[k] = find(
                                    specifier=v, condition=lambda _: True
                                )
                            finally:
                                timer.cancel()
                except KeyboardInterrupt:
                    # Why interrupt_main() uses a KeyboardInterrupt is beyond me,
                    # but I suppose this is what we have to do
                    for _ in test_handler.iterate_devices():
                        test_handler.set_result(
                            "skipped", "Timeout generating example."
                        )
                    continue
                except Exception as e:
                    for _ in test_handler.iterate_devices():
                        test_handler.set_result("skipped", "In hypothesis: " + str(e))
                    continue

                for _ in test_handler.iterate_devices():
                    if isinstance(test_handler, BackendFileTester) and (
                        fn_name == "conv_general_transpose"
                        and test_handler.backend == "paddle"
                        and dtype == "float16"
                    ):
                        test_handler.set_result(
                            "skipped",
                            "paddle_backend.conv_general_transpose causes a segfault"
                            " for dtype float16.",
                        )
                        continue
                    try:
                        _test_function_called = False
                        with SuppressPrint():
                            test_handler.run_test_function(
                                test_file.__dict__[test_fn].original, min_example
                            )
                        if not _test_function_called:
                            raise NoTestFunctionException()
                        test_handler.set_result("supported")
                    except FromFunctionException as ffe:
                        e = ffe.wrapped_error
                        if "IvyNotImplementedException" in repr(type(e)):
                            test_handler.set_result("skipped", "Not implemented")
                        if test_handler.is_dtype_err(e):
                            test_handler.set_result("unsupported")
                        else:
                            test_handler.set_result("unsure", str(e) or type(e))
                    except NoTestFunctionException:
                        test_handler.set_result(
                            "skipped", "Test does not use test_function"
                        )
                    except hyp_errors.UnsatisfiedAssumption:
                        test_handler.set_result("skipped", "Unsatisfied assumption")
                    except Exception as e:
                        test_handler.set_result(
                            "skipped", "In test function: " + (str(e) or type(e))
                        )
        test_handler.complete_test()
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
        choices=["gpu", "tpu"],
        default=["gpu", "tpu"],
        help=(
            "specify devices to check. Others will be skipped. CPU will always be"
            " checked."
        ),
    )
    parser.add_argument(
        "-u",
        "--unsure",
        choices=["supported", "unsupported", "interactive", "as_original", "error"],
        default="supported",
        help=(
            "specify how to classify unknown error messages. Can be"
            " supported/unsupported, interactive (you will be shown the error message"
            " and asked to decide), as_original (follow any existing dtype decorator)"
            " or error (an error will be raised when an unknown message is found)."
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

    devices = [d + ":0" if d in ["gpu", "tpu"] else d for d in args.devices]

    run_dtype_setter(
        files_list,
        ["cpu"] + devices,
        args.functions,
        {
            "verbosity": args.verbosity,
            "handle_unsure": args.unsure,
            "safety_mode": args.safety_mode,
        },
    )


if __name__ == "__main__":
    main()
