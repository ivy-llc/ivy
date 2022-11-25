import ivy
import functools
import logging
from types import FunctionType
from typing import Callable
import Cython_func_wrapper

# import typing


# for wrapping (sequence matters)
FN_DECORATORS = Cython_func_wrapper.FN_DECORATORS


# Helpers #
# --------#


def _get_first_array(*args, **kwargs):
    return Cython_func_wrapper._get_first_array(args,kwargs)


# Array Handling #
# ---------------#


def handle_array_like(fn: Callable):
    return Cython_func_wrapper.handle_array_like(fn)


def inputs_to_native_arrays(fn: Callable):
    return Cython_func_wrapper.inputs_to_native_arrays(fn)

def inputs_to_ivy_arrays(fn: Callable):
    return Cython_func_wrapper.inputs_to_ivy_array(fn)


def outputs_to_ivy_arrays(fn: Callable):
    return Cython_func_wrapper.outputs_to_ivy_arrays(fn)


def _is_zero_dim_array(x):
    return  Cython_func_wrapper._is_zero_dim_array(x)


def from_zero_dim_arrays_to_float(fn: Callable):
    return Cython_func_wrapper.from_zero_dim_arrays_to_float(fn)


def to_native_arrays_and_back(fn: Callable):
    return Cython_func_wrapper.to_native_arrays_and_back(fn)


# Data Type Handling #
# -------------------#


def infer_dtype(fn: Callable):
    return Cython_func_wrapper.infer_dtype(fn)


def integer_arrays_to_float(fn: Callable):
    return Cython_func_wrapper.integer_arrays_to_float(fn)


# Device Handling #
# ----------------#


def infer_device(fn: Callable):
    return Cython_func_wrapper.infer_device(fn)


# Inplace Update Handling #
# ------------------------#


def handle_out_argument(fn: Callable):
    return Cython_func_wrapper.handle_out_argument(fn)


# Nestable Handling #
# ------------------#


def handle_nestable(fn: Callable):
    return Cython_func_wrapper.handle_nestable(fn)


# Functions #


def _wrap_function(key: str, to_wrap: Callable, original: Callable):
    return Cython_func_wrapper._wrap_function(key,to_wrap,original)


# Gets dtype from a version dictionary
def _dtype_from_version(dic, version):
    return Cython_func_wrapper._dtype_from_version(dic,version)


def _versioned_attribute_factory(attribute_function, base):
    return Cython_func_wrapper._versioned_attribute_factory(attribute_function, base)


def _dtype_device_wrapper_creator(attrib, t):
    return Cython_func_wrapper._dtype_device_wrapper_creator(attrib, t)


# nans Handling #
# --------------#


def _leaf_has_nans(x):
    return Cython_func_wrapper._leaf_has_nans(x)


def _nest_has_nans(x):
    return Cython_func_wrapper._nest_has_nans(x)


def handle_nans(fn: Callable):
    return Cython_func_wrapper.handle_nans(fn)


# Decorators to allow for versioned attributes
# These are already calling the C functions so no need for  Cython_func_wrapper
with_unsupported_dtypes = _dtype_device_wrapper_creator("unsupported_dtypes", tuple)
with_supported_dtypes = _dtype_device_wrapper_creator("supported_dtypes", tuple)
with_unsupported_devices = _dtype_device_wrapper_creator("unsupported_devices", tuple)
with_supported_devices = _dtype_device_wrapper_creator("supported_devices", tuple)
with_unsupported_device_and_dtypes = _dtype_device_wrapper_creator(
    "unsupported_device_and_dtype", dict
)
with_supported_device_and_dtypes = _dtype_device_wrapper_creator(
    "supported_device_and_dtype", dict
)
