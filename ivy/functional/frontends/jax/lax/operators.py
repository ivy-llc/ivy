# global
from typing import Any
import ivy


def add(x, y):
    return ivy.add(x, y)


def tan(x):
    return ivy.tan(x)


tan.unsupported_dtypes = {"torch": ("float16",)}


def concatenate(operands, dimension):
    return ivy.concat(operands, axis=dimension)


def full(shape, fill_value, dtype=None):
    return ivy.full(shape, fill_value, dtype=dtype)


def max(x: Any, y: Any):
    return ivy.maximum(x, y)


def abs(x):
    return ivy.abs(x)


def sqrt(x):
    return ivy.sqrt(x)


sqrt.unsupported_dtypes = {"torch": ("float16",)}


def acos(x):
    return ivy.acos(x)


acos.unsupported_dtypes = {"torch": ("float16",)}


def sin(x):
    return ivy.sin(x)


sin.unsupported_dtypes = {"torch": ("float16",)}


def sign(x):
    return ivy.sign(x)


def asin(x):
    return ivy.asin(x)


asin.unsupported_dtypes = {"torch": ("float16",)}


def sinh(x):
    return ivy.sinh(x)


sinh.unsupported_dtypes = {"torch": ("float16",)}


def atan2(x, y):
    return ivy.atan2(x, y)


atan2.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def min(x, y):
    return ivy.minimum(x, y)


def mul(x, y):
    return ivy.multiply(x, y)


def eq(x, y):
    return ivy.equal(x, y)


def atan(x):
    return ivy.atan(x)


atan.unsupported_dtypes = {"torch": ("float16",)}


def cos(x):
    return ivy.cos(x)


cos.unsupported_dtypes = {"torch": ("float16",)}


def ceil(x):
    return ivy.ceil(x)


ceil.unsupported_dtypes = {"torch": ("float16",)}


def bitwise_and(x, y):
    return ivy.bitwise_and(x, y)


def bitwise_or(x, y):
    return ivy.bitwise_or(x, y)


def bitwise_not(x):
    return ivy.bitwise_invert(x)


def neg(x):
    return ivy.negative(x)


def argmax(operand, axis, index_dtype):
    return ivy.astype(ivy.argmax(operand, axis=axis), index_dtype)


def argmin(operand, axis, index_dtype):
    return ivy.astype(ivy.argmin(operand, axis=axis), index_dtype)


def bitwise_xor(x, y):
    return ivy.bitwise_xor(x, y)


def full_like(x, fill_value, dtype=None, shape=None):
    if shape is None:
        return ivy.full_like(x, fill_value, dtype=dtype)
    return ivy.full(shape, fill_value, dtype=dtype)


def exp(x):
    return ivy.exp(x)


exp.unsupported_dtypes = {"torch": ("float16",)}


def convert_element_type(operand, new_dtype):
    return ivy.astype(operand, new_dtype)


def cumprod(operand, axis=0, reverse=False):
    if reverse:
        return ivy.flip(ivy.cumprod(ivy.flip(operand), axis, dtype=operand.dtype))
    return ivy.cumprod(operand, axis, dtype=operand.dtype)


cumprod.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def cumsum(operand, axis=0, reverse=False):
    if reverse:
        return ivy.flip(ivy.cumsum(ivy.flip(operand), axis, dtype=operand.dtype))
    return ivy.cumsum(operand, axis, dtype=operand.dtype)


cumsum.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def ge(x, y):
    return ivy.greater_equal(x, y)


def gt(x, y):
    return ivy.greater(x, y)


def reshape(operand, new_sizes, dimensions=None):
    if dimensions:
        operand = ivy.permute_dims(operand, dimensions)
    return ivy.reshape(operand, new_sizes)


def reciprocal(x):
    return ivy.reciprocal(x)


reciprocal.unsupported_dtypes = {
    "torch": ("float16",),
    "tensorflow": (
        "uint8",
        "int8",
        "uint16",
        "int16",
        "uint32",
        "int32",
        "uint64",
        "int64",
    ),
}


def broadcast(operand, sizes):
    ret = ivy.zeros(tuple(sizes) + tuple(ivy.shape(operand)), dtype=ivy.dtype(operand))
    return ret + operand


def sort(operand, dimension=-1, is_stable=True, num_keys=1):
    return ivy.sort(operand, axis=dimension, stable=is_stable)


def le(x, y):
    return ivy.less_equal(x, y)


def ne(x, y):
    return ivy.not_equal(x, y)


def cosh(x):
    return ivy.cosh(x)


cosh.unsupported_dtypes = {"torch": ("float16",)}


def round(x):
    return ivy.round(x)


round.unsupported_dtypes = {"torch": ("float16",)}


def lt(x, y):
    return ivy.less(x, y)


def pow(x, y):
    return ivy.pow(x, y)


pow.unsupported_dtypes = ("int64", "int32", "int16", "uint64", "uint32", "uint16")


def clamp(min, x, max):
    return ivy.clip(x, min, max)


clamp.unsupported_dtypes = {"torch": ("float16",)}


def log(x):
    return ivy.log(x)


log.unsupported_dtypes = {"torch": ("float16",)}


def rev(operand, dimensions):
    return ivy.flip(operand, axis=dimensions)


def div(x, y):
    return ivy.astype(ivy.divide(x, y), x.dtype)


def rsqrt(x):
    return ivy.reciprocal(ivy.sqrt(x))


rsqrt.unsupported_dtypes = {
    "jax": ("int64", "int32", "int16", "uint64", "uint32", "uint16"),
    "torch": ("float16",),
}


def expm1(x):
    return ivy.expm1(x)


expm1.supported_dtypes = ("bfloat16", "float16", "float32", "float64")


def log1p(x):
    return ivy.log1p(x)


def pad(operand, padding_value, padding_config):
    operand_new = []
    padding_row = []
    for i in range(0, operand.shape[1] + (operand.shape[1] - 1) * padding_config[1][2]):
        padding_row.append(padding_value)
    for i in range(operand.shape[0] * 2 - 1):
        if i % 2 != 0:
            for k in range(0, padding_config[0][2]):
                operand_new.append(padding_row)
        else:
            row = []
            for j in range(operand.shape[1] * 2 - 1):
                if j % 2 == 0:
                    row.append(operand[int(i / 2)][int(j / 2)])
                else:
                    for k in range(0, padding_config[1][2]):
                        row.append(padding_value)
            operand_new.append(row)
    operand_new = ivy.array(operand_new)
    operand_copy = operand_new
    ret_new = []
    for j in range(0, padding_config[0][0]):
        row = []
        for i in range(operand_copy.shape[1] + padding_config[1][0] + padding_config[1][1]):
            row.append(padding_value)
        ret_new.append(row)
    for k in range(0, operand_copy.shape[0]):
        col = []
        for i in range(padding_config[1][0]):
            col.append(padding_value)
        for j in range(0, operand_copy.shape[1]):
            col.append(operand_copy[k][j])
        for i in range(padding_config[1][1]):
            col.append(padding_value)
        ret_new.append(col)
    for j in range(0, padding_config[0][1]):
        row = []
        for i in range(operand_copy.shape[1] + padding_config[1][0] + padding_config[1][1]):
            row.append(padding_value)
        ret_new.append(row)
    ret_new = ivy.asarray(ret_new, dtype=padding_value.dtype)
    return ret_new
