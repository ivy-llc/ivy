# Copyright 2023 The ml_dtypes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Overload of numpy.finfo to handle dtypes defined in ml_dtypes."""

from typing import Dict

from ml_dtypes._custom_floats import bfloat16
from ml_dtypes._custom_floats import float8_e4m3b11
from ml_dtypes._custom_floats import float8_e4m3fn
from ml_dtypes._custom_floats import float8_e5m2

import numpy as np

_bfloat16_dtype = np.dtype(bfloat16)
_float8_e4m3b11_dtype = np.dtype(float8_e4m3b11)
_float8_e4m3fn_dtype = np.dtype(float8_e4m3fn)
_float8_e5m2_dtype = np.dtype(float8_e5m2)


class _Bfloat16MachArLike:

  def __init__(self):
    smallest_normal = float.fromhex("0x1p-126")
    self.smallest_normal = bfloat16(smallest_normal)
    smallest_subnormal = float.fromhex("0x1p-133")
    self.smallest_subnormal = bfloat16(smallest_subnormal)


class _Float8E4m3B11MachArLike:

  def __init__(self):
    smallest_normal = float.fromhex("0x1p-10")
    self.smallest_normal = float8_e4m3b11(smallest_normal)
    smallest_subnormal = float.fromhex("0x1p-13")
    self.smallest_subnormal = float8_e4m3b11(smallest_subnormal)


class _Float8E4m3FnMachArLike:

  def __init__(self):
    smallest_normal = float.fromhex("0x1p-6")
    self.smallest_normal = float8_e4m3fn(smallest_normal)
    smallest_subnormal = float.fromhex("0x1p-9")
    self.smallest_subnormal = float8_e4m3fn(smallest_subnormal)


class _Float8E5m2MachArLike:

  def __init__(self):
    smallest_normal = float.fromhex("0x1p-14")
    self.smallest_normal = float8_e5m2(smallest_normal)
    smallest_subnormal = float.fromhex("0x1p-16")
    self.smallest_subnormal = float8_e5m2(smallest_subnormal)


class finfo(np.finfo):  # pylint: disable=invalid-name,missing-class-docstring
  __doc__ = np.finfo.__doc__
  _finfo_cache: Dict[np.dtype, np.finfo] = {}

  @staticmethod
  def _bfloat16_finfo():
    def float_to_str(f):
      return "%12.4e" % float(f)

    tiny = float.fromhex("0x1p-126")
    resolution = 0.01
    eps = float.fromhex("0x1p-7")
    epsneg = float.fromhex("0x1p-8")
    max_ = float.fromhex("0x1.FEp127")

    obj = object.__new__(np.finfo)
    obj.dtype = _bfloat16_dtype
    obj.bits = 16
    obj.eps = bfloat16(eps)
    obj.epsneg = bfloat16(epsneg)
    obj.machep = -7
    obj.negep = -8
    obj.max = bfloat16(max_)
    obj.min = bfloat16(-max_)
    obj.nexp = 8
    obj.nmant = 7
    obj.iexp = obj.nexp
    obj.maxexp = 128
    obj.precision = 2
    obj.resolution = bfloat16(resolution)
    # pylint: disable=protected-access
    obj._machar = _Bfloat16MachArLike()
    if not hasattr(obj, "tiny"):
      obj.tiny = bfloat16(tiny)
    if not hasattr(obj, "smallest_normal"):
      obj.smallest_normal = obj._machar.smallest_normal
    obj.smallest_subnormal = obj._machar.smallest_subnormal

    obj._str_tiny = float_to_str(tiny)
    obj._str_smallest_normal = float_to_str(tiny)
    obj._str_max = float_to_str(max_)
    obj._str_epsneg = float_to_str(epsneg)
    obj._str_eps = float_to_str(eps)
    obj._str_resolution = float_to_str(resolution)
    # pylint: enable=protected-access
    return obj

  @staticmethod
  def _float8_e4m3b11_finfo():
    def float_to_str(f):
      return "%6.2e" % float(f)

    tiny = float.fromhex("0x1p-10")
    resolution = 0.1
    eps = float.fromhex("0x1p-3")
    epsneg = float.fromhex("0x1p-4")
    max_ = float.fromhex("0x1.Ep4")

    obj = object.__new__(np.finfo)
    obj.dtype = _float8_e4m3b11_dtype
    obj.bits = 8
    obj.eps = float8_e4m3b11(eps)
    obj.epsneg = float8_e4m3b11(epsneg)
    obj.machep = -3
    obj.negep = -4
    obj.max = float8_e4m3b11(max_)
    obj.min = float8_e4m3b11(-max_)
    obj.nexp = 4
    obj.nmant = 3
    obj.iexp = obj.nexp
    obj.maxexp = 5
    obj.precision = 1
    obj.resolution = float8_e4m3b11(resolution)
    # pylint: disable=protected-access
    obj._machar = _Float8E4m3B11MachArLike()
    if not hasattr(obj, "tiny"):
      obj.tiny = float8_e4m3b11(tiny)
    if not hasattr(obj, "smallest_normal"):
      obj.smallest_normal = obj._machar.smallest_normal
    obj.smallest_subnormal = obj._machar.smallest_subnormal

    obj._str_tiny = float_to_str(tiny)
    obj._str_smallest_normal = float_to_str(tiny)
    obj._str_max = float_to_str(max_)
    obj._str_epsneg = float_to_str(epsneg)
    obj._str_eps = float_to_str(eps)
    obj._str_resolution = float_to_str(resolution)
    # pylint: enable=protected-access
    return obj

  @staticmethod
  def _float8_e4m3fn_finfo():
    def float_to_str(f):
      return "%6.2e" % float(f)

    tiny = float.fromhex("0x1p-6")
    resolution = 0.1
    eps = float.fromhex("0x1p-3")
    epsneg = float.fromhex("0x1p-4")
    max_ = float.fromhex("0x1.Cp8")

    obj = object.__new__(np.finfo)
    obj.dtype = _float8_e4m3fn_dtype
    obj.bits = 8
    obj.eps = float8_e4m3fn(eps)
    obj.epsneg = float8_e4m3fn(epsneg)
    obj.machep = -3
    obj.negep = -4
    obj.max = float8_e4m3fn(max_)
    obj.min = float8_e4m3fn(-max_)
    obj.nexp = 4
    obj.nmant = 3
    obj.iexp = obj.nexp
    obj.maxexp = 9
    obj.precision = 1
    obj.resolution = float8_e4m3fn(resolution)
    # pylint: disable=protected-access
    obj._machar = _Float8E4m3FnMachArLike()
    if not hasattr(obj, "tiny"):
      obj.tiny = float8_e4m3fn(tiny)
    if not hasattr(obj, "smallest_normal"):
      obj.smallest_normal = obj._machar.smallest_normal
    obj.smallest_subnormal = obj._machar.smallest_subnormal

    obj._str_tiny = float_to_str(tiny)
    obj._str_smallest_normal = float_to_str(tiny)
    obj._str_max = float_to_str(max_)
    obj._str_epsneg = float_to_str(epsneg)
    obj._str_eps = float_to_str(eps)
    obj._str_resolution = float_to_str(resolution)
    # pylint: enable=protected-access
    return obj

  @staticmethod
  def _float8_e5m2_finfo():
    def float_to_str(f):
      return "%6.2e" % float(f)

    tiny = float.fromhex("0x1p-14")
    resolution = 0.1
    eps = float.fromhex("0x1p-2")
    epsneg = float.fromhex("0x1p-3")
    max_ = float.fromhex("0x1.Cp15")

    obj = object.__new__(np.finfo)
    obj.dtype = _float8_e5m2_dtype
    obj.bits = 8
    obj.eps = float8_e5m2(eps)
    obj.epsneg = float8_e5m2(epsneg)
    obj.machep = -2
    obj.negep = -3
    obj.max = float8_e5m2(max_)
    obj.min = float8_e5m2(-max_)
    obj.nexp = 5
    obj.nmant = 2
    obj.iexp = obj.nexp
    obj.maxexp = 16
    obj.precision = 1
    obj.resolution = float8_e5m2(resolution)
    # pylint: disable=protected-access
    obj._machar = _Float8E5m2MachArLike()
    if not hasattr(obj, "tiny"):
      obj.tiny = float8_e5m2(tiny)
    if not hasattr(obj, "smallest_normal"):
      obj.smallest_normal = obj._machar.smallest_normal
    obj.smallest_subnormal = obj._machar.smallest_subnormal

    obj._str_tiny = float_to_str(tiny)
    obj._str_smallest_normal = float_to_str(tiny)
    obj._str_max = float_to_str(max_)
    obj._str_epsneg = float_to_str(epsneg)
    obj._str_eps = float_to_str(eps)
    obj._str_resolution = float_to_str(resolution)
    # pylint: enable=protected-access
    return obj

  def __new__(cls, dtype):
    if (
        isinstance(dtype, str)
        and dtype == "bfloat16"
        or dtype == _bfloat16_dtype
    ):
      if _bfloat16_dtype not in cls._finfo_cache:
        cls._finfo_cache[_bfloat16_dtype] = cls._bfloat16_finfo()
      return cls._finfo_cache[_bfloat16_dtype]
    if (
        isinstance(dtype, str)
        and dtype == "float8_e4m3b11"
        or dtype == _float8_e4m3b11_dtype
    ):
      if _float8_e4m3b11_dtype not in cls._finfo_cache:
        cls._finfo_cache[_float8_e4m3b11_dtype] = cls._float8_e4m3b11_finfo()
      return cls._finfo_cache[_float8_e4m3b11_dtype]
    if (
        isinstance(dtype, str)
        and dtype == "float8_e4m3fn"
        or dtype == _float8_e4m3fn_dtype
    ):
      if _float8_e4m3fn_dtype not in cls._finfo_cache:
        cls._finfo_cache[_float8_e4m3fn_dtype] = cls._float8_e4m3fn_finfo()
      return cls._finfo_cache[_float8_e4m3fn_dtype]
    if (
        isinstance(dtype, str)
        and dtype == "float8_e5m2"
        or dtype == _float8_e5m2_dtype
    ):
      if _float8_e5m2_dtype not in cls._finfo_cache:
        cls._finfo_cache[_float8_e5m2_dtype] = cls._float8_e5m2_finfo()
      return cls._finfo_cache[_float8_e5m2_dtype]
    return super().__new__(cls, dtype)
