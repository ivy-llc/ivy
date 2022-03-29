"""
Collection of MXNet general functions, wrapped to fit Ivy syntax and signature.
"""

# global
import ivy
_round = round
import logging
import mxnet as _mx
import numpy as _np
import math as _math
from numbers import Number
from operator import mul as _mul
from functools import reduce as _reduce
import multiprocessing as _multiprocessing

# local
from ivy.functional.ivy import default_dtype
from ivy.functional.ivy.device import default_device
from ivy.functional.backends.mxnet.device import _callable_dev
from ivy.functional.backends.mxnet.general import unstack
from ivy.functional.backends.mxnet import _handle_flat_arrays_in_out, _mxnet_init_context,\
    _scalar_or_flat_array_to_scalar, _handle_flat_arrays_in, _flat_array_to_1_dim_array, _1_dim_array_to_flat_array

#temporary imports
from ivy.functional.backends.mxnet.general import linspace


# API #
# ----#














