"""
Collection of TensorFlow general functions, wrapped to fit Ivy syntax and signature.
"""

# global
import ivy
_round = round
import numpy as _np
import math as _math
import tensorflow as tf
from numbers import Number
from collections import Iterable

import multiprocessing as _multiprocessing
from tensorflow.python.types.core import Tensor

# local
from ivy.functional.ivy import default_dtype
from ivy.functional.ivy.device import default_device
from ivy.functional.backends.tensorflow.device import _dev_callable, dev_from_str



# API #
# ----#


