"""
Collection of Jax general functions, wrapped to fit Ivy syntax and signature.
"""

# global
import jax
import math as _math
import numpy as _onp
import jax.numpy as jnp
import jaxlib as _jaxlib
from numbers import Number
from collections import Iterable
from operator import mul as _mul
from functools import reduce as _reduce
from jaxlib.xla_extension import Buffer
import multiprocessing as _multiprocessing
from haiku._src.data_structures import FlatMapping

# local
import ivy
from ivy.functional.ivy.device import default_device
from ivy.functional.ivy import default_dtype
from ivy.functional.backends.jax.device import to_dev, dev as callable_dev



# Helpers #
# --------#





# API #
# ----#












