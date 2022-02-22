
# global
import numpy as np
from numbers import Number
from typing import Union, Iterable

# local
import ivy
from ivy.functional.ivy.core.device import dev

from ivy.framework_handler import current_framework as _cur_framework


def zeros(shape: Iterable[int], dtype: ivy.Dtype = None, dev: ivy.Device = None) \
        -> ivy.Array:
    return _cur_framework.zeros(shape, dtype, dev)
