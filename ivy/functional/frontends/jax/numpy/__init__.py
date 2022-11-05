# flake8: noqa
from typing import Optional

import numpy as np
from etils.array_types import ArrayLike, Array
from jax._src.numpy.fft import _fft_core_1d
from jax._src.numpy.util import _wraps
from jaxlib import xla_client

from . import fft
from . import linalg
from . import name_space_functions
from .name_space_functions import *


@_wraps(np.fft.fft)
def fft(a: ArrayLike, n: Optional[int] = None,
        axis: int = -1, norm: Optional[str] = None) -> Array:
    return _fft_core_1d('fft', xla_client.FftType.FFT, a, n=n, axis=axis,
                        norm=norm)
