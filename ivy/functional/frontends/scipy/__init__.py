# global
import sys

import ivy
import ivy.functional.frontends.numpy as np

# local
from ivy.functional.frontends import set_frontend_to_specific_version

from . import (
    cluster,
    constants,
    fft,
    fftpack,
    integrate,
    interpolate,
    linalg,
    ndimage,
    odr,
    optimize,
    signal,
    sparse,
    spatial,
    special,
    stats,
)

array = _frontend_array = np.array

# setting to specific version #
# --------------------------- #

if ivy.is_local():
    module = ivy.utils._importlib.import_cache[__name__]
else:
    module = sys.modules[__name__]

set_frontend_to_specific_version(module)
