# global
import sys

# local
from ivy.functional.frontends import set_frontend_to_specific_version
from . import cluster
from . import constants
from . import fft
from . import fftpack
from . import integrate
from . import interpolate
from . import linalg
from . import ndimage
from . import odr
from . import optimize
from . import signal
from . import sparse
from . import sparse
from . import spatial
from . import special
from . import stats

import ivy.functional.frontends.numpy as np


array = _frontend_array = np.array

# setting to specific version #
# --------------------------- #

set_frontend_to_specific_version(sys.modules[__name__])
