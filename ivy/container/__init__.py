"""
Base Container Object
"""

# global
import re
import colorama
try:
    # noinspection PyPackageRequirements
    import h5py as _h5py
except ModuleNotFoundError:
    _h5py = None

# local
from ivy.container.container import Container, MultiDevContainer

colorama.init(strip=False)
