"""
Base Container Object
"""

# global
import colorama
try:
    # noinspection PyPackageRequirements
    import h5py as _h5py
except ModuleNotFoundError:
    _h5py = None

# local
from .container import Container, MultiDevContainer
from .wrapping import add_ivy_container_instance_methods

colorama.init(strip=False)
