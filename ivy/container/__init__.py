"""Base Container Object."""

# global
import colorama

try:
    # noinspection PyPackageRequirements
    import h5py
except ModuleNotFoundError:
    h5py = None

from .container import Container, ContainerBase  # noqa

# local
from .wrapping import add_ivy_container_instance_methods  # noqa

colorama.init(strip=False)
