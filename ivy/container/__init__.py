"""Base Container Object."""

# global
import colorama

try:
    # noinspection PyPackageRequirements
    import h5py
except ModuleNotFoundError:
    h5py = None

# local
from .wrapping import add_ivy_container_instance_methods  # noqa
from .container import ContainerBase, Container  # noqa

colorama.init(strip=False)
