"""Base Container Object."""

# global
import colorama

try:
    # noinspection PyPackageRequirements
    import h5py as _h5py
except ModuleNotFoundError:
    _h5py = None

# local
from .wrapping import add_ivy_container_instance_methods, _wrap_function  # noqa
from .container import ContainerBase, Container, MultiDevContainer  # noqa

colorama.init(strip=False)
