"""Base Container Object."""

# global

try:
    # noinspection PyPackageRequirements
    import h5py
except ModuleNotFoundError:
    h5py = None

# local
from .wrapping import add_ivy_container_instance_methods  # noqa
from .container import ContainerBase, Container  # noqa
