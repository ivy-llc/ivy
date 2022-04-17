# local
import ivy
from .activations import ContainerWithActivations
from .base import ContainerBase
from .creation import ContainerWithCreation
from .data_types import ContainerWithDataTypes
from .device import ContainerWithDevice
from .elementwise import ContainerWithElementwise
from .general import ContainerWithGeneral
from .gradients import ContainerWithGradients
from .image import ContainerWithImage
from .layers import ContainerWithLayers
from .linear_algebra import ContainerWithLinearAlgebra
from .losses import ContainerWithLosses
from .manipulation import ContainerWithManipulation
from .norms import ContainerWithNorms
from .random import ContainerWithRandom
from .searching import ContainerWithSearching
from .set import ContainerWithSet
from .sorting import ContainerWithSorting
from .statistical import ContainerWithStatistical
from .utility import ContainerWithUtility


class Container(ContainerWithActivations, ContainerWithCreation, ContainerWithDataTypes,
                ContainerWithDevice, ContainerWithElementwise, ContainerWithGeneral, ContainerWithGradients,
                ContainerWithImage, ContainerWithLayers, ContainerWithLinearAlgebra, ContainerWithLosses,
                ContainerWithManipulation, ContainerWithNorms, ContainerWithRandom, ContainerWithSearching,
                ContainerWithSet, ContainerWithSorting, ContainerWithStatistical, ContainerWithUtility):

    def __init__(self, dict_in=None, queues=None, queue_load_sizes=None, container_combine_method='list_join',
                 queue_timeout=None, print_limit=10, key_length_limit=None, print_indent=4, print_line_spacing=0,
                 ivyh=None, default_key_color='green', keyword_color_dict=None, rebuild_child_containers=False,
                 types_to_iteratively_nest=None, alphabetical_keys=True, **kwargs):
        ContainerBase.__init__(
            self, dict_in, queues, queue_load_sizes, container_combine_method, queue_timeout, print_limit,
            key_length_limit, print_indent, print_line_spacing, ivyh, default_key_color, keyword_color_dict,
            rebuild_child_containers, types_to_iteratively_nest, alphabetical_keys, **kwargs)
        ContainerWithActivations.__init__(self)
        ContainerWithCreation.__init__(self)
        ContainerWithDataTypes.__init__(self)
        ContainerWithDevice.__init__(self)
        ContainerWithElementwise.__init__(self)
        ContainerWithGeneral.__init__(self)
        ContainerWithGradients.__init__(self)
        ContainerWithImage.__init__(self)
        ContainerWithLayers.__init__(self)
        ContainerWithLinearAlgebra.__init__(self)
        ContainerWithLosses.__init__(self)
        ContainerWithManipulation.__init__(self)
        ContainerWithNorms.__init__(self)
        ContainerWithRandom.__init__(self)
        ContainerWithSearching.__init__(self)
        ContainerWithSet.__init__(self)
        ContainerWithSorting.__init__(self)
        ContainerWithStatistical.__init__(self)
        ContainerWithUtility.__init__(self)


class MultiDevContainer(Container):

    def __init__(self, dict_in, devs, queues=None, queue_load_sizes=None, container_combine_method='list_join',
                 queue_timeout=None, print_limit=10, print_indent=4, print_line_spacing=0, ivyh=None,
                 keyword_color_dict=None, rebuild_child_containers=False, **kwargs):
        super().__init__(dict_in, queues, queue_load_sizes, container_combine_method, queue_timeout, print_limit,
                         print_indent, print_line_spacing, ivyh, keyword_color_dict, rebuild_child_containers, **kwargs)
        self._devs = devs
        self._num_devs = len(devs)

    def at_dev(self, dev):
        return self.map(lambda x, kc: x[dev] if isinstance(x, ivy.MultiDevItem) else x)

    def at_devs(self):
        return {ds: self.at_dev(ds) for ds in self._devs}
