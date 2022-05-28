# global
import copy
from operator import lt as _lt
from operator import le as _le
from operator import eq as _eq
from operator import ne as _ne
from operator import gt as _gt
from operator import ge as _ge
from operator import mul as _mul
from operator import pow as _pow
from operator import not_ as _not
from functools import reduce as _reduce
from operator import truediv as _truediv
from operator import floordiv as _floordiv

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


class Container(
    ContainerWithActivations,
    ContainerWithCreation,
    ContainerWithDataTypes,
    ContainerWithDevice,
    ContainerWithElementwise,
    ContainerWithGeneral,
    ContainerWithGradients,
    ContainerWithImage,
    ContainerWithLayers,
    ContainerWithLinearAlgebra,
    ContainerWithLosses,
    ContainerWithManipulation,
    ContainerWithNorms,
    ContainerWithRandom,
    ContainerWithSearching,
    ContainerWithSet,
    ContainerWithSorting,
    ContainerWithStatistical,
    ContainerWithUtility,
):
    def __init__(
        self,
        dict_in=None,
        queues=None,
        queue_load_sizes=None,
        container_combine_method="list_join",
        queue_timeout=None,
        print_limit=10,
        key_length_limit=None,
        print_indent=4,
        print_line_spacing=0,
        ivyh=None,
        default_key_color="green",
        keyword_color_dict=None,
        rebuild_child_containers=False,
        types_to_iteratively_nest=None,
        alphabetical_keys=True,
        **kwargs
    ):
        ContainerBase.__init__(
            self,
            dict_in,
            queues,
            queue_load_sizes,
            container_combine_method,
            queue_timeout,
            print_limit,
            key_length_limit,
            print_indent,
            print_line_spacing,
            ivyh,
            default_key_color,
            keyword_color_dict,
            rebuild_child_containers,
            types_to_iteratively_nest,
            alphabetical_keys,
            **kwargs
        )

    # Built-ins #
    # ----------#

    def __pos__(self):
        return self

    def __neg__(self):
        return self.map(lambda x, kc: -x)

    def __pow__(self, power):
        if isinstance(power, ivy.Container):
            return self.reduce([self, power], lambda x: _reduce(_pow, x))
        return self.map(lambda x, kc: x**power)

    def __rpow__(self, power):
        return self.map(lambda x, kc: power**x)

    def __add__(self, other):
        if isinstance(other, ivy.Container):
            return self.reduce([self, other], sum)
        return self.map(lambda x, kc: x + other)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, ivy.Container):
            return self.reduce([self, -other], sum)
        return self.map(lambda x, kc: x - other)

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        if isinstance(other, ivy.Container):
            return self.reduce([self, other], lambda x: _reduce(_mul, x))
        return self.map(lambda x, kc: x * other)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, ivy.Container):
            return self.reduce([self, other], lambda x: _reduce(_truediv, x))
        return self.map(lambda x, kc: x / other)

    def __rtruediv__(self, other):
        return self.map(lambda x, kc: other / x)

    def __floordiv__(self, other):
        if isinstance(other, ivy.Container):
            return self.reduce([self, other], lambda x: _reduce(_floordiv, x))
        return self.map(lambda x, kc: x // other)

    def __rfloordiv__(self, other):
        return self.map(lambda x, kc: other // x)

    def __abs__(self):
        return self.map(lambda x, kc: self._ivy.abs(x))

    def __lt__(self, other):
        if isinstance(other, ivy.Container):
            return self.reduce([self, other], lambda x: _reduce(_lt, x))
        return self.map(lambda x, kc: x < other)

    def __le__(self, other):
        if isinstance(other, ivy.Container):
            return self.reduce([self, other], lambda x: _reduce(_le, x))
        return self.map(lambda x, kc: x <= other)

    def __eq__(self, other):
        if isinstance(other, ivy.Container):
            return self.reduce([self, other], lambda x: _reduce(_eq, x))
        return self.map(lambda x, kc: x == other)

    def __ne__(self, other):
        if isinstance(other, ivy.Container):
            return self.reduce([self, other], lambda x: _reduce(_ne, x))
        return self.map(lambda x, kc: x != other)

    def __gt__(self, other):
        if isinstance(other, ivy.Container):
            return self.reduce([self, other], lambda x: _reduce(_gt, x))
        return self.map(lambda x, kc: x > other)

    def __ge__(self, other):
        if isinstance(other, ivy.Container):
            return self.reduce([self, other], lambda x: _reduce(_ge, x))
        return self.map(lambda x, kc: x >= other)

    def __and__(self, other):
        if isinstance(other, ivy.Container):
            return self.reduce([self, other], lambda x: x[0] and x[1])
        return self.map(lambda x, kc: x and other)

    def __rand__(self, other):
        return self.map(lambda x, kc: other and x)

    def __or__(self, other):
        if isinstance(other, ivy.Container):
            return self.reduce([self, other], lambda x: x[0] or x[1])
        return self.map(lambda x, kc: x or other)

    def __ror__(self, other):
        return self.map(lambda x, kc: other or x)

    def __invert__(self):
        return self.map(lambda x, kc: _not(x))

    def __xor__(self, other):
        if isinstance(other, ivy.Container):
            return self.reduce([self, other], lambda x: x[0] != x[1])
        return self.map(lambda x, kc: x != other)

    def __rxor__(self, other):
        return self.map(lambda x, kc: other != x)

    def __getstate__(self):
        state_dict = copy.copy(self.__dict__)
        state_dict["_local_ivy"] = ivy.try_else_none(
            lambda: state_dict["_local_ivy"].current_backend_str()
        )
        config_in = copy.copy(state_dict["_config_in"])
        config_in["ivyh"] = ivy.try_else_none(
            lambda: config_in["ivyh"].current_backend_str()
        )
        state_dict["_config_in"] = config_in
        config = copy.copy(state_dict["_config"])
        config["ivyh"] = ivy.try_else_none(lambda: config["ivyh"].current_backend_str())
        state_dict["_config"] = config
        return state_dict

    def __setstate__(self, state_dict):
        if "_local_ivy" in state_dict:
            if ivy.exists(state_dict["_local_ivy"]):
                state_dict["_local_ivy"] = ivy.get_backend(state_dict["_local_ivy"])
        if "_config_in" in state_dict:
            config_in = copy.copy(state_dict["_config_in"])
            if "ivyh" in config_in:
                if ivy.exists(config_in["ivyh"]):
                    config_in["ivyh"] = ivy.get_backend(config_in["ivyh"])
            state_dict["_config_in"] = config_in
        if "_config" in state_dict:
            config = copy.copy(state_dict["_config"])
            if "ivyh" in config:
                if ivy.exists(config["ivyh"]):
                    config["ivyh"] = ivy.get_backend(config["ivyh"])
            state_dict["_config"] = config
        self.__dict__.update(state_dict)


class MultiDevContainer(Container):
    def __init__(
        self,
        dict_in,
        devs,
        queues=None,
        queue_load_sizes=None,
        container_combine_method="list_join",
        queue_timeout=None,
        print_limit=10,
        key_length_limit=None,
        print_indent=4,
        print_line_spacing=0,
        ivyh=None,
        default_key_color="green",
        keyword_color_dict=None,
        rebuild_child_containers=False,
        types_to_iteratively_nest=None,
        alphabetical_keys=True,
        **kwargs
    ):
        super().__init__(
            dict_in,
            queues,
            queue_load_sizes,
            container_combine_method,
            queue_timeout,
            print_limit,
            key_length_limit,
            print_indent,
            print_line_spacing,
            ivyh,
            default_key_color,
            keyword_color_dict,
            rebuild_child_containers,
            types_to_iteratively_nest,
            alphabetical_keys,
            **kwargs
        )
        self._devs = devs
        self._num_devs = len(devs)

    def at_dev(self, dev):
        return self.map(lambda x, kc: x[dev] if isinstance(x, ivy.MultiDevItem) else x)

    def at_devs(self):
        return {ds: self.at_dev(ds) for ds in self._devs}
