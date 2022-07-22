# global
import copy
import operator

# local
import ivy
from .activations import ContainerWithActivations
from .base import ContainerBase
from .creation import ContainerWithCreation
from .data_type import ContainerWithDataTypes
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
            return self.reduce([self, power], lambda xs: pow(xs[0], xs[1]))
        return self.map(lambda x, kc: x**power)

    def __rpow__(self, power):
        return self.map(lambda x, kc: power**x)

    def __add__(self, other):
        """
        ivy.Container special method for the add operator, calling :code:`operator.add`
        for each of the corresponding leaves of the two containers.

        Parameters
        ----------
        self
            first input container. Should have a numeric data type.
        other
            second input array or container. Must be compatible with ``self``
            (see :ref:`broadcasting`). Should have a numeric data type.
        
        Returns
        -------
        ret
            a container containing the element-wise sums. The returned array must have a
            data type determined by :ref:`type-promotion`.

        Examples
        --------
        With :code:`Number` instances at the leaves:

        >>> x = ivy.Container(a=1, b=2)
        >>> y = ivy.Container(a=3, b=4)
        >>> z = x + y
        >>> print(z)
        {
            a: 4,
            b: 6
        }

        With :code:`ivy.Array` instances at the leaves:

        >>> x = ivy.Container(a=ivy.array([1, 2, 3]),\
                              b=ivy.array([2, 3, 4]))
        >>> y = ivy.Container(a=ivy.array([4, 5, 6]), \
                              b=ivy.array([5, 6, 7]))
        >>> z = x + y
        >>> print(z)
        {
            a: ivy.array([5, 7, 9]),
            b: ivy.array([7, 9, 11])
        }

        With a mix of :code:`ivy.Container` and :code:`ivy.Array` instances:

        >>> x = ivy.Container(a=ivy.array([[4.], [5.], [6.]]),\
                              b=ivy.array([[5.], [6.], [7.]]))
        >>> y = ivy.array([[1.1, 2.3, -3.6]])
        >>> z = x + y
        >>> print(z)
        {
            a: ivy.array([[5.1, 6.3, 0.4],
                          [6.1, 7.3, 1.4],
                          [7.1, 8.3, 2.4]]),
            b: ivy.array([[6.1, 7.3, 1.4],
                          [7.1, 8.3, 2.4],
                          [8.1, 9.3, 3.4]])
        }
        """
        return ivy.Container.multi_map(
            lambda xs, _: operator.add(xs[0], xs[1]), [self, other], map_nests=True
        )

    def __radd__(self, other):
        """
        ivy.Container reverse special method for the add operator, calling
        :code:`operator.add` for each of the corresponding leaves of the two containers.

        Parameters
        ----------
        self
            first input container. Should have a numeric data type.
        other
            second input array or container. Must be compatible with ``self``
            (see :ref:`broadcasting`). Should have a numeric data type.

        Returns
        -------
        ret
            a container containing the element-wise sums. The returned array must have a
            data type determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = 1
        >>> y = ivy.Container(a=3, b=4)
        >>> z = x + y
        >>> print(z)
        {
            a: 4,
            b: 5
        }
        """
        return ivy.Container.multi_map(
            lambda xs, _: operator.add(xs[0], xs[1]), [other, self], map_nests=True
        )

    def __sub__(self, other):
        return self.static_subtract(self, other)

    def __rsub__(self, other):
        return self.static_subtract(other, self)

    def __mul__(self, other):
        return self.static_multiply(self, other)

    def __rmul__(self, other):
        return self.static_multiply(other, self)

    def __truediv__(self, other):
        return self.static_divide(self, other)

    def __rtruediv__(self, other):
        return self.static_divide(other, self)

    def __floordiv__(self, other):
        if isinstance(other, ivy.Container):
            return self.reduce(
                [self, other], lambda xs: operator.floordiv(xs[0], xs[1])
            )
        return self.map(lambda x, kc: x // other)

    def __rfloordiv__(self, other):
        return self.map(lambda x, kc: other // x)

    def __abs__(self):
        return self.map(lambda x, kc: self._ivy.abs(x))

    def __lt__(self, other):
        if isinstance(other, ivy.Container):
            return self.reduce([self, other], lambda xs: operator.lt(xs[0], xs[1]))
        return self.map(lambda x, kc: x < other)

    def __le__(self, other):
        if isinstance(other, ivy.Container):
            return self.reduce([self, other], lambda xs: operator.le(xs[0], xs[1]))
        return self.map(lambda x, kc: x <= other)

    def __eq__(self, other):
        if isinstance(other, ivy.Container):
            return self.reduce([self, other], lambda xs: operator.eq(xs[0], xs[1]))
        return self.map(lambda x, kc: x == other)

    def __ne__(self, other):
        if isinstance(other, ivy.Container):
            return self.reduce([self, other], lambda xs: operator.ne(xs[0], xs[1]))
        return self.map(lambda x, kc: x != other)

    def __gt__(self, other):
        if isinstance(other, ivy.Container):
            return self.reduce([self, other], lambda xs: operator.gt(xs[0], xs[1]))
        return self.map(lambda x, kc: x > other)

    def __ge__(self, other):
        if isinstance(other, ivy.Container):
            return self.reduce([self, other], lambda xs: operator.ge(xs[0], xs[1]))
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
        return self.map(lambda x, kc: operator.not_(x))

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
