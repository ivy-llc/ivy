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
        return self.map(lambda x, kc: -x, map_sequences=True)

    def __pow__(self, power):
        if isinstance(power, ivy.Container):
            return ivy.Container.multi_map(
                lambda xs, _: operator.pow(xs[0], xs[1]), [self, power], map_nests=True
            )
        return self.map(lambda x, kc: x**power, map_sequences=True)

    def __rpow__(self, power):
        return self.map(lambda x, kc: power**x, map_sequences=True)

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
        return ivy.Container.multi_map(
            lambda xs, _: operator.sub(xs[0], xs[1]), [self, other], map_nests=True
        )

    def __rsub__(self, other):
        return ivy.Container.multi_map(
            lambda xs, _: operator.sub(xs[0], xs[1]), [other, self], map_nests=True
        )

    def __mul__(self, other):
        return ivy.Container.multi_map(
            lambda xs, _: operator.mul(xs[0], xs[1]), [self, other], map_nests=True
        )

    def __rmul__(self, other):
        return ivy.Container.multi_map(
            lambda xs, _: operator.mul(xs[0], xs[1]), [other, self], map_nests=True
        )

    def __truediv__(self, other):
        return ivy.Container.multi_map(
            lambda xs, _: operator.truediv(xs[0], xs[1]), [self, other], map_nests=True
        )

    def __rtruediv__(self, other):
        return ivy.Container.multi_map(
            lambda xs, _: operator.truediv(xs[0], xs[1]), [other, self], map_nests=True
        )

    def __floordiv__(self, other):
        if isinstance(other, ivy.Container):
            return ivy.Container.multi_map(
                lambda xs, _: operator.floordiv(xs[0], xs[1]),
                [self, other],
                map_nests=True,
            )
        return self.map(lambda x, kc: x // other, map_sequences=True)

    def __rfloordiv__(self, other):
        return self.map(lambda x, kc: other // x, map_sequences=True)

    def __abs__(self):
        return self.map(lambda x, kc: operator.abs(x), map_sequences=True)

    def __lt__(self, other):
        if isinstance(other, ivy.Container):
            return ivy.Container.multi_map(
                lambda xs, _: operator.lt(xs[0], xs[1]), [self, other], map_nests=True
            )
        return self.map(lambda x, kc: x < other, map_sequences=True)

    def __le__(self, other):
        if isinstance(other, ivy.Container):
            return ivy.Container.multi_map(
                lambda xs, _: operator.le(xs[0], xs[1]), [self, other], map_nests=True
            )
        return self.map(lambda x, kc: x <= other, map_sequences=True)

    def __eq__(self, other):
        if isinstance(other, ivy.Container):
            return ivy.Container.multi_map(
                lambda xs, _: operator.eq(xs[0], xs[1]), [self, other], map_nests=True
            )
        return self.map(lambda x, kc: x == other, map_sequences=True)

    def __ne__(self, other):
        if isinstance(other, ivy.Container):
            return ivy.Container.multi_map(
                lambda xs, _: operator.ne(xs[0], xs[1]), [self, other], map_nests=True
            )
        return self.map(lambda x, kc: x != other, map_sequences=True)

    def __gt__(self, other):
        if isinstance(other, ivy.Container):
            return ivy.Container.multi_map(
                lambda xs, _: operator.gt(xs[0], xs[1]), [self, other], map_nests=True
            )
        return self.map(lambda x, kc: x > other, map_sequences=True)

    def __ge__(self, other):
        if isinstance(other, ivy.Container):
            return ivy.Container.multi_map(
                lambda xs, _: operator.ge(xs[0], xs[1]), [self, other], map_nests=True
            )
        return self.map(lambda x, kc: x >= other, map_sequences=True)

    def __and__(self, other):
        if isinstance(other, ivy.Container):
            return ivy.Container.multi_map(
                lambda xs, _: operator.and_(xs[0], xs[1]), [self, other], map_nests=True
            )
        return self.map(lambda x, kc: x and other, map_sequences=True)

    def __rand__(self, other):
        return self.map(lambda x, kc: other and x, map_sequences=True)

    def __or__(self, other):
        if isinstance(other, ivy.Container):
            return ivy.Container.multi_map(
                lambda xs, _: operator.or_(xs[0], xs[1]), [self, other], map_nests=True
            )
        return self.map(lambda x, kc: x or other, map_sequences=True)

    def __ror__(self, other):
        return self.map(lambda x, kc: other or x, map_sequences=True)

    def __invert__(self):
        return self.map(lambda x, kc: operator.not_(x), map_sequences=True)

    def __xor__(self, other):
        if isinstance(other, ivy.Container):
            return ivy.Container.multi_map(
                lambda xs, _: operator.xor(xs[0], xs[1]), [self, other], map_nests=True
            )
        return self.map(lambda x, kc: operator.xor(x, other), map_sequences=True)

    def __rxor__(self, other):
        return self.map(lambda x, kc: other != x, map_sequences=True)

    def __rshift__(self, other):
        """
        ivy.Container special method for the right shift operator, calling 
        :code:`operator.rshift` for each of the corresponding leaves of the
        two containers.

        Parameters
        ----------
        self
            first input container. Should have an integer data type.
        other
            second input array or container. Must be compatible with ``self`` 
            (see :ref:`broadcasting`). Should have an integer data type. 
            Each element must be greater than or equal to ``0``.

        Returns
        -------
        ret
            a container containing the element-wise results. The returned array 
            must have a data type determined by :ref:`type-promotion`.

        Examples
        --------
        With :code:`Number` instances at the leaves:

        >>> x = ivy.Container(a=128, b=43)
        >>> y = ivy.Container(a=5, b=3)
        >>> z = x >> y
        >>> print(z)
        {
            a: 4,
            b: 5
        }

        With :code:`ivy.Array` instances at the leaves:

        >>> x = ivy.Container(a=ivy.array([16, 40, 120]),\
                              b=ivy.array([15, 45, 143]))
        >>> y = ivy.Container(a=ivy.array([1, 2, 3]), \
                              b=ivy.array([0, 3, 4]))
        >>> z = x >> y
        >>> print(z)
        {
            a: ivy.array([8, 10, 15]),
            b: ivy.array([15, 5, 8])
        }

        With a mix of :code:`ivy.Container` and :code:`ivy.Array` instances:

        >>> x = ivy.Container(a=ivy.array([16, 40, 120]),\
                              b=ivy.array([15, 45, 143]))
        >>> y = ivy.array([1, 2, 3])
        >>> z = x >> y
        >>> print(z)
        {
            a: ivy.array([8, 10, 15]),
            b: ivy.array([7, 11, 17])
        }
        """
        if isinstance(other, ivy.Container):
            return ivy.Container.multi_map(
                lambda xs, _: operator.rshift(xs[0], xs[1]),
                [self, other],
                map_nests=True,
            )
        return self.map(lambda x, kc: operator.rshift(x, other), map_sequences=True)

    def __rrshift__(self, other):
        """
        ivy.Container reverse special method for the right shift operator, calling
        :code:`operator.rshift` for each of the corresponding leaves of the two
        containers.

        Parameters
        ----------
        self
            first input container. Should have an integer data type.
        other
            second input array or container. Must be compatible with ``self``
            (see :ref:`broadcasting`). Should have an integer data type. Each element
            must be greater than or equal to ``0``.

        Returns
        -------
        ret
            a container containing the element-wise results. The returned array
            must have a data type determined by :ref:`type-promotion`.

        Examples
        --------
        >>> a = 64
        >>> b = ivy.Container(a = ivy.array([0, 1, 2]), \
                              b = ivy.array([3, 4, 5]))
        >>> y = a >> b
        >>> print(y)
        {
            a: ivy.array([64, 32, 16]),
            b: ivy.array([8, 4, 2])
        }
        """
        return self.map(lambda x, kc: other >> x, map_sequences=True)

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
