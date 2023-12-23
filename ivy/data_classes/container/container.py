# global
import operator

# local
import ivy
from .activations import _ContainerWithActivations
from .base import ContainerBase
from .conversions import _ContainerWithConversions
from .creation import _ContainerWithCreation
from .data_type import _ContainerWithDataTypes
from .device import _ContainerWithDevice
from .elementwise import _ContainerWithElementwise
from .general import _ContainerWithGeneral
from .gradients import _ContainerWithGradients
from .image import _ContainerWithImage
from .layers import _ContainerWithLayers
from .linear_algebra import _ContainerWithLinearAlgebra
from .losses import _ContainerWithLosses
from .manipulation import _ContainerWithManipulation
from .norms import _ContainerWithNorms
from .random import _ContainerWithRandom
from .searching import _ContainerWithSearching
from .set import _ContainerWithSet
from .sorting import _ContainerWithSorting
from .statistical import _ContainerWithStatistical
from .utility import _ContainerWithUtility
from ivy.data_classes.container.experimental import (
    _ContainerWithActivationExperimental,
    _ContainerWithConversionExperimental,
    _ContainerWithCreationExperimental,
    _ContainerWithData_typeExperimental,
    _ContainerWithDeviceExperimental,
    _ContainerWithElementWiseExperimental,
    _ContainerWithGeneralExperimental,
    _ContainerWithGradientsExperimental,
    _ContainerWithImageExperimental,
    _ContainerWithLayersExperimental,
    _ContainerWithLinearAlgebraExperimental,
    _ContainerWithManipulationExperimental,
    _ContainerWithNormsExperimental,
    _ContainerWithRandomExperimental,
    _ContainerWithSearchingExperimental,
    _ContainerWithSetExperimental,
    _ContainerWithSortingExperimental,
    _ContainerWithStatisticalExperimental,
    _ContainerWithUtilityExperimental,
    _ContainerWithLossesExperimental,
)


class Container(
    _ContainerWithActivations,
    _ContainerWithConversions,
    _ContainerWithCreation,
    _ContainerWithDataTypes,
    _ContainerWithDevice,
    _ContainerWithElementwise,
    _ContainerWithGeneral,
    _ContainerWithGradients,
    _ContainerWithImage,
    _ContainerWithLayers,
    _ContainerWithLinearAlgebra,
    _ContainerWithLosses,
    _ContainerWithManipulation,
    _ContainerWithNorms,
    _ContainerWithRandom,
    _ContainerWithSearching,
    _ContainerWithSet,
    _ContainerWithSorting,
    _ContainerWithStatistical,
    _ContainerWithUtility,
    _ContainerWithActivationExperimental,
    _ContainerWithConversionExperimental,
    _ContainerWithCreationExperimental,
    _ContainerWithData_typeExperimental,
    _ContainerWithDeviceExperimental,
    _ContainerWithElementWiseExperimental,
    _ContainerWithGeneralExperimental,
    _ContainerWithGradientsExperimental,
    _ContainerWithImageExperimental,
    _ContainerWithLayersExperimental,
    _ContainerWithLinearAlgebraExperimental,
    _ContainerWithManipulationExperimental,
    _ContainerWithNormsExperimental,
    _ContainerWithRandomExperimental,
    _ContainerWithSearchingExperimental,
    _ContainerWithSetExperimental,
    _ContainerWithSortingExperimental,
    _ContainerWithStatisticalExperimental,
    _ContainerWithUtilityExperimental,
    _ContainerWithLossesExperimental,
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
        dynamic_backend=None,
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
            dynamic_backend,
            **kwargs
        )

    # Built-ins #
    # ----------#

    def __pos__(self):
        return self

    def __neg__(self):
        return self.cont_map(lambda x, kc: -x, map_sequences=True)

    def __pow__(self, power):
        """ivy.Container special method for the power operator, calling
        :code:`operator.pow` for each of the corresponding leaves of the two
        containers.

        Parameters
        ----------
        self
            input container. Should have a numeric data type.
        power
            input array or container of powers. Must be compatible
            with ``self`` (see :ref:`broadcasting`). Should have a numeric
            data type.

        Returns
        -------
        ret
            a container containing the element-wise sums. The returned array must have a
            data type determined by :ref:`type-promotion`.

        Examples
        --------
        With :class:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([0, 1]), b=ivy.array([2, 3]))
        >>> y = x ** 2
        >>> print(y)
        {
            a: ivy.array([0, 1]),
            b: ivy.array([4, 9])
        }
        >>> x = ivy.Container(a=ivy.array([0, 1.2]), b=ivy.array([2.2, 3.]))
        >>> y = x ** 3.1
        >>> print(y)
        {
            a: ivy.array([0., 1.75979435]),
            b: ivy.array([11.52153397, 30.13532257])
        }
        """
        if isinstance(power, ivy.Container):
            return ivy.Container.cont_multi_map(
                lambda xs, _: operator.pow(xs[0], xs[1]), [self, power], map_nests=True
            )
        return self.cont_map(lambda x, kc: x**power, map_sequences=True)

    def __rpow__(self, power):
        return self.cont_map(lambda x, kc: power**x, map_sequences=True)

    def __ipow__(self, power):
        if isinstance(power, ivy.Container):
            return ivy.Container.cont_multi_map(
                lambda xs, _: operator.ipow(xs[0], xs[1]), [self, power], map_nests=True
            )
        return self.cont_map(lambda x, _: operator.ipow(x, power), map_sequences=True)

    def __add__(self, other):
        """ivy.Container special method for the add operator, calling
        :code:`operator.add` for each of the corresponding leaves of the two
        containers.

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

        With :class:`ivy.Array` instances at the leaves:

        >>> x = ivy.Container(a=ivy.array([1, 2, 3]),
        ...                   b=ivy.array([2, 3, 4]))
        >>> y = ivy.Container(a=ivy.array([4, 5, 6]),
        ...                   b=ivy.array([5, 6, 7]))
        >>> z = x + y
        >>> print(z)
        {
            a: ivy.array([5, 7, 9]),
            b: ivy.array([7, 9, 11])
        }

        With a mix of :class:`ivy.Container` and :class:`ivy.Array` instances:

        >>> x = ivy.Container(a=ivy.array([[4.], [5.], [6.]]),
        ...                   b=ivy.array([[5.], [6.], [7.]]))
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
        return ivy.Container.cont_multi_map(
            lambda xs, _: operator.add(xs[0], xs[1]), [self, other], map_nests=True
        )

    def __radd__(self, other):
        """ivy.Container reverse special method for the add operator, calling
        :code:`operator.add` for each of the corresponding leaves of the two
        containers.

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
        return ivy.Container.cont_multi_map(
            lambda xs, _: operator.add(xs[0], xs[1]), [other, self], map_nests=True
        )

    def __iadd__(self, other):
        return ivy.Container.cont_multi_map(
            lambda xs, _: operator.iadd(xs[0], xs[1]), [self, other], map_nests=True
        )

    def __sub__(self, other):
        """ivy.Container special method for the subtract operator, calling
        :code:`operator.sub` for each of the corresponding leaves of the two
        containers.

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
            a container containing the element-wise differences. The returned array must
            have a data type determined by :ref:`type-promotion`.

        Examples
        --------
        With :code:`Number` instances at the leaves:

        >>> x = ivy.Container(a=1, b=2)
        >>> y = ivy.Container(a=3, b=4)
        >>> z = x - y
        >>> print(z)
        {
            a: -2,
            b: -2
        }

        With :class:`ivy.Array` instances at the leaves:

        >>> x = ivy.Container(a=ivy.array([1, 2, 3]),
        ...                   b=ivy.array([4, 3, 2]))
        >>> y = ivy.Container(a=ivy.array([4, 5, 6]),
        ...                   b=ivy.array([6, 5, 4]))
        >>> z = x - y
        >>> print(z)
        {
            a: ivy.array([-3, -3, -3]),
            b: ivy.array([-2, -2, -2])
        }

        With a mix of :class:`ivy.Container` and :class:`ivy.Array` instances:

        >>> x = ivy.Container(a=ivy.array([[4.], [5.], [6.]]),
        ...                   b=ivy.array([[5.], [6.], [7.]]))
        >>> y = ivy.array([[1.1, 2.3, -3.6]])
        >>> z = x - y
        >>> print(z)
        {
            a: ivy.array([[2.9, 1.7, 7.6],
                          [3.9, 2.7, 8.6],
                          [4.9, 3.7, 9.6]]),
            b: ivy.array([[3.9, 2.7, 8.6],
                          [4.9, 3.7, 9.6],
                          [5.9, 4.7, 10.6]])
        }
        """
        return ivy.Container.cont_multi_map(
            lambda xs, _: operator.sub(xs[0], xs[1]), [self, other], map_nests=True
        )

    def __isub__(self, other):
        return ivy.Container.cont_multi_map(
            lambda xs, _: operator.isub(xs[0], xs[1]), [self, other], map_nests=True
        )

    def __rsub__(self, other):
        """ivy.Container reverse special method for the subtract operator,
        calling :code:`operator.sub` for each of the corresponding leaves of
        the two containers.

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
            a container containing the element-wise differences. The returned array must
            have a data type determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = 1
        >>> y = ivy.Container(a=3, b=4)
        >>> z = x - y
        >>> print(z)
        {
            a: -2,
            b: -3
        }
        """
        return ivy.Container.cont_multi_map(
            lambda xs, _: operator.sub(xs[0], xs[1]), [other, self], map_nests=True
        )

    def __mul__(self, other):
        return ivy.Container.cont_multi_map(
            lambda xs, _: operator.mul(xs[0], xs[1]), [self, other], map_nests=True
        )

    def __rmul__(self, other):
        return ivy.Container.cont_multi_map(
            lambda xs, _: operator.mul(xs[0], xs[1]), [other, self], map_nests=True
        )

    def __imul__(self, other):
        return ivy.Container.cont_multi_map(
            lambda xs, _: operator.imul(xs[0], xs[1]), [self, other], map_nests=True
        )

    def __mod__(self, other):
        return ivy.Container.cont_multi_map(
            lambda xs, _: operator.mod(xs[0], xs[1]), [self, other], map_nests=True
        )

    def __rmod__(self, other):
        return ivy.Container.cont_multi_map(
            lambda xs, _: operator.mod(xs[0], xs[1]), [other, self], map_nests=True
        )

    def __imod__(self, other):
        return ivy.Container.cont_multi_map(
            lambda xs, _: operator.imod(xs[0], xs[1]), [self, other], map_nests=True
        )

    def __divmod__(self, other):
        return ivy.Container.cont_multi_map(
            lambda xs, _: (operator.truediv(xs[0], xs[1]), operator.mod(xs[0], xs[1])),
            [self, other],
            map_nests=True,
        )

    def __rdivmod__(self, other):
        return ivy.Container.cont_multi_map(
            lambda xs, _: (operator.truediv(xs[0], xs[1]), operator.mod(xs[0], xs[1])),
            [other, self],
            map_nests=True,
        )

    def __truediv__(self, other):
        """ivy.Container special method for the divide operator, calling
        :code:`operator.truediv` for each of the corresponding leaves of the
        two containers.

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
            a container containing the element-wise differences. The returned array must
            have a data type determined by :ref:`type-promotion`.

        Examples
        --------
        With :code:`Number` instances at the leaves:

        >>> x = ivy.Container(a=1, b=2)
        >>> y = ivy.Container(a=5, b=4)
        >>> z = x / y
        >>> print(z)
        {
            a: 0.2,
            b: 0.5
        }

        With :class:`ivy.Array` instances at the leaves:

        >>> x = ivy.Container(a=ivy.array([1, 2, 3]),
        ...                   b=ivy.array([4, 3, 2]))
        >>> y = ivy.Container(a=ivy.array([4, 5, 6]),
        ...                   b=ivy.array([6, 5, 4]))
        >>> z = x / y
        >>> print(z)
        {
            a: ivy.array([0.25, 0.40000001, 0.5]),
            b: ivy.array([0.66666669, 0.60000002, 0.5])
        }
        """
        return ivy.Container.cont_multi_map(
            lambda xs, _: operator.truediv(xs[0], xs[1]), [self, other], map_nests=True
        )

    def __rtruediv__(self, other):
        return ivy.Container.cont_multi_map(
            lambda xs, _: operator.truediv(xs[0], xs[1]), [other, self], map_nests=True
        )

    def __itruediv__(self, other):
        return ivy.Container.cont_multi_map(
            lambda xs, _: operator.itruediv(xs[0], xs[1]), [self, other], map_nests=True
        )

    def __floordiv__(self, other):
        if isinstance(other, ivy.Container):
            return ivy.Container.cont_multi_map(
                lambda xs, _: operator.floordiv(xs[0], xs[1]),
                [self, other],
                map_nests=True,
            )
        return self.cont_map(lambda x, kc: x // other, map_sequences=True)

    def __rfloordiv__(self, other):
        return self.cont_map(lambda x, kc: other // x, map_sequences=True)

    def __ifloordiv__(self, other):
        if isinstance(other, ivy.Container):
            return ivy.Container.cont_multi_map(
                lambda xs, _: operator.ifloordiv(xs[0], xs[1]),
                [self, other],
                map_nests=True,
            )
        return self.cont_map(
            lambda x, kc: operator.ifloordiv(x, other), map_sequences=True
        )

    def __matmul__(self, other):
        if isinstance(other, ivy.Container):
            return ivy.Container.cont_multi_map(
                lambda xs, _: operator.matmul(xs[0], xs[1]),
                [self, other],
                map_nests=True,
            )
        return self.cont_map(
            lambda x, kc: operator.matmul(x, other), map_sequences=True
        )

    def __rmatmul__(self, other):
        if isinstance(other, ivy.Container):
            return ivy.Container.cont_multi_map(
                lambda xs, _: operator.matmul(xs[0], xs[1]),
                [other, self],
                map_nests=True,
            )
        return self.cont_map(
            lambda x, kc: operator.matmul(other, x), map_sequences=True
        )

    def __imatmul__(self, other):
        if isinstance(other, ivy.Container):
            return ivy.Container.cont_multi_map(
                lambda xs, _: operator.imatmul(xs[0], xs[1]),
                [other, self],
                map_nests=True,
            )
        return self.cont_map(
            lambda x, kc: operator.imatmul(x, other), map_sequences=True
        )

    def __abs__(self):
        """ivy.Container special method for the abs operator, calling
        :code:`operator.abs` for each of the corresponding leaves of the two
        containers.

        Parameters
        ----------
        self
            input Container. Should have leaves with numeric data type.

        Returns
        -------
        ret
            A container containing the element-wise results.

        Examples
        --------
        With :class:`ivy.Container` instances:

        >>> x = ivy.Container(a=ivy.array([1, -2, 3]),
        ...                    b=ivy.array([-1, 0, 5]))
        >>> y = abs(x)
        >>> print(y)
        {
            a: ivy.array([1, 2, 3]),
            b: ivy.array([1, 0, 5])
        }
        """
        return self.cont_map(lambda x, kc: operator.abs(x), map_sequences=True)

    def __lt__(self, other):
        """ivy.Container special method for the less operator, calling
        :code:`operator.lt` for each of the corresponding leaves of the two
        containers.

        Parameters
        ----------
        self
            first input Container. May have any data type.
        other
            second input Container. Must be compatible with x1 (with Broadcasting).
            May have any data type.

        Returns
        -------
        ret
            A container containing the element-wise results. Any returned array inside
            must have a data type of bool.

        Examples
        --------
        With :class:`ivy.Container` instances:

        >>> x = ivy.Container(a=ivy.array([4, 5, 6]),b=ivy.array([2, 3, 4]))
        >>> y = ivy.Container(a=ivy.array([1, 5, 3]),b=ivy.array([5, 3, 7]))
        >>> z = x < y
        >>> print(z)
        {
            a: ivy.array([False, False, False]),
            b: ivy.array([True, False, True])
        }
        """
        if isinstance(other, ivy.Container):
            return ivy.Container.cont_multi_map(
                lambda xs, _: operator.lt(xs[0], xs[1]), [self, other], map_nests=True
            )
        return self.cont_map(lambda x, kc: x < other, map_sequences=True)

    def __le__(self, other):
        """ivy.Container special method for the less_equal operator, calling
        :code:`operator.le` for each of the corresponding leaves of the two
        containers.

        Parameters
        ----------
        self
            first input Container. May have any data type.
        other
            second input Container. Must be compatible with x1 (with Broadcasting).
            May have any data type.

        Returns
        -------
        ret
            A container containing the element-wise results. Any returned array inside
            must have a data type of bool.

        Examples
        --------
        With :class:`ivy.Container` instances:

        >>> x = ivy.Container(a=ivy.array([4, 5, 6]),b=ivy.array([2, 3, 4]))
        >>> y = ivy.Container(a=ivy.array([1, 5, 3]),b=ivy.array([5, 3, 7]))
        >>> z = x <= y
        >>> print(z)
        {
            a: ivy.array([False, True, False]),
            b: ivy.array([True, True, True])
        }
        """
        if isinstance(other, ivy.Container):
            return ivy.Container.cont_multi_map(
                lambda xs, _: operator.le(xs[0], xs[1]), [self, other], map_nests=True
            )
        return self.cont_map(lambda x, kc: x <= other, map_sequences=True)

    def __eq__(self, other):
        """ivy.Container special method for the equal operator, calling
        :code:`operator.eq` for each of the corresponding leaves of the two
        containers.

        Parameters
        ----------
        self
            first input Container. May have any data type.
        other
            second input Container. Must be compatible with x1 (with Broadcasting).
            May have any data type.

        Returns
        -------
        ret
            A container containing the element-wise results. Any returned array inside
            must have a data type of bool.

        Examples
        --------
        With :class:`ivy.Container` instances:

        >>> x1 = ivy.Container(a=ivy.array([1, 2, 3]),
        ...                    b=ivy.array([1, 3, 5]))
        >>> x2 = ivy.Container(a=ivy.array([1, 2, 3]),
        ...                    b=ivy.array([1, 4, 5]))
        >>> y = x1 == x2
        >>> print(y)
        {
            a: ivy.array([True, True, True]),
            b: ivy.array([True, False, True])
        }

        >>> x1 = ivy.Container(a=ivy.array([1.0, 2.0, 3.0]),
        ...                    b=ivy.array([1, 4, 5]))
        >>> x2 = ivy.Container(a=ivy.array([1, 3, 3.0]),
        ...                    b=ivy.array([1.0, 4.0, 5.0]))
        >>> y = x1 == x2
        >>> print(y)
        {
            a: ivy.array([True, False, True]),
            b: ivy.array([True, True, True])
        }

        >>> x1 = ivy.Container(a=ivy.array([1.0, 2.0, 3.0]),
        ...                    b=ivy.array([1, 4, 5]))
        >>> x2 = ivy.Container(a=ivy.array([1, 2, 3.0]),
        ...                    b=ivy.array([1.0, 4.0, 5.0]))
        >>> y = x1 == x2
        >>> print(y)
        {
            a: ivy.array([True, True, True]),
            b: ivy.array([True, True, True])
        }
        """
        if isinstance(other, ivy.Container):
            return ivy.Container.cont_multi_map(
                lambda xs, _: operator.eq(xs[0], xs[1]), [self, other], map_nests=True
            )
        return self.cont_map(lambda x, kc: x == other, map_sequences=True)

    def __ne__(self, other):
        """ivy.Container special method for the not_equal operator, calling
        :code:`operator.ne` for each of the corresponding leaves of the two
        containers.

        Parameters
        ----------
        self
            first input Container. May have any data type.
        other
            second input Container. Must be compatible with x1 (with Broadcasting).
            May have any data type.

        Returns
        -------
        ret
            A container containing the element-wise results. Any returned array inside
            must have a data type of bool.

        Examples
        --------
        With :class:`ivy.Container` instances:

        >>> x1 = ivy.Container(a=ivy.array([1, 2, 3]),
        ...                    b=ivy.array([1, 3, 5]))
        >>> x2 = ivy.Container(a=ivy.array([1, 2, 3]),
        ...                    b=ivy.array([1, 4, 5]))
        >>> y = x1 != x2
        >>> print(y)
        {
            a: ivy.array([False, False, False]),
            b: ivy.array([False, True, False])
        }

        >>> x1 = ivy.Container(a=ivy.array([1.0, 2.0, 3.0]),
        ...                    b=ivy.array([1, 4, 5]))
        >>> x2 = ivy.Container(a=ivy.array([1, 3, 3.0]),
        ...                    b=ivy.array([1.0, 4.0, 5.0]))
        >>> y = x1 != x2
        >>> print(y)
        {
            a: ivy.array([False, True, False]),
            b: ivy.array([False, False, False])
        }

        >>> x1 = ivy.Container(a=ivy.array([1.0, 2.0, 3.0]),
        ...                    b=ivy.array([1, 4, 5]))
        >>> x2 = ivy.Container(a=ivy.array([1, 2, 3.0]),
        ...                    b=ivy.array([1.0, 4.0, 5.0]))
        >>> y = x1 != x2
        >>> print(y)
        {
            a: ivy.array([False, False, False]),
            b: ivy.array([False, False, False])
        }
        """
        if isinstance(other, ivy.Container):
            return ivy.Container.cont_multi_map(
                lambda xs, _: operator.ne(xs[0], xs[1]), [self, other], map_nests=True
            )
        return self.cont_map(lambda x, kc: x != other, map_sequences=True)

    def __gt__(self, other):
        """ivy.Container special method for the greater operator, calling
        :code:`operator.gt` for each of the corresponding leaves of the two
        containers.

        Parameters
        ----------
        self
            first input Container. May have any data type.
        other
            second input Container. Must be compatible with x1 (with Broadcasting).
            May have any data type.

        Returns
        -------
        ret
            A container containing the element-wise results. Any returned array inside
            must have a data type of bool.

        Examples
        --------
        With :class:`ivy.Container` instances:

        >>> x = ivy.Container(a=ivy.array([4, 5, 6]),b=ivy.array([2, 3, 4]))
        >>> y = ivy.Container(a=ivy.array([1, 5, 3]),b=ivy.array([5, 3, 7]))
        >>> z = x > y
        >>> print(z)
        {
            a:ivy.array([True,False,True]),
            b:ivy.array([False,False,False])
        }
        """
        if isinstance(other, ivy.Container):
            return ivy.Container.cont_multi_map(
                lambda xs, _: operator.gt(xs[0], xs[1]), [self, other], map_nests=True
            )
        return self.cont_map(lambda x, kc: x > other, map_sequences=True)

    def __ge__(self, other):
        """ivy.Container special method for the greater_equal operator, calling
        :code:`operator.ge` for each of the corresponding leaves of the two
        containers.

        Parameters
        ----------
        self
            first input Container. May have any data type.
        other
            second input Container. Must be compatible with x1 (with Broadcasting).
            May have any data type.

        Returns
        -------
        ret
            A container containing the element-wise results. Any returned array inside
            must have a data type of bool.

        Examples
        --------
        With :class:`ivy.Container` instances:

        >>> x = ivy.Container(a=ivy.array([4, 5, 6]),b=ivy.array([2, 3, 4]))
        >>> y = ivy.Container(a=ivy.array([1, 5, 3]),b=ivy.array([5, 3, 7]))
        >>> z = x >= y
        >>> print(z)
        {
            a:ivy.array([True,True,True]),
            b:ivy.array([False,True,False])
        }
        """
        if isinstance(other, ivy.Container):
            return ivy.Container.cont_multi_map(
                lambda xs, _: operator.ge(xs[0], xs[1]), [self, other], map_nests=True
            )
        return self.cont_map(lambda x, kc: x >= other, map_sequences=True)

    def __and__(self, other):
        if isinstance(other, ivy.Container):
            return ivy.Container.cont_multi_map(
                lambda xs, _: operator.and_(xs[0], xs[1]), [self, other], map_nests=True
            )
        return self.cont_map(lambda x, kc: x and other, map_sequences=True)

    def __iand__(self, other):
        if isinstance(other, ivy.Container):
            return ivy.Container.cont_multi_map(
                lambda xs, _: operator.iand(xs[0], xs[1]), [self, other], map_nests=True
            )
        return self.cont_map(lambda x, kc: operator.iand(x, other), map_sequences=True)

    def __rand__(self, other):
        return self.cont_map(lambda x, kc: other and x, map_sequences=True)

    def __or__(self, other):
        if isinstance(other, ivy.Container):
            return ivy.Container.cont_multi_map(
                lambda xs, _: operator.or_(xs[0], xs[1]), [self, other], map_nests=True
            )
        return self.cont_map(lambda x, kc: x or other, map_sequences=True)

    def __ror__(self, other):
        return self.cont_map(lambda x, kc: other or x, map_sequences=True)

    def __ior__(self, other):
        return ivy.Container.cont_multi_map(
            lambda xs, _: operator.ior(xs[0], xs[1]), [self, other], map_nests=True
        )

    def __invert__(self):
        return self.cont_map(lambda x, kc: operator.not_(x), map_sequences=True)

    def __xor__(self, other):
        """ivy.Container special method for the ge operator, calling
        :code:`operator.ge` for each of the corresponding leaves of the two
        containers.

        Parameters
        ----------
        self
            first input Container.
        other
            second input Container. Arrays inside must be compatible with ``x1``
            (see :ref:`broadcasting`). Should have an integer or boolean data type.

        Returns
        -------
        ret
            a container containing the element-wise results. Any returned arrays inside
            must have a data type determined by :ref:`type-promotion`.

        Examples
        --------
        With :class:`ivy.Container` instances:

        >>> x = ivy.Container(a=ivy.array([89]), b=ivy.array([2]))
        >>> y = ivy.Container(a=ivy.array([12]), b=ivy.array([3]))
        >>> z = x ^ y
        >>> print(z)
        {
            a: ivy.array([85]),
            b: ivy.array([1])
        }
        """
        if isinstance(other, ivy.Container):
            return ivy.Container.cont_multi_map(
                lambda xs, _: operator.xor(xs[0], xs[1]), [self, other], map_nests=True
            )
        return self.cont_map(lambda x, kc: operator.xor(x, other), map_sequences=True)

    def __rxor__(self, other):
        return self.cont_map(lambda x, kc: other != x, map_sequences=True)

    def __ixor__(self, other):
        if isinstance(other, ivy.Container):
            return ivy.Container.cont_multi_map(
                lambda xs, _: operator.ixor(xs[0], xs[1]), [self, other], map_nests=True
            )
        return self.cont_map(lambda x, kc: operator.ixor(x, other), map_sequences=True)

    def __lshift__(self, other):
        if isinstance(other, ivy.Container):
            return ivy.Container.cont_multi_map(
                lambda xs, _: operator.lshift(xs[0], xs[1]),
                [self, other],
                map_nests=True,
            )
        return self.cont_map(
            lambda x, kc: operator.lshift(x, other), map_sequences=True
        )

    def __rlshift__(self, other):
        return self.cont_map(
            lambda x, kc: operator.lshift(other, x), map_sequences=True
        )

    def __ilshift__(self, other):
        if isinstance(other, ivy.Container):
            return ivy.Container.cont_multi_map(
                lambda xs, _: operator.ilshift(xs[0], xs[1]),
                [self, other],
                map_nests=True,
            )
        return self.cont_map(
            lambda x, kc: operator.ilshift(x, other), map_sequences=True
        )

    def __rshift__(self, other):
        """ivy.Container special method for the right shift operator, calling
        :code:`operator.rshift` for each of the corresponding leaves of the two
        containers.

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

        With :class:`ivy.Array` instances at the leaves:

        >>> x = ivy.Container(a=ivy.array([16, 40, 120]),
        ...                   b=ivy.array([15, 45, 143]))
        >>> y = ivy.Container(a=ivy.array([1, 2, 3]),
        ...                   b=ivy.array([0, 3, 4]))
        >>> z = x >> y
        >>> print(z)
        {
            a: ivy.array([8, 10, 15]),
            b: ivy.array([15, 5, 8])
        }

        With a mix of :class:`ivy.Container` and :class:`ivy.Array` instances:

        >>> x = ivy.Container(a=ivy.array([16, 40, 120]),
        ...                   b=ivy.array([15, 45, 143]))
        >>> y = ivy.array([1, 2, 3])
        >>> z = x >> y
        >>> print(z)
        {
            a: ivy.array([8, 10, 15]),
            b: ivy.array([7, 11, 17])
        }
        """
        if isinstance(other, ivy.Container):
            return ivy.Container.cont_multi_map(
                lambda xs, _: operator.rshift(xs[0], xs[1]),
                [self, other],
                map_nests=True,
            )
        return self.cont_map(
            lambda x, kc: operator.rshift(x, other), map_sequences=True
        )

    def __rrshift__(self, other):
        """ivy.Container reverse special method for the right shift operator,
        calling :code:`operator.rshift` for each of the corresponding leaves of
        the two containers.

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
        >>> b = ivy.Container(a = ivy.array([0, 1, 2]),
        ...                   b = ivy.array([3, 4, 5]))
        >>> y = a >> b
        >>> print(y)
        {
            a: ivy.array([64, 32, 16]),
            b: ivy.array([8, 4, 2])
        }
        """
        return self.cont_map(lambda x, kc: other >> x, map_sequences=True)

    def __irshift__(self, other):
        if isinstance(other, ivy.Container):
            return ivy.Container.cont_multi_map(
                lambda xs, _: operator.irshift(xs[0], xs[1]),
                [self, other],
                map_nests=True,
            )
        return self.cont_map(
            lambda x, kc: operator.irshift(x, other), map_sequences=True
        )
