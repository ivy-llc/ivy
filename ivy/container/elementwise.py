# global
from typing import Optional, Union, List, Dict

# local
import ivy
from ivy.container.base import ContainerBase


# noinspection PyMissingConstructor
class ContainerWithElementwise(ContainerBase):
    @staticmethod
    def static_abs(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.abs. This method simply wraps the
        function, and so the docstring for ivy.abs also applies to this method
        with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a numeric data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the absolute value of each element in ``x``. The
            returned container must have the same data type as ``x``.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0., 2.6, -3.5]),\
                            b=ivy.array([4.5, -5.3, -0, -2.3]))
        >>> y = ivy.Container.static_abs(x)
        >>> print(y)
        {
            a: ivy.array([0., 2.6, 3.5]),
            b: ivy.array([4.5, 5.3, 0, 2.3])
        }

        """
        return ContainerBase.multi_map_in_static_method(
            "abs",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def abs(
        self: ivy.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.abs. This method simply wraps the
        function, and so the docstring for ivy.abs also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a numeric data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the absolute value of each element in ``self``. The
            returned container must have the same data type as ``self``.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([-1.6, 2.6, -3.5]),\
                            b=ivy.array([4.5, -5.3, -2.3]))
        >>> y = x.abs()
        >>> print(y)
        {
            a: ivy.array([1.6, 2.6, 3.5]),
            b: ivy.array([4.5, 5.3, 2.3])
        }

        """
        return self.static_abs(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_acosh(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.cosh. This method simply wraps the
        function, and so the docstring for ivy.cosh also applies to this method
        with minimal changes.

        Parameters
        ----------
        x
            input container whose elements each represent the area of a hyperbolic
            sector. Should have a floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the inverse hyperbolic cosine of each element
            in ``x``. The returned container must have a floating-point data
            type determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1., 2., 3, 4]),\
                              b=ivy.array([1., 3., 10.0, 6]))
        >>> y = ivy.Container.static_acosh(x)
        >>> print(y)
        {
            a: ivy.array([0., 1.32, 1.76, 2.06]),
            b: ivy.array([0., 1.76, 2.99, 2.48])
        }

        """
        return ContainerBase.multi_map_in_static_method(
            "acosh",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def acosh(
        self: ivy.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.acosh.
        This method simply wraps the function, and so the docstring for
        ivy.acosh also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container whose elements each represent the area of a hyperbolic
            sector. Should have a floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the inverse hyperbolic cosine of each element in
            ``self``. The returned container must have a floating-point data
            type determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1., 2., 3, 4]),\
                              b=ivy.array([1., 3., 10.0, 6]))
        >>> y = x.acosh()
        >>> print(y)
        {
            a: ivy.array([0., 1.32, 1.76, 2.06]),
            b: ivy.array([0., 1.76, 2.99, 2.48])
        }

        """
        return self.static_acosh(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_acos(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.acos.
        This method simply wraps the function, and so the docstring for
        ivy.acos also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a real-valued floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the inverse cosine of each element in ``x``.
            The returned container must have a floating-point data type
            determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0., -1, 1]), b=ivy.array([1., 0., -1.]))
        >>> y = ivy.Container.static_acos(x)
        >>> print(y)
        {
            a: ivy.array([1.57, 3.14, 0.]),
            b: ivy.array([0., 1.57, 3.14])
        }

        """
        return ContainerBase.multi_map_in_static_method(
            "acos",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_add(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.add.
        This method simply wraps the function, and so the docstring for
        ivy.add also applies to this method with minimal changes.

        Parameters
        ----------
        x1
            first input array or container. Should have a numeric data type.
        x2
            second input array or container. Must be compatible with ``x1``
            (see :ref:`broadcasting`). Should have a numeric data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise sums.
            The returned container must have a data type determined
            by :ref:`type-promotion`.

        Examples
        --------
        With one :code:`ivy.Container` input:

        >>> x = ivy.array([[1.1, 2.3, -3.6]])
        >>> y = ivy.Container(a=ivy.array([[4.], [5.], [6.]]),\
                            b=ivy.array([[5.], [6.], [7.]]))
        >>> z = ivy.Container.static_add(x, y)
        >>> print(z)
        {
            a: ivy.array([[5.1, 6.3, 0.4],
                          [6.1, 7.3, 1.4],
                          [7.1, 8.3, 2.4]]),
            b: ivy.array([[6.1, 7.3, 1.4],
                          [7.1, 8.3, 2.4],
                          [8.1, 9.3, 3.4]])
        }

        With multiple :code:`ivy.Container` inputs:

        >>> x = ivy.Container(a=ivy.array([1, 2, 3]), \
                            b=ivy.array([2, 3, 4]))
        >>> y = ivy.Container(a=ivy.array([4, 5, 6]),\
                            b=ivy.array([5, 6, 7]))
        >>> z = ivy.Container.static_add(x, y)
        >>> print(z)
        {
            a: ivy.array([5, 7, 9]),
            b: ivy.array([7, 9, 11])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "add",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def acos(
        self: ivy.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.acos.
        This method simply wraps the function, and so the docstring for
        ivy.acos also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a real-valued floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the inverse cosine of each element in ``self``.
            The returned container must have a floating-point data type
            determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0., -1, 1]), b=ivy.array([1., 0., -1.]))
        >>> y = x.acos()
        >>> print(y)
        {
            a: ivy.array([1.57, 3.14, 0.]),
            b: ivy.array([0., 1.57, 3.14])
        }

        """
        return self.static_acos(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def add(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.add.
        This method simply wraps the function, and so the docstring for
        ivy.add also applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input container. Should have a numeric data type.
        x2
            second input array or container. Must be compatible with ``self``
            (see :ref:`broadcasting`). Should have a numeric data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise sums.
            The returned container must have a data type determined
            by :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1, 2, 3]),\
                             b=ivy.array([2, 3, 4]))
        >>> y = ivy.Container(a=ivy.array([4, 5, 6]),\
                             b=ivy.array([5, 6, 7]))

        >>> z = x.add(y)
        >>> print(z)
        {
            a: ivy.array([5, 7, 9]),
            b: ivy.array([7, 9, 11])
        }
        """
        return self.static_add(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_asin(
        x: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.asin.
        This method simply wraps the function, and so the docstring for
        ivy.asin also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a real-valued floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the inverse sine of each element in ``x``.
            The returned container must have a floating-point data
            type determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0., -0.5, -1.]),\
                              b=ivy.array([0.1, 0.8, 2.]))
        >>> y = ivy.Container.static_asin()
        >>> print(y)
        {
            a: ivy.array([0., -0.524, -1.57]),
            b: ivy.array([0.1, 0.927, nan])
        }

        >>> x = ivy.Container(a=ivy.array([0.4, 0.9, -0.9]),\
                              b=ivy.array([[4, -3, -0.2]))
        >>> y = ivy.Container(a=ivy.zeros(3), b=ivy.zeros(3))
        >>> ivy.Container.static_asin(out=y)
        >>> print(y)
        {
            a: ivy.array([0.412, 1.12, -1.12]),
            b: ivy.array([nan, nan, -0.201])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "asin",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def asin(
        self: ivy.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.asin.
        This method simply wraps the function, and so the docstring for
        ivy.asin also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a real-valued floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the inverse sine of each element in ``self``.
            The returned container must have a floating-point
            data type determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0., 0.5, 1.]),\
                              b=ivy.array([-4., 0.8, 2.]))
        >>> y = x.asin()
        >>> print(y)
        {
            a: ivy.array([0., 0.524, 1.57]),
            b: ivy.array([nan, 0.927, nan])
        }

        >>> x = ivy.Container(a=ivy.array([12., 1.5, 0.]),\
                              b=ivy.array([-0.85, 0.6, 0.3]))
        >>> y = ivy.Container(a=ivy.zeros(3), b=ivy.zeros(3))
        >>> x.asin(out=y)
        >>> print(y)
        {
            a: ivy.array([nan, nan, 0.]),
            b: ivy.array([-1.02, 0.644, 0.305])
        }
        """
        return self.static_asin(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_asinh(
        x: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.asinh.
        This method simply wraps the function, and so the docstring for
        ivy.asinh also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container whose elements each represent the area of a hyperbolic
            sector. Should have a floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the inverse hyperbolic sine of each element
            in ``x``. The returned container must have a floating-point data
            type determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1.5, 0., -3.5]),\
                            b=ivy.array([3.4, -5.3, -0, -2.8]))
        >>> y = ivy.Container.static_asinh(x)
        >>> print(y)
        {
            a: ivy.array([1.19, 0., -1.97]),
            b: ivy.array([1.94, -2.37, 0., -1.75])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "asinh",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def asinh(
        self: ivy.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.asinh.
        This method simply wraps the function, and so the docstring
        for ivy.asinh also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container whose elements each represent the area of a hyperbolic
            sector. Should have a floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the inverse hyperbolic sine of each element in
            ``self``. The returned container must have a floating-point
            data type determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([-1, 3.7, -5.1]),\
                            b=ivy.array([4.5, -2.4, -1.5]))
        >>> y = x.asinh()
        >>> print(y)
        {
            a: ivy.array([-0.881, 2.02, -2.33]),
            b: ivy.array([2.21, -1.61, -1.19])
        }
        """
        return self.static_asinh(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_atan(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.atan. This method simply wraps the
        function, and so the docstring for ivy.atan also applies to this method
        with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a real-valued floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the inverse tangent of each element in ``x``.
            The returned container must have a floating-point data type
            determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0., -1, 1]), b=ivy.array([1., 0., -6]))
        >>> y = ivy.Container.static_atan(x)
        >>> print(y)
        {
            a: ivy.array([0., -0.785, 0.785]),
            b: ivy.array([0.785, 0., -1.41])
        }


        """
        return ContainerBase.multi_map_in_static_method(
            "atan",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def atan(
        self: ivy.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.atan.
        This method simply wraps the function, and so the docstring for
        ivy.atan also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a real-valued floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the inverse tangent of each element in ``x``.
            The returned container must have a floating-point data
            type determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0., -1, 1]), b=ivy.array([1., 0., -6]))
        >>> y = x.atan()
        >>> print(y)
        {
            a: ivy.array([0., -0.785, 0.785]),
            b: ivy.array([0.785, 0., -1.41])
        }

        """
        return self.static_atan(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_atan2(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.atan2.
        This method simply wraps the function, and so the docstring for
        ivy.atan2 also applies to this method with minimal changes.

        Parameters
        ----------
        x1
            first input array or container corresponding to the y-coordinates.
            Should have a real-valued floating-point data type.
        x2
            second input array or container corresponding to the x-coordinates.
            Must be compatible with ``x1``
            (see :ref:`broadcasting`). Should have a real-valued
            floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the inverse tangent of the quotient ``x1/x2``.
            The returned array must have a real-valued floating-point data type
            determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0., 2.6, -3.5]),\
                            b=ivy.array([4.5, -5.3, -0]))
        >>> y = ivy.array([3.0, 2.0, 1.0])
        >>> ivy.Container.static_atan2(x, y)
        {
            a: ivy.array([0., 0.915, -1.29]),
            b: ivy.array([0.983, -1.21, 0.])
        }

        >>> x = ivy.Container(a=ivy.array([0., 2.6, -3.5]),\
                              b=ivy.array([4.5, -5.3, -0, -2.3]))
        >>> y = ivy.Container(a=ivy.array([-2.5, 1.75, 3.5]),\
                              b=ivy.array([2.45, 6.35, 0, 1.5]))
        >>> z = ivy.Container.static_atan2(x, y)
        >>> print(z)
        {
            a: ivy.array([3.14, 0.978, -0.785]),
            b: ivy.array([1.07, -0.696, 0., -0.993])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "atan2",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def atan2(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.atan2.
        This method simply wraps the function, and so the docstring for ivy.atan2
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array or container corresponding to the y-coordinates.
            Should have a real-valued floating-point data type.
        x2
            second input array or container corresponding to the x-coordinates.
            Must be compatible with ``self`` (see :ref:`broadcasting`).
            Should have a real-valued floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the inverse tangent of the quotient ``self/x2``.
            The returned array must have a real-valued floating-point data
            type determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0., 2.6, -3.5]),\
                            b=ivy.array([4.5, -5.3, -0]))
        >>> y = ivy.array([3.0, 2.0, 1.0])
        >>> x.atan2(y)
        {
            a: ivy.array([0., 0.915, -1.29]),
            b: ivy.array([0.983, -1.21, 0.])
        }

        >>> x = ivy.Container(a=ivy.array([0., 2.6, -3.5]),\
                              b=ivy.array([4.5, -5.3, -0, -2.3]))
        >>> y = ivy.Container(a=ivy.array([-2.5, 1.75, 3.5]),\
                              b=ivy.array([2.45, 6.35, 0, 1.5]))
        >>> z = x.atan2(y)
        >>> print(z)
        {
            a: ivy.array([3.14, 0.978, -0.785]),
            b: ivy.array([1.07, -0.696, 0., -0.993])
        }
        """
        return self.static_atan2(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_atanh(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.atanh.
        This method simply wraps the function, and so the docstring for
        ivy.atanh also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container whose elements each represent the area of a hyperbolic
            sector. Should have a floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the inverse hyperbolic tangent of each
            element in ``x``. The returned container must have a floating-point data
            type determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0, 0.5, -0.5]), b=ivy.array([0., 0.2, 0.9]))
        >>> y = ivy.Container.static_atanh(x)
        >>> print(y)
        {
            a: ivy.array([0., 0.549, -0.549]),
            b: ivy.array([0., 0.203, 1.47])
        }

        """
        return ContainerBase.multi_map_in_static_method(
            "atanh",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def atanh(
        self: ivy.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.atanh.
        This method simply wraps the function, and so the docstring for
        ivy.atanh also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container whose elements each represent the area of a
            hyperbolic sector. Should have a floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the inverse hyperbolic tangent of each element
            in ``self``. The returned container must have a floating-point
            data type determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0, 0.5, -0.5]), b=ivy.array([0., 0.2, 0.9]))
        >>> y = x.atanh()
        >>> print(y)
        {
            a: ivy.array([0., 0.549, -0.549]),
            b: ivy.array([0., 0.203, 1.47])
        }

        """
        return self.static_atanh(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_bitwise_and(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.bitwise_and.
        This method simply wraps the function, and so the docstring for
        ivy.bitwise_and also applies to this method with minimal changes.

        Parameters
        ----------
        x1
            first input array or container. Should have an integer or boolean
            data type.
        x2
            second input array or container Must be compatible with ``x1``
            (see :ref:`broadcasting`).
            Should have an integer or boolean data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise results.
            The returned container must have a data type determined
            by :ref:`type-promotion`.
        """
        return ContainerBase.multi_map_in_static_method(
            "bitwise_and",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def bitwise_and(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.bitwise_and.
        This method simply wraps the function, and so the docstring for
        ivy.bitwise_and also applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array or container. Should have an integer or boolean
            data type.
        x2
            second input array or container Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have an integer or boolean data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise results.
            The returned container must have a data type determined
            by :ref:`type-promotion`.
        """
        return self.static_bitwise_and(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_bitwise_left_shift(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.bitwise_left_shift.
        This method simply wraps the function, and so the docstring for
        ivy.bitwise_left_shift also applies to this method with minimal changes.

        Parameters
        ----------
        x1
            first input array or container. Should have an integer or boolean
            data type.
        x2
            second input array or container Must be compatible with ``x1``
            (see :ref:`broadcasting`).
            Should have an integer or boolean data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise results.
            The returned container must have a data type determined by
            :ref:`type-promotion`.
        """
        return ContainerBase.multi_map_in_static_method(
            "bitwise_left_shift",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def bitwise_left_shift(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.bitwise_left_shift.
        This method simply wraps the function, and so the docstring for
        ivy.bitwise_left_shift also applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array or container. Should have an integer or boolean
            data type.
        x2
            second input array or container Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have an integer or boolean data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise results. The returned container
            must have a data type determined by :ref:`type-promotion`.
        """
        return self.static_bitwise_left_shift(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_bitwise_invert(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.bitwise_invert.
        This method simply wraps the function, and so the docstring for
        ivy.bitwise_invert also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have an integer or boolean data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise results.
            The returned array must have the same data type as ``x``.
        """
        return ContainerBase.multi_map_in_static_method(
            "bitwise_invert",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def bitwise_invert(
        self: ivy.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.bitwise_invert.
        This method simply wraps the function, and so the docstring for
        ivy.bitwise_invert also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have an integer or boolean data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise results.
            The returned array must have the same data type as ``self``.
        """
        return self.static_bitwise_invert(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_cos(
        x: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.cos.
        This method simply wraps the function, and so the docstring for
        ivy.cos also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container whose elements are each expressed in radians.
            Should have a floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the cosine of each element in ``x``. The returned
            container must have a floating-point data type determined by
            :ref:`type-promotion`.

        Examples
        --------
        With :code:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([0., -1, 1]), b=ivy.array([1., 0., -6]))
        >>> y = ivy.Container.static_cos(x)
        >>> print(y)
        {
            a: ivy.array([1., 0.54, 0.54]),
            b: ivy.array([0.54, 1., 0.96])
        }
        """
        return ivy.ContainerBase.multi_map_in_static_method(
            "cos",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def cos(
        self: ivy.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.cos.
        This method simply wraps the function, and so the docstring for
        ivy.cos also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container whose elements are each expressed in radians.
            Should have a floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the cosine of each element in ``self``.
            The returned container must have a floating-point data type
            determined by :ref:`type-promotion`.

        Examples
        --------
        With :code:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([0., -1, 1]), b=ivy.array([1., 0., -6]))
        >>> y = x.cos()
        >>> print(y)
        {
            a: ivy.array([1., 0.54, 0.54]),
            b: ivy.array([0.54, 1., 0.96])
        }
        """
        return self.static_cos(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_bitwise_or(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.bitwise_or. This method simply
        wraps the function, and so the docstring for ivy.bitwise_or also applies
        to this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have an integer or boolean data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise results.
            The returned array must have the same data type as ``x``.

        Examples
        --------
        With one :code:`ivy.Container` input:

        >>> y = ivy.array([1, 2, 3])
        >>> x = ivy.Container(a=ivy.array([4, 5, 6]))
        >>> z = ivy.Container.static_bitwise_or(x, y)
        >>> print(z)
        {
            a: ivy.array([5, 7, 7]),
        }

        With multiple :code:`ivy.Container` inputs:

        >>> x = ivy.Container(a=ivy.array([1, 2, 3]), \
                            b=ivy.array([2, 3, 4]))
        >>> y = ivy.Container(a=ivy.array([4, 5, 6]),\
                            b=ivy.array([5, 6, 7]))
        >>> z = ivy.Container.static_bitwise_or(x, y)
        >>> print(z)
        {
            a: ivy.array([5, 7, 7]),
            b: ivy.array([7, 7, 7])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "bitwise_or",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def bitwise_or(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.bitwise_or. This method simply
        wraps the function, and so the docstring for ivy.bitwise_or also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have an integer or boolean data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise results.
            The returned array must have the same data type as ``self``.

        Examples
        --------
        Using :code:`ivy.Container` instance method:

        >>> x = ivy.Container(a=ivy.array([1, 2, 3]), \
                                b=ivy.array([2, 3, 4]))
        >>> y = ivy.Container(a=ivy.array([4, 5, 6]), \
                                b=ivy.array([5, 6, 7]))
        >>> z = x.bitwise_or(y)
        >>> print(z)
        {
            a: ivy.array([5, 7, 7]),
            b: ivy.array([7, 7, 7])
        }
        """
        return self.static_bitwise_or(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_bitwise_right_shift(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.bitwise_right_shift.
        This method simply wraps the function, and so the docstring for
        ivy.bitwise_right_shift also applies to this method with minimal changes.

        Parameters
        ----------
        x1
            first input array or container. Should have an integer or boolean data type.
        x2
            second input array or container Must be compatible with ``x1``
            (see :ref:`broadcasting`).
            Should have an integer or boolean data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise results.
            The returned container must have a data type determined
            by :ref:`type-promotion`.
        """
        return ContainerBase.multi_map_in_static_method(
            "bitwise_right_shift",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def bitwise_right_shift(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.bitwise_right_shift.
        This method simply wraps the function, and so the docstring for
        ivy.bitwise_right_shift also applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array or container. Should have an integer or boolean data type.
        x2
            second input array or container Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have an integer or boolean data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise results. The returned container
            must have a data type determined by :ref:`type-promotion`.
        """
        return self.static_bitwise_right_shift(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_bitwise_xor(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.bitwise_xor.
        This method simply wraps the function, and so the docstring for
        ivy.bitwise_xor also applies to this method with minimal changes.

        Parameters
        ----------
        x1
            first input array or container. Should have an integer or boolean
            data type.
        x2
            second input array or container Must be compatible with ``x1``
            (see :ref:`broadcasting`).
            Should have an integer or boolean data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise results.
            The returned container must have a data type determined by
            :ref:`type-promotion`.
        """
        return ContainerBase.multi_map_in_static_method(
            "bitwise_xor",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def bitwise_xor(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.bitwise_xor.
        This method simply wraps the function, and so the docstring for ivy.bitwise_xor
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array or container. Should have an integer or
            boolean data type.
        x2
            second input array or container Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have an integer or boolean data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise results.
            The returned container must have a data type determined
            by :ref:`type-promotion`.
        """
        return self.static_bitwise_xor(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_ceil(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.ceil.
        This method simply wraps the function, and so the docstring for ivy.ceil
        also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a numeric data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            an container containing the rounded result for each element in ``x``.
            The returned array must have the same data type as ``x``.
        """
        return ContainerBase.multi_map_in_static_method(
            "ceil",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def ceil(
        self: ivy.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.ceil.
        This method simply wraps the function, and so the docstring for
        ivy.ceil also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a numeric data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            an container containing the rounded result for each element in ``self``.
            The returned container must have the same data type as ``self``.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([2.5, 0.5, -1.4]),\
                              b=ivy.array([5.4, -3.2, 5.2]))
        >>> y = x.ceil()
        >>> print(y)
        {
            a: ivy.array([3., 1., -1.]),
            b: ivy.array([6., -3., 6.])
        }
        """
        return self.static_ceil(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_cosh(
        x: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.cosh. This method simply wraps the
        function, and so the docstring for ivy.cosh also applies to this method
        with minimal changes.

        Parameters
        ----------
        x
            input container whose elements each represent a hyperbolic angle. Should
            have a floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            an container containing the hyperbolic cosine of each element in ``x``. The
            returned container must have a floating-point data type determined by
            :ref:`type-promotion`.

        Examples
        --------
        With :code:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([-1, 0.23, 1.12]), b=ivy.array([1, -2, 0.76]))
        >>> y = ivy.Container.static_cosh(x)
        >>> print(y)
        {
            a: ivy.array([1.54, 1.03, 1.7]),
            b: ivy.array([1.54, 3.76, 1.3])
        }

        >>> x = ivy.Container(a=ivy.array([-3, 0.34, 2.]),\
                    b=ivy.array([0.67, -0.98, -3]))
        >>> y = ivy.Container(a=ivy.zeros(1), b=ivy.zeros(1))
        >>> ivy.Container.static_cosh(x, out=y)
        >>> print(y)
        {
            a: ivy.array([10.1, 1.06, 3.76]),
            b: ivy.array([1.23, 1.52, 10.1])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "cosh",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def cosh(
        self: ivy.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.cosh. This method simply wraps the
        function, and so the docstring for ivy.cosh also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input container whose elements each represent a hyperbolic angle. Should
            have a floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            an container containing the hyperbolic cosine of each element in ``self``.
            The returned container must have a floating-point data type determined by
            :ref:`type-promotion`.

        Examples
        --------
        With :code:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([-1, 0.23, 1.12]), b=ivy.array([1, -2, 0.76]))
        >>> y = x.cosh()
        >>> print(y)
        {
            a: ivy.array([1.54, 1.03, 1.7]),
            b: ivy.array([1.54, 3.76, 1.3])
        }

        >>> x = ivy.Container(a=ivy.array([-3, 0.34, 2.]),\
                    b=ivy.array([0.67, -0.98, -3]))
        >>> y = ivy.Container(a=ivy.zeros(1), b=ivy.zeros(1))
        >>> ivy.Container.cosh(x, out=y)
        >>> print(y)
        {
            a: ivy.array([10.1, 1.06, 3.76]),
            b: ivy.array([1.23, 1.52, 10.1])
        }
        """
        return self.static_cosh(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_divide(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.divide. This method simply wraps
        the function, and so the docstring for ivy.divide also applies to this method
        with minimal changes.

        Parameters
        ----------
        x1
            dividend input array or container. Should have a real-valued data type.
        x2
            divisor input array or container. Must be compatible with ``x1``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise results.
            The returned container must have a data type determined
            by :ref:`type-promotion`.

        Examples
        --------
        With :code:`ivy.Container` inputs:

        >>> x1 = ivy.Container(a=ivy.array([12., 3.5, 6.3]), b=ivy.array([3., 1., 0.9]))
        >>> x2 = ivy.Container(a=ivy.array([1., 2.3, 3]), b=ivy.array([2.4, 3., 2.]))
        >>> y = ivy.Container.static_divide(x1, x2)
        >>> print(y)
        {
            a: ivy.array([12., 1.52, 2.1]),
            b: ivy.array([1.25, 0.333, 0.45])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "divide",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def divide(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.divide.
        This method simply wraps the function, and so the docstring for
        ivy.divide also applies to this method with minimal changes.

        Parameters
        ----------
        self
            dividend input array or container. Should have a real-valued
            data type.
        x2
            divisor input array or container. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise results.
            The returned container must have a data type determined
            by :ref:`type-promotion`.

        Examples
        --------
        With :code:`ivy.Container` inputs:

        >>> x1 = ivy.Container(a=ivy.array([12., 3.5, 6.3]), b=ivy.array([3., 1., 0.9]))
        >>> x2 = ivy.Container(a=ivy.array([1., 2.3, 3]), b=ivy.array([2.4, 3., 2.]))
        >>> y = x1.divide(x2)
        >>> print(y)
        {
            a: ivy.array([12., 1.52, 2.1]),
            b: ivy.array([1.25, 0.333, 0.45])
        }
        """
        return self.static_divide(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_equal(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.equal.
        This method simply wraps the function, and so the docstring for
        ivy.equal also applies to this method with minimal changes.

        Parameters
        ----------
        x1
            input array or container. May have any data type.
        x2
            input array or container. Must be compatible with ``x1``
            (see :ref:`broadcasting`).
            May have any data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise results. The returned container
            must have a data type of ``bool``.
        """
        return ContainerBase.multi_map_in_static_method(
            "equal",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def equal(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.equal.
        This method simply wraps the function, and so the docstring for
        ivy.equal also applies to this method with minimal changes.

        Parameters
        ----------
        x1
            input array or container. May have any data type.
        x2
            input array or container. Must be compatible with ``x1``
            (see :ref:`broadcasting`).
            May have any data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise results. The returned container
            must have a data type of ``bool``.

        With :code:`ivy.Container` inputs:

        >>> x1 = ivy.Container(a=ivy.array([12, 3.5, 6.3]), b=ivy.array([3., 1., 0.9]))
        >>> x2 = ivy.Container(a=ivy.array([12, 2.3, 3]), b=ivy.array([2.4, 3., 2.]))
        >>> y = x1.equal(x2)
        >>> print(y)
        {
            a: ivy.array([True, False, False]),
            b: ivy.array([False, False, False])
        }

        With mixed :code:`ivy.Container` and :code:`ivy.Array` inputs:

        >>> x1 = ivy.Container(a=ivy.array([12., 3.5, 6.3]), b=ivy.array([3., 1., 0.9]))
        >>> x2 = ivy.array([3., 1., 0.9])
        >>> y = x1.equal(x2)
        >>> print(y)
        {
            a: ivy.array([False, False, False]),
            b: ivy.array([True, True, True])
        }
        """
        return self.static_equal(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_exp(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.exp. This method simply
        wraps the function, and so the docstring for ivy.exp also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the evaluated result for each element in ``x``.
            The returned array must have a real-valued floating-point data type
            determined by :ref:`type-promotion`.

        """
        return ContainerBase.multi_map_in_static_method(
            "exp",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def exp(
        self: ivy.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.exp.
        This method simply wraps the function, and so the docstring
        for ivy.exp also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the evaluated result for each element in ``self``.
            The returned array must have a real-valued floating-point data type
            determined by :ref:`type-promotion`.

        """
        return self.static_exp(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_expm1(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.expm1.
        This method simply wraps thefunction, and so the docstring
        for ivy.expm1 also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the evaluated result for each element in ``x``.
            The returned array must have areal-valued floating-point data type
            determined by :ref:`type-promotion`.

        """
        return ContainerBase.multi_map_in_static_method(
            "expm1",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def expm1(
        self: ivy.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.expm1.
        This method simply wraps the function, and so the docstring
        for ivy.expm1 also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the evaluated result for each element in ``self``.
            The returned array must have a real-valued floating-point data type
            determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([2.5, 0.5]),\
                              b=ivy.array([5.4, -3.2]))
        >>> y = x.expm1()
        >>> print(y)
        {
            a: ivy.array([11.2, 0.649]),
            b: ivy.array([220., -0.959])
        }

        >>> y = ivy.Container(a=ivy.array([0., 0.]))
        >>> x = ivy.Container(a=ivy.array([4., -2.]))
        >>> _ = x.expm1(out=y)
        >>> print(y)
        {
            a: ivy.array([53.6, -0.865])
        }

        """
        return self.static_expm1(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_floor(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.floor.
        This method simply wraps thefunction, and so the docstring for
        ivy.floor also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a numeric data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the rounded result for each element in ``x``. The
            returned array must have the same data type as ``x``.

        """
        return ContainerBase.multi_map_in_static_method(
            "floor",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def floor(
        self: ivy.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.floor.
        This method simply wraps the function, and so the docstring for
        ivy.floor also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a numeric data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the rounded result for each element in ``self``.
            The returned array must have the same data type as ``self``.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([2.5, 0.5, -1.4]),\
                              b=ivy.array([5.4, -3.2, 5.2]))
        >>> y = x.floor()
        >>> print(y)
        {
            a: ivy.array([2., 0., -2.]),
            b: ivy.array([5., -4., 5.])
        }
        """
        return self.static_floor(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_floor_divide(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.floor_divide.
        This method simply wraps the function, and so the docstring for
        ivy.floor_divide also applies to this method with minimal changes.

        Parameters
        ----------
        x1
            dividend input array or container. Should have a real-valued data type.
        x2
            divisor input array or container. Must be compatible with ``x1``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise results.
            The returned container must have a data type determined
            by :ref:`type-promotion`.

        Examples
        --------
        With :code:`ivy.Container` inputs:

        >>> x1 = ivy.Container(a=ivy.array([4., 5., 6.]), b=ivy.array([7., 8., 9.]))
        >>> x2 = ivy.Container(a=ivy.array([5., 4., 2.5]), b=ivy.array([2.3, 3.7, 5]))
        >>> y = ivy.Container.static_floor_divide(x1, x2)
        >>> print(y)
        {
            a: ivy.array([0., 1., 2.]),
            b: ivy.array([3., 2., 1.])
        }

        With mixed :code:`ivy.Container` and :code:`ivy.Array` inputs:

        >>> x1 = ivy.Container(a=ivy.array([4., 5., 6.]), b=ivy.array([7., 8., 9.]))
        >>> x2 = ivy.array([2, 3, 4])
        >>> y = ivy.Container.static_floor_divide(x1, x2)
        >>> print(y)
        {
            a: ivy.array([2., 1., 1.]),
            b: ivy.array([3., 2., 2.])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "floor_divide",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def floor_divide(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.floor_divide.
        This method simply wraps the function, and so the docstring for
        ivy.floor_divide also applies to this method with minimal changes.

        Parameters
        ----------
        self
            dividend input array or container. Should have a real-valued
            data type.
        x2
            divisor input array or container. Must be compatible with ``x1``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise results.
            The returned container must have a data type determined
            by :ref:`type-promotion`.

        Examples
        --------
        With :code:`ivy.Container` inputs:

        >>> x1 = ivy.Container(a=ivy.array([4., 5., 6.]), b=ivy.array([7., 8., 9.]))
        >>> x2 = ivy.Container(a=ivy.array([5., 4., 2.5]), b=ivy.array([2.3, 3.7, 5]))
        >>> y = x1.floor_divide(x2)
        >>> print(y)
        {
            a: ivy.array([0., 1., 2.]),
            b: ivy.array([3., 2., 1.])
        }

        With mixed :code:`ivy.Container` and :code:`ivy.Array` inputs:

        >>> x1 = ivy.Container(a=ivy.array([4., 5., 6.]), b=ivy.array([7., 8., 9.]))
        >>> x2 = ivy.array([2, 3, 4])
        >>> y = x1.floor_divide(x2)
        >>> print(y)
        {
            a: ivy.array([2., 1., 1.]),
            b: ivy.array([3., 2., 2.])
        }
        """
        return self.static_floor_divide(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_greater(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.greater.
        This method simply wraps the function, and so the docstring
        for ivy.greater also applies to this method with minimal changes.

        Parameters
        ----------
        x1
            input array or container. Should have a real-valued data type.
        x2
            divisor input array or container. Must be compatible with ``x1``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise results. The returned array must
            have a data type of ``bool``.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([4, 5, 6]),\
                          b=ivy.array([2, 3, 4]))
        >>> y = ivy.Container(a=ivy.array([1, 2, 3]),\
                          b=ivy.array([5, 6, 7]))
        >>> z = ivy.Container.static_greater(y,x)
        >>> print(z)
        {
            a: ivy.array([False, False, False]),
            b: ivy.array([True, True, True])
        }

        """
        return ContainerBase.multi_map_in_static_method(
            "greater",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def greater(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.greater.
        This method simply wraps the function, and so the docstring for
        ivy.greater also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array or container. Should have a real-valued data type.
        x2
            divisor input array or container. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise results. The returned array must
            have a data type of ``bool``.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([4, 5, 6]),\
                          b=ivy.array([2, 3, 4]))
        >>> y = ivy.Container(a=ivy.array([1, 2, 3]),\
                          b=ivy.array([5, 6, 7]))
        >>> z = x.greater(y)
        >>> print(z)
        {
            a: ivy.array([True, True, True]),
            b: ivy.array([False, False, False])
        }

        """
        return self.static_greater(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_greater_equal(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.greater_equal.
        This method simply wraps the function, and so the docstring for
        ivy.greater_equal also applies to this method with minimal changes.

        Parameters
        ----------
        x1
            input array or container. Should have a real-valued data type.
        x2
            input array or container. Must be compatible with ``x1``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise results. The returned container
            must have a data type of ``bool``.
        """
        return ContainerBase.multi_map_in_static_method(
            "greater_equal",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def greater_equal(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.greater_equal.
        This method simply wraps the function, and so the docstring for
        ivy.greater_equal also applies to this metho with minimal changes.

        Parameters
        ----------
        self
            input array or container. Should have a real-valued data type.
        x2
            input array or container. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise results. The returned container
            must have a data type of ``bool``.
        """
        return self.static_greater_equal(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_isfinite(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.isfinite.
        This method simply wraps the function, and so the docstring for
        ivy.isfinite also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a real-valued data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the test result. An element ``out_i`` is ``True``
            if ``x_i`` is finite and ``False`` otherwise.
            The returned array must have a data type of ``bool``.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0., 999999999999]),\
                          b=ivy.array([float('-0'), ivy.nan]))
        >>> y = ivy.Container.static_isfinite(x)
        >>> print(y)
        {
            a: ivy.array([True, True]),
            b: ivy.array([True, False])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "isfinite",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def isfinite(
        self: ivy.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.isfinite.
        This method simply wraps the function, and so the docstring for
        ivy.isfinite also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a real-valued data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the test result. An element ``out_i`` is ``True``
            if ``self_i`` is finite and ``False`` otherwise.
            The returned array must have a data type of ``bool``.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0., 999999999999]),\
                          b=ivy.array([float('-0'), ivy.nan]))
        >>> y = x.isfinite()
        >>> print(y)
        {
            a: ivy.array([True, True]),
            b: ivy.array([True, False])
        }
        """
        return self.static_isfinite(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_isinf(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.isinf.
        This method simply wraps the function, and so the docstring for
        ivy.isinf also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a real-valued data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the test result. An element ``out_i`` is ``True``
            if ``x_i`` is either positive or negative infinity and ``False``
            otherwise. The returned array must have a data type of ``bool``.

        """
        return ContainerBase.multi_map_in_static_method(
            "isinf",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def isinf(
        self: ivy.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.isinf.
        This method simply wraps the function, and so the docstring for
        ivy.isinf also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a real-valued data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the test result. An element ``out_i`` is ``True``
            if ``self_i`` is either positive or negative infinity and ``False``
            otherwise. The returned array must have a data type of ``bool``.

        """
        return self.static_isinf(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_isnan(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.isnan.
        This method simply wraps the function, and so the docstring for
        ivy.isnan also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a real-valued data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the test result. An element ``out_i`` is ``True``
            if ``x_i`` is ``NaN`` and ``False`` otherwise.
            The returned array should have a data type of ``bool``.

        """
        return ContainerBase.multi_map_in_static_method(
            "isnan",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def isnan(
        self: ivy.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.isnan.
        This method simply wraps the function, and so the docstring
        for ivy.isnan also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a real-valued data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the test result. An element ``out_i`` is ``True``
            if ``self_i`` is ``NaN`` and ``False`` otherwise.
            The returned array should have a data type of ``bool``.

        """
        return self.static_isnan(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_less(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.less.
        This method simply wraps the function, and so the docstring for
         ivy.less also applies to this method with minimal changes.

        Parameters
        ----------
        x1
            input array or container. Should have a real-valued data type.
        x2
            input array or container. Must be compatible with ``x1``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise results. The returned container
            must have a data type of ``bool``.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([4, 5, 6]),\
                              b=ivy.array([2, 3, 4]))
        >>> y = ivy.Container(a=ivy.array([1, 2, 3]),\
                              b=ivy.array([5, 6, 7]))
        >>> z = ivy.Container.static_less(y,x)
        >>> print(z)
        {
            a: ivy.array([True, True, True]),
            b: ivy.array([False, False, False])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "less",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def less(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.less.
        This method simply wraps the function, and so the docstring for
        ivy.less also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array or container. Should have a real-valued data type.
        x2
            input array or container. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise results. The returned container
            must have a data type of ``bool``.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([4, 5, 6]),\
                              b=ivy.array([2, 3, 4]))
        >>> y = ivy.Container(a=ivy.array([1, 2, 3]),\
                              b=ivy.array([5, 6, 7]))
        >>> z = x.less(y)
        >>> print(z)
        {
            a: ivy.array([False, False, False]),
            b: ivy.array([True, True, True])
        }

        """
        return self.static_less(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_less_equal(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.less_equal.
        This method simply wraps the function, and so the docstring for
        ivy.less_equal also applies to this method with minimal changes.

        Parameters
        ----------
        x1
            input array or container. Should have a real-valued data type.
        x2
            input array or container. Must be compatible with ``x1``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise results. The returned container
            must have a data type of ``bool``.
        """
        return ContainerBase.multi_map_in_static_method(
            "less_equal",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def less_equal(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.less_equal.
        This method simply wraps the function, and so the docstring for
        ivy.less_equal also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array or container. Should have a real-valued data type.
        x2
            input array or container. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise results. The returned container
            must have a data type of ``bool``.

        With :code:'ivy.Container' inputs:

        >>> x1 = ivy.Container(a=ivy.array([12, 3.5, 9.2]), b=ivy.array([2., 1.1, 5.5]))
        >>> x2 = ivy.Container(a=ivy.array([12, 2.2, 4.1]), b=ivy.array([1, 0.7, 3.8]))
        >>> y = x1.less_equal(x2)
        >>> print(y)
        {
            a: ivy.array([True, False, False]),
            b: ivy.array([False, False, False])
        }

        With mixed :code:'ivy.Container' and :code:'ivy.Array' inputs:

        >>> x1 = ivy.Container(a=ivy.array([12., 3.5, 9.2]), b=ivy.array([2., 1., 5.5]))
        >>> x2 = ivy.array([2., 1.1, 5.5])
        >>> y = x1.less_equal(x2)
        >>> print(y)
        {
            a: ivy.array([False, False, False]),
            b: ivy.array([True, True, True])
        }


        """
        return self.static_less_equal(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_log(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.log.
        This method simply wraps the function, and so the docstring for
        ivy.log also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a real-valued floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the log for each element in ``x``.
            The returned array must have a real-valued floating-point data type
            determined by :ref:`type-promotion`.

        """
        return ContainerBase.multi_map_in_static_method(
            "log",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def log(
        self: ivy.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.log.
        This method simply wraps the function, and so the docstring for
        ivy.log also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a real-valued floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the log for each element in ``self``.
            The returned array must have a real-valued floating-point data type
            determined by :ref:`type-promotion`.

        """
        return self.static_log(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_log1p(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.log1p.
        This method simply wraps the function, and so the docstring for
        ivy.log1p also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a real-valued floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the evaluated result for each element in ``x``.
            The returned array must have a real-valued floating-point data type
            determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([3., 4., 5.1]))
        >>> y = ivy.Container.static_log1p(x)
        >>> print(y)
        {
            a: ivy.array([0., 0.693, 1.1]),
            b: ivy.array([1.39, 1.61, 1.81])
        }

        >>> x = ivy.Container(a=ivy.array([0., 2.]), b=ivy.array([ 4., 5.1]))
        >>> ivy.Container.static_log1p(x , out = x)
        >>> print(y)
        {
            a: ivy.array([0., 0.693, 1.1]),
            b: ivy.array([1.39, 1.61, 1.81])
        }

        """
        return ContainerBase.multi_map_in_static_method(
            "log1p",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def log1p(
        self: ivy.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.log1p.
        This method simply wraps the function, and so the docstring for
        ivy.log1p also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a real-valued floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the evaluated result for each element in ``self``.
            The returned array must have a real-valued floating-point data type
            determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1.6, 2.6, 3.5]),\
                            b=ivy.array([4.5, 5.3, 2.3]))
        >>> y = x.log1p()
        >>> print(y)
        {
            a: ivy.array([0.956, 1.28, 1.5]),
            b: ivy.array([1.7, 1.84, 1.19])
        }

        """
        return self.static_log1p(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_log2(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.log2.
        This method simply wraps the function, and so the docstring for
        ivy.log2 also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a real-valued floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the evaluated base ``2`` logarithm for
            each element in ``x``. The returned array must have a real-valued
            floating-point data type determined by :ref:`type-promotion`.

        """
        return ContainerBase.multi_map_in_static_method(
            "log2",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def log2(
        self: ivy.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.log2.
        This method simply wraps the function, and so the docstring for
        ivy.log2 also applies to this metho with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a real-valued floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the evaluated base ``2`` logarithm for each
            element in ``self``. The returned array must have a real-valued
            floating-point data type determined by :ref:`type-promotion`.

        """
        return self.static_log2(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_log10(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.log10.
        This method simply wraps the function, and so the docstring for
        ivy.log10 also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a real-valued floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the evaluated base ``10`` logarithm for each
            element in ``x``. The returned array must have a real-valued
            floating-point data type determined by :ref:`type-promotion`.

        Examples
        --------
        Using :code:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([0.0, float('nan')]),\
                              b=ivy.array([-0., -3.9, float('+inf')]),\
                              c=ivy.array([7.9, 1.1, 1.]))
        >>> y = ivy.Container.static_log10(x)
        >>> print(y)
        {
            a: ivy.array([-inf, nan]),
            b: ivy.array([-inf, nan, inf]),
            c: ivy.array([0.898, 0.0414, 0.])
        }

        """
        return ContainerBase.multi_map_in_static_method(
            "log10",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def log10(
        self: ivy.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.log10.
        This method simply wraps the function, and so the docstring for
        ivy.log10 also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a real-valued floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the evaluated base ``10`` logarithm for
            each element in ``self``. The returned array must have a real-valued
            floating-point data type determined by :ref:`type-promotion`.

        Examples
        --------
        Using :code:`ivy.Container` instance method:

        >>> x = ivy.Container(a=ivy.array([0.0, float('nan')]), \
                              b=ivy.array([-0., -3.9, float('+inf')]), \
                              c=ivy.array([7.9, 1.1, 1.]))
        >>> y = x.log10()
        >>> print(y)
        {
            a: ivy.array([-inf, nan]),
            b: ivy.array([-inf, nan, inf]),
            c: ivy.array([0.898, 0.0414, 0.])
        }
        """
        return self.static_log10(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_logaddexp(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.greater_equal.
        This method simply wraps the function, and so the docstring for
        ivy.greater_equal also applies to this method with minimal changes.

        Parameters
        ----------
        x1
            input array or container. Should have a real-valued data type.
        x2
            input array or container. Must be compatible with ``x1``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise results. The returned container
            must have a real-valued floating-point data type determined
            by :ref:`type-promotion`.
        """
        return ContainerBase.multi_map_in_static_method(
            "logaddexp",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def logaddexp(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.greater_equal.
        This method simply wraps the function, and so the docstring for
        ivy.greater_equal also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array or container. Should have a real-valued data type.
        x2
            input array or container. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise results. The returned container
            must have a real-valued floating-point data type determined
            by :ref:`type-promotion`.
        """
        return self.static_logaddexp(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_logical_and(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.logical_and.
        This method simply wraps the function, and so the docstring for
        ivy.logical_and also applies to this method with minimal changes.

        Parameters
        ----------
        x1
            input array or container. Should have a boolean data type.
        x2
            input array or container. Must be compatible with ``x1``
            (see :ref:`broadcasting`).
            Should have a boolean data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise results. The returned container
            must have a data type of ``bool``.

        Examples
        --------
        Using 'ivy.Container' instance

        >>> i = ivy.Container(a=ivy.array([True, False, True, False]))
        >>> j = ivy.Container(a=ivy.array([True, True, False, False]))
        >>> k = ivy.Container(a=ivy.array([True, False, True]), \
            b=ivy.array([True, False, False]))
        >>> l = ivy.Container(a=ivy.array([True, True, True]), \
            b=ivy.array([False, False, False]))
        >>> m = ivy.array([False, True, False, True])
        >>> n = ivy.array([True, False, True, False])

        >>> w = ivy.Container.static_logical_and(i, j)
        >>> x = ivy.Container.static_logical_and(j, m)
        >>> y = ivy.Container.static_logical_and(m, n)
        >>> z = ivy.Container.static_logical_and(k, l)

        {
            a: ivy.array([True, False, False, False])
        }

        >>> print(x)
        {
            a: ivy.array([False, True, False, False])
        }

        >>> print(y)
            ivy.array([False, False, False, False])

        >>> print(z)
        {
            a: ivy.array([True, False, True]),
            b: ivy.array([False, False, False])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "logical_and",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def logical_and(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.logical_and.
        This method simply wraps the function, and so the docstring for
        ivy.logical_and also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array or container. Should have a boolean data type.
        x2
            input array or container. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a boolean data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise results. The returned container
            must have a data type of ``bool``.

        Examples
        --------
        Using 'ivy.Container' instance

        >>> i = ivy.Container(a=ivy.array([True, False, True, False]))
        >>> j = ivy.Container(a=ivy.array([True, True, False, False]))
        >>> k = ivy.Container(a=ivy.array([True, False, True]), \
            b=ivy.array([True, False, False]))
        >>> l = ivy.Container(a=ivy.array([True, True, True]), \
            b=ivy.array([False, False, False]))
        >>> m = ivy.array([False, True, False, True])
        >>> n = ivy.array([True, False, True, False])

        >>> w = i.logical_and(j)
        >>> x = j.logical_and(m)
        >>> y = m.logical_and(n)
        >>> z = k.logical_and(l)

        >>> print(w)
        {a:ivy.array([True,False,False,False])}

        >>> print(x)
        {a:ivy.array([False,True,False,False])}

        >>> print(y)
            ivy.array([False, False, False, False])

        >>> print(z)
        {a:ivy.array([True,False,True]),b:ivy.array([False,False,False])}
        """
        return self.static_logical_and(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_logical_not(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.logical_not.
        This method simply wraps the function, and so the docstring for
        ivy.logical_not also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a boolean data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the evaluated result for each element in ``x``.
            The returned container must have a data type of ``bool``.

        """
        return ContainerBase.multi_map_in_static_method(
            "logical_not",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def logical_not(
        self: ivy.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.logical_not.
        This method simply wraps the function, and so the docstring for
        ivy.logical_not also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a boolean data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the evaluated result for each element in ``self``.
            The returned container must have a data type of ``bool``.

        """
        return self.static_logical_not(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_logical_or(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.logical_or.
        This method simply wraps the function, and so the docstring for
        ivy.logical_or also applies to this method with minimal changes.

        Parameters
        ----------
        x1
            input array or container. Should have a boolean data type.
        x2
            input array or container. Must be compatible with ``x1``
            (see :ref:`broadcasting`).
            Should have a boolean data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise results. The returned container
            must have a data type of ``bool``.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([False, False, True]), \
                            b=ivy.array([True, False, True]))
        >>> y = ivy.Container(a=ivy.array([False, True, False]), \
                                b=ivy.array([True, True, False]))
        >>> z = ivy.Container.static_logical_or(x, y)
        >>> print(z)
        {
            a: ivy.array([False, True, True]),
            b: ivy.array([True, True, True])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "logical_or",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def logical_or(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.logical_or.
        This method simply wraps the function, and so the docstring for
        ivy.logical_or also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array or container. Should have a boolean data type.
        x2
            input array or container. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a boolean data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise results. The returned container
            must have a data type of ``bool``.

        This function conforms to the `Array API Standard
        <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
        `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.elementwise_functions.logical_or.html>`_ # noqa
        in the standard.

        Both the description and the type hints above assumes an array input for simplicity,
        but this function is *nestable*, and therefore also accepts :code:`ivy.Container`
        instances in place of any of the arguments.

        Examples
        --------
        Using :code:`ivy.Container` instance method:

        >>> x = ivy.Container(a=ivy.array([False,True,True]), b=ivy.array([3.14, 2.718, 1.618]))
        >>> y = ivy.Container(a=ivy.array([0, 5.2, 0.8]), b=ivy.array([0.2, 0, 0.9]))
        >>> z = x.logical_or(y)
        >>> print(z)
        {
            a: ivy.array([False, True, True]),
            b: ivy.array([True, True, True])
        }
        """
        return self.static_logical_or(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_logical_xor(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.logical_xor.
        This method simply wraps the function, and so the docstring for
        ivy.logical_xor also applies to this method with minimal changes.

        Parameters
        ----------
        x1
            input array or container. Should have a boolean data type.
        x2
            input array or container. Must be compatible with ``x1``
            (see :ref:`broadcasting`).
            Should have a boolean data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise results. The returned container
            must have a data type of ``bool``.

        Examples
        --------
        With one :code:`ivy.Container` input:

        >>> x = ivy.array([0,0,1,1,0])
        >>> y = ivy.Container(a=ivy.array([1,0,0,1,0]), b=ivy.array([1,0,1,0,0]))
        >>> z = ivy.Container.static_logical_xor(x, y)
        >>> print(z)
        {
            a: ivy.array([True, False, True, False, False]),
            b: ivy.array([True, False, False, True, False])
        }

        With multiple :code:`ivy.Container` inputs:

        >>> x = ivy.Container(a=ivy.array([1,0,0,1,0]), b=ivy.array([1,0,1,0,0]))
        >>> y = ivy.Container(a=ivy.array([0,0,1,1,0]), b=ivy.array([1,0,1,1,0]))
        >>> z = ivy.Container.static_logical_xor(x, y)
        >>> print(z)
        {
            a: ivy.array([True, False, True, False, False]),
            b: ivy.array([False, False, False, True, False])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "logical_xor",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def logical_xor(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.logical_xor.
        This method simply wraps the function, and so the docstring for
        ivy.logical_xor also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array or container. Should have a boolean data type.
        x2
            input array or container. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a boolean data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise results. The returned container
            must have a data type of ``bool``.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1,0,0,1,0]), b=ivy.array([1,0,1,0,0]))
        >>> y = ivy.Container(a=ivy.array([0,0,1,1,0]), b=ivy.array([1,0,1,1,0]))
        >>> z = x.logical_xor(y)
        >>> print(z)
        {
            a: ivy.array([True, False, True, False, False]),
            b: ivy.array([False, False, False, True, False])
        }
        """
        return self.static_logical_xor(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_multiply(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.multiply.
        This method simply wraps the function, and so the docstring for
        ivy.multiply also applies to this method with minimal changes.

        Parameters
        ----------
        x1
            input array or container. Should have a real-valued data type.
        x2
            input array or container. Must be compatible with ``x1``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise results. The returned container
            must have a data type determined by :ref:`type-promotion`.
        """
        return ContainerBase.multi_map_in_static_method(
            "multiply",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def multiply(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.multiply.
        This method simply wraps the function, and so the docstring for
        ivy.multiply also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array or container. Should have a real-valued data type.
        x2
            input array or container. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise results. The returned container
            must have a data type determined by :ref:`type-promotion`.
        """
        return self.static_multiply(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_negative(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.negative.
        This method simply wraps the function, and so the docstring for
        ivy.negative also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a numeric data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the evaluated result for each element in ``x``.
            The returned container must have the same data type as ``x``.

        """
        return ContainerBase.multi_map_in_static_method(
            "negative",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def negative(
        self: ivy.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.negative.
        This method simply wraps the function, and so the docstring for
        ivy.negative also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a numeric data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the evaluated result for each element in ``self``.
            The returned container must have the same data type as ``self``.

        """
        return self.static_negative(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_not_equal(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.not_equal.
        This method simply wraps the function, and so the docstring for
        ivy.not_equal also applies to this method with minimal changes.

        Parameters
        ----------
        x1
            input array or container. May have any data type.
        x2
            input array or container. Must be compatible with ``x1``
            (see :ref:`broadcasting`).
            May have any data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise results. The returned container
            must have a data type of ``bool``.
        """
        return ContainerBase.multi_map_in_static_method(
            "not_equal",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def not_equal(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.not_equal.
        This method simply wraps the function, and so the docstring for
        ivy.not_equal also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array or container. May have any data type.
        x2
            input array or container. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            May have any data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise results. The returned container
            must have a data type of ``bool``.
        """
        return self.static_not_equal(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_positive(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.positive.
        This method simply wraps the function, and so the docstring for
        ivy.positive also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a numeric data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the evaluated result for each element in ``x``.
            The returned container must have the same data type as ``x``.

        """
        return ContainerBase.multi_map_in_static_method(
            "positive",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def positive(
        self: ivy.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.positive.
        This method simply wraps the function, and so the docstring for
        ivy.positive also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a numeric data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the evaluated result for each element in ``self``.
            The returned container must have the same data type as ``self``.

        """
        return self.static_positive(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_pow(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.pow. This method simply wraps
        the function, and so the docstring for ivy.pow also applies to this
        method with minimal changes.

        Parameters
        ----------
        x1
            input array or container. Should have a real-valued data type.
        x2
            input array or container. Must be compatible with ``x1``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise results. The returned container
            must have a data type determined by :ref:`type-promotion`.
        """
        return ContainerBase.multi_map_in_static_method(
            "pow",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def pow(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.pow. This method simply
        wraps the function, and so the docstring for ivy.pow also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array or container. Should have a real-valued data type.
        x2
            input array or container. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise results. The returned container
            must have a data type determined by :ref:`type-promotion`.
        """
        return self.static_pow(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_remainder(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.remainder.
        This method simply wraps the function, and so the docstring for
        ivy.remainder also applies to this method with minimal changes.

        Parameters
        ----------
        x1
            input array or container. Should have a real-valued data type.
        x2
            input array or container. Must be compatible with ``x1``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise results. The returned container
            must have the same sign as the respective element ``x2_i``.

        Examples
        --------
        With :code:`ivy.Container` inputs:

        >>> x1 = ivy.Container(a=ivy.array([2., 3., 5.]), b=ivy.array([2., 2., 4.]))
        >>> x2 = ivy.Container(a=ivy.array([1., 3., 4.]), b=ivy.array([1., 3., 3.]))
        >>> y = ivy.Container.static_remainder(x1, x2)
        >>> print(y)
        {
            a: ivy.array([0., 0., 1.]),
            b: ivy.array([0., 2., 1.])
        }

        With mixed :code:`ivy.Container` and `ivy.Array` inputs:

        >>> x1 = ivy.Container(a=ivy.array([2., 3., 5.]), b=ivy.array([2., 2., 4.]))
        >>> x2 = ivy.array([1., 2., 3.])
        >>> y = ivy.Container.static_remainder(x1, x2)
        >>> print(y)
        {
            a: ivy.array([0., 1., 2.]),
            b: ivy.array([0., 0., 1.])
        }

        With mixed :code:`ivy.Container` and `ivy.NativeArray` inputs:

        >>> x1 = ivy.Container(a=ivy.array([2., 3., 5.]), b=ivy.array([2., 2., 4.]))
        >>> x2 = ivy.native_array([1., 2., 3.])
        >>> y = ivy.Container.static_remainder(x1, x2)
        >>> print(y)
        {
            a: ivy.array([0., 1., 2.]),
            b: ivy.array([0., 0., 1.])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "remainder",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def remainder(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.remainder.
        This method simply wraps the function, and so the docstring for
        ivy.remainder also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array or container. Should have a real-valued data type.
        x2
            input array or container. Must be compatible with ``self``
            (see :ref:`broadcasting`).
            Should have a real-valued data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise results. The returned container
            must have the same sign as the respective element ``x2_i``.

        Examples
        --------
        With :code:`ivy.Container` inputs:

        >>> x1 = ivy.Container(a=ivy.array([2., 3., 5.]), b=ivy.array([2., 2., 4.]))
        >>> x2 = ivy.Container(a=ivy.array([1., 3., 4.]), b=ivy.array([1., 3., 3.]))
        >>> y = x1.remainder(x2)
        >>> print(y)
        {
            a: ivy.array([0., 0., 1.]),
            b: ivy.array([0., 2., 1.])
        }

        With mixed :code:`ivy.Container` and `ivy.Array` inputs:

        >>> x1 = ivy.Container(a=ivy.array([2., 3., 5.]), b=ivy.array([2., 2., 4.]))
        >>> x2 = ivy.array([1., 2., 3.])
        >>> y = x1.remainder(x2)
        >>> print(y)
        {
            a: ivy.array([0., 1., 2.]),
            b: ivy.array([0., 0., 1.])
        }

        With mixed :code:`ivy.Container` and `ivy.NativeArray` inputs:

        >>> x1 = ivy.Container(a=ivy.array([2., 3., 5.]), b=ivy.array([2., 2., 4.]))
        >>> x2 = ivy.native_array([1., 2., 3.])
        >>> y = x1.remainder(x2)
        >>> print(y)
        {
            a: ivy.array([0., 1., 2.]),
            b: ivy.array([0., 0., 1.])
        }
        """
        return self.static_remainder(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_round(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.round. This method simply
        wraps thevfunction, and so the docstring for ivy.round also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a numeric data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the rounded result for each element in ``x``.
            The returned container must have the same data type as ``x``.

        Examples
        --------
        With :code:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([4.20, 8.6, 6.90, 0.0]),\
                    b=ivy.array([-300.9, -527.3, 4.5]))
        >>> y = ivy.Container.static_round(x)
        >>> print(y)
        {
            a: ivy.array([4., 9., 7., 0.]),
            b: ivy.array([-301., -527., 4.])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "round",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def round(
        self: ivy.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.round. This method simply
        wraps the function, and so the docstring for ivy.round also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a numeric data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the rounded result for each element in ``self``.
            The returned container must have the same data type as ``self``.

        Examples
        --------
        With :code:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([4.20, 8.6, 6.90, 0.0]),\
                    b=ivy.array([-300.9, -527.3, 4.5]))
        >>> y = x.round()
        >>> print(y)
        {
            a: ivy.array([4., 9., 7., 0.]),
            b: ivy.array([-301., -527., 4.])
        }
        """
        return self.static_round(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_sign(
        x: Union[float, ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.sign. This method simply
        wraps the function, and so the docstring for ivy.sign also applies
        to this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a numeric data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the evaluated result for each element in ``x``.
            The returned container must have the same data type as ``x``.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0, -1., 6.6]),\
                            b=ivy.array([-14.2, 8.3, 0.1, -0]))
        >>> y = ivy.Container.static_sign(x)
        >>> print(y)
        {
            a: ivy.array([0., -1., 1.]),
            b: ivy.array([-1., 1., 1., 0.])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "sign",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def sign(
        self: ivy.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.sign. This method simply
        wraps the function, and so the docstring for ivy.sign also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a numeric data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the evaluated result for each element in ``self``.
            The returned container must have the same data type as ``self``.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([-6.7, 2.4, -8.5]),\
                              b=ivy.array([1.5, -0.3, 0]),\
                              c=ivy.array([-4.7, -5.4, 7.5]))
        >>> y = x.sign()
        >>> print(y)
        {
            a: ivy.array([-1., 1., -1.]),
            b: ivy.array([1., -1., 0.]),
            c: ivy.array([-1., -1., 1.])
        }
        """
        return self.static_sign(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_sin(
        x: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.sin. This method simply
        wraps the function, and so the docstring for ivy.sin also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container whose elements are each expressed in radians.
            Should have a floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the sine of each element in ``x``. The returned
            container must have a floating-point data type determined by
            :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([-1., -2., -3.]),\
                              b=ivy.array([4., 5., 6.]))
        >>> y = ivy.Container.static_sin(x)
        >>> print(y)
        {
            a: ivy.array([-0.841, -0.909, -0.141]),
            b: ivy.array([-0.757, -0.959, -0.279])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "sin",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def sin(
        self: ivy.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.sin. This method simply
        wraps the function, and so the docstring for ivy.sin also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container whose elements are each expressed in radians.
            Should have a floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the sine of each element in ``self``.
            The returned container must have a floating-point data type
            determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1., 2., 3.]),\
                              b=ivy.array([-4., -5., -6.]))
        >>> y = x.sin()
        >>> print(y)
        {
            a: ivy.array([0.841, 0.909, 0.141]),
            b: ivy.array([0.757, 0.959, 0.279])
        }
        """
        return self.static_sin(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_sinh(
        x,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.sinh.
        This method simply wraps the function, and so the docstring for
        ivy.sinh also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container whose elements each represent a hyperbolic angle.
            Should have a floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            an container containing the hyperbolic sine of each element in ``x``.
            The returned container must have a floating-point data type determined
            by :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([-1, 0.23, 1.12]), b=ivy.array([1, -2, 0.76]))
        >>> y = ivy.Container.static_sinh(x)
        >>> print(y)
        {
            a: ivy.array([-1.18, 0.232, 1.37]),
            b: ivy.array([1.18, -3.63, 0.835])
        }

        >>> x = ivy.Container(a=ivy.array([-3, 0.34, 2.]),\
                    b=ivy.array([0.67, -0.98, -3]))
        >>> y = ivy.Container(a=ivy.zeros(1), b=ivy.zeros(1))
        >>> ivy.Container.static_sinh(x, out=y)
        >>> print(y)
        {
            a: ivy.array([-10., 0.347, 3.63]),
            b: ivy.array([0.721, -1.14, -10.])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "sinh",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def sinh(
        self: ivy.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.sinh.
        This method simply wraps the function, and so the docstring for
        ivy.sinh also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container whose elements each represent a hyperbolic angle.
            Should have a floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            an container containing the hyperbolic sine of each element in ``self``.
            The returned container must have a floating-point data type determined
            by :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([-1, 0.23, 1.12]), b=ivy.array([1, -2, 0.76]))
        >>> y = x.sinh()
        >>> print(y)
        {
            a: ivy.array([-1.18, 0.232, 1.37]),
            b: ivy.array([1.18, -3.63, 0.835])
        }

        >>> x = ivy.Container(a=ivy.array([-3, 0.34, 2.]),\
                    b=ivy.array([0.67, -0.98, -3]))
        >>> y = ivy.Container(a=ivy.zeros(1), b=ivy.zeros(1))
        >>> x.sinh(out=y)
        >>> print(y)
        {
            a: ivy.array([-10., 0.347, 3.63]),
            b: ivy.array([0.721, -1.14, -10.])
        }
        """
        return self.static_sinh(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_square(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.square.
        This method simply wraps the function, and so the docstring for
        ivy.square also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a real-valued floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the square of each element in ``x``.
            The returned container must have a real-valued floating-point
            data type determined by :ref:`type-promotion`.

        """
        return ContainerBase.multi_map_in_static_method(
            "square",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def square(
        self: ivy.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.square.
        This method simply wraps the function, and so the docstring for
        ivy.square also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a real-valued floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the square of each element in ``self``.
            The returned container must have a real-valued floating-point
            data type determined by :ref:`type-promotion`.

        """
        return self.static_square(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_sqrt(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.sqrt.
        This method simply wraps the function, and so the docstring for
        ivy.sqrt also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a real-valued floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the square root of each element in ``x``.
            The returned container must have a real-valued floating-point
            data type determined by :ref:`type-promotion`.

        """
        return ContainerBase.multi_map_in_static_method(
            "sqrt",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def sqrt(
        self: ivy.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.sqrt.
        This method simply wraps the function, and so the docstring
        for ivy.sqrt also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a real-valued floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the square root of each element in
            ``self``. The returned container must have a real-valued
            floating-point data type determined by :ref:`type-promotion`.

        """
        return self.static_sqrt(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_subtract(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.subtract.
        This method simply wraps the function, and so the docstring
        for ivy.subtract also applies to this method with minimal changes.

        Parameters
        ----------
        x1
            first input array or container. Should have a numeric data type.
        x2
            second input array or container. Must be compatible with ``x1``
            (see :ref:`broadcasting`). Should have a numeric data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise sums.
            The returned container must have a data type determined
            by :ref:`type-promotion`.
        """
        return ContainerBase.multi_map_in_static_method(
            "subtract",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def subtract(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.subtract.
        This method simply wraps the function, and so the docstring
        for ivy.subtract also applies to this method with minimal changes.

        Parameters
        ----------
        self
            first input array or container. Should have a numeric data type.
        x2
            second input array or container. Must be compatible with ``self``
            (see :ref:`broadcasting`). Should have a numeric data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the element-wise sums.
            The returned container must have a data type determined
            by :ref:`type-promotion`.
        """
        return self.static_subtract(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_tan(
        x: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.tan.
        This method simply wraps the function, and so the docstring for
        ivy.tan also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input array whose elements are expressed in radians. Should have a
            floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output, for writing the result to. It must have a shape that the
            inputs broadcast to.

        Returns
        -------
        ret
            an array containing the tangent of each element in ``x``.
            The return must have a floating-point data type determined
            by :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([3., 4., 5.]))
        >>> y = ivy.Container.static_tan(x)
        >>> print(y)
        {
            a: ivy.array([0., 1.56, -2.19]),
            b: ivy.array([-0.143, 1.16, -3.38])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "tan",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def tan(
        self: ivy.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.tan.
        This method simply wraps the function, and so the docstring for
        ivy.tan also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array whose elements are expressed in radians. Should have a
            floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output, for writing the result to. It must have a shape that the
            inputs broadcast to.

        Returns
        -------
        ret
            a container containing the tangent of each element in ``self``.
            The return must have a floating-point data type determined
            by :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([3., 4., 5.]))
        >>> y = x.tan()
        >>> print(y)
        {
            a:ivy.array([0., 1.56, -2.19]),
            b:ivy.array([-0.143, 1.16, -3.38])
        }
        """
        return self.static_tan(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_tanh(
        x: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.tanh.
        This method simply wraps the function, and so the docstring for
        ivy.tanh also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container whose elements each represent a hyperbolic angle.
            Should have a real-valued floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the hyperbolic tangent of each element in ``x``.
            The returned array must have a real-valued floating-point data type
            determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([3., 4., 5.]))
        >>> y = ivy.Container.static_tanh(x)
        >>> print(y)
        {
            a: ivy.array([0., 0.76, 0.96]),
            b: ivy.array([0.995, 0.999, 0.9999])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "tanh",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def tanh(
        self: ivy.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.tanh.
        This method simply wraps the function, and so the docstring for
        ivy.tanh also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container whose elements each represent a hyperbolic angle.
            Should have a real-valued floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the hyperbolic tangent of each element in
            ``self``. The returned container must have a real-valued floating-point
            data type determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0., 1., 2.]),\
                              b=ivy.array([3., 4., 5.]))
        >>> y = x.tanh()
        >>> print(y)
        {
            a:ivy.array([0., 0.762, 0.964]),
            b:ivy.array([0.995, 0.999, 1.])
        }
        """
        return self.static_tanh(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_trunc(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.trunc.
        This method simply wraps the function, and so the docstring for
        ivy.trunc also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container. Should have a real-valued data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the rounded result for each element in ``x``.
            The returned container must have the same data type as ``x``.

        """
        return ContainerBase.multi_map_in_static_method(
            "trunc",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def trunc(
        self: ivy.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.trunc.
        This method simply wraps the function, and so the docstring for
        ivy.trunc also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a real-valued data type.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the rounded result for each element in ``self``.
            The returned container must have the same data type as ``self``.

        """
        return self.static_trunc(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_erf(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.erf.
        This method simply wraps the function, and so the docstring for
        ivy.erf also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container to compute exponential for.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the Gauss error of ``x``.

        """
        return ContainerBase.multi_map_in_static_method(
            "erf",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def erf(
        self: ivy.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.erf.
        This method simply wraps thefunction, and so the docstring for
        ivy.erf also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container to compute exponential for.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the Gauss error of ``self``.

        """
        return self.static_erf(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_minimum(
        x1: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.minimum.
        This method simply wraps the function, and so the docstring for
        ivy.minimum also applies to this method with minimal changes.

        Parameters
        ----------
        x1
            Input array containing elements to minimum threshold.
        x2
            The other container or number to compute the minimum against.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
            Container object with all sub-arrays having the minimum values computed.

        """
        return ContainerBase.multi_map_in_static_method(
            "minimum",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def minimum(
        self: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.minimum.
        This method simply wraps the function, and so the docstring for
        ivy.minimum also applies to this method with minimal changes.


        Parameters
        ----------
        self
            Input array containing elements to minimum threshold.
        x2
            The other container or number to compute the minimum against.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
            Container object with all sub-arrays having the minimum values computed.

        """
        return self.static_minimum(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_maximum(
        x1: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.maximum.
        This method simply wraps the function, and so the docstring for
        ivy.maximum also applies to this method with minimal changes.

        Parameters
        ----------
        x1
            Input array containing elements to maximum threshold.
        x2
            Tensor containing maximum values, must be broadcastable to x1.
        out
            optional output array, for writing the result to.
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            An array with the elements of x1, but clipped to not be lower than the x2
            values.


        """
        return ContainerBase.multi_map_in_static_method(
            "maximum",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def maximum(
        self: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.maximum.
        This method simply wraps the function, and so the docstring for
        ivy.maximum also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input array containing elements to maximum threshold.
        x2
            Tensor containing maximum values, must be broadcastable to x1.
        out
            optional output array, for writing the result to.
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            An array with the elements of x1, but clipped to not be lower than the x2
            values.


        """
        return self.static_maximum(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )
