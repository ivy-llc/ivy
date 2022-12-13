# global
from numbers import Number
from typing import Any, Union, List, Dict, Iterable, Optional, Callable

# local
from ivy.container.base import ContainerBase
import ivy

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class ContainerWithGeneral(ContainerBase):
    @staticmethod
    def static_is_native_array(
        x: ivy.Container,
        /,
        *,
        exclusive: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.is_native_array.
        This method simply wraps the function, and so the docstring for
        ivy.is_native_array also applies to this method with minimal changes.

        Parameters
        ----------
        x
            The input to check
        exclusive
            Whether to check if the data type is exclusively an array, rather than a
            variable or traced array.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Returns
        -------
        ret
            Boolean, whether or not x is a native array.
        """
        return ContainerBase.cont_multi_map_in_static_method(
            "is_native_array",
            x,
            exclusive=exclusive,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def is_native_array(
        self: ivy.Container,
        /,
        *,
        exclusive: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.is_native_array.
        This method simply wraps the function, and so the docstring for
        ivy.ivy.is_native_array also applies to this method with minimal changes.

        Parameters
        ----------
        self
            The input to check
        exclusive
            Whether to check if the data type is exclusively an array, rather than a
            variable or traced array.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Returns
        -------
        ret
            Boolean, whether or not x is a native array.
        """
        return self.static_is_native_array(
            self,
            exclusive=exclusive,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_is_ivy_array(
        x: ivy.Container,
        /,
        *,
        exclusive: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.is_ivy_array.
        This method simply wraps the function, and so the docstring for
        ivy.is_ivy_array also applies to this method with minimal changes.

        Parameters
        ----------
        x
            The input to check
        exclusive
            Whether to check if the data type is exclusively an array, rather than a
            variable or traced array.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Returns
        -------
        ret
            Boolean, whether or not x is an array.
        """
        return ContainerBase.cont_multi_map_in_static_method(
            "is_ivy_array",
            x,
            exclusive=exclusive,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def is_ivy_array(
        self: ivy.Container,
        /,
        *,
        exclusive: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.is_native_array.
        This method simply wraps the function, and so the docstring for
        ivy.ivy.is_native_array also applies to this method with minimal changes.

        Parameters
        ----------
        self
            The input to check
        exclusive
            Whether to check if the data type is exclusively an array, rather than a
            variable or traced array.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Returns
        -------
        ret
            Boolean, whether or not x is an array.
        """
        return self.static_is_ivy_array(
            self,
            exclusive=exclusive,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_is_array(
        x: ivy.Container,
        /,
        *,
        exclusive: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.is_array.
        This method simply wraps the function, and so the docstring for
        ivy.ivy.is_array also applies to this method with minimal changes.

        Parameters
        ----------
        x
            The input to check
        exclusive
            Whether to check if the data type is exclusively an array, rather than a
            variable or traced array.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output, for writing the result to. It must have a shape that the
            inputs broadcast to.

        Returns
        -------
        ret
            Boolean, whether or not x is an array.
        """
        return ContainerBase.cont_multi_map_in_static_method(
            "is_array",
            x,
            exclusive=exclusive,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def is_array(
        self: ivy.Container,
        /,
        *,
        exclusive: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.is_array.
        This method simply wraps the function, and so the docstring for
        ivy.is_array also applies to this method with minimal changes.

        Parameters
        ----------
        self
            The input to check
        exclusive
            Whether to check if the data type is exclusively an array, rather than a
            variable or traced array.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Returns
        -------
        ret
            Boolean, whether or not x is an array.
        """
        return self.static_is_array(
            self,
            exclusive=exclusive,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_clip_vector_norm(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        max_norm: float,
        /,
        *,
        p: float = 2.0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.clip_vector_norm. This method
        simply wraps the function, and so the docstring for ivy.clip_vector_norm
        also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input array
        max_norm
            float, the maximum value of the array norm.
        p
            optional float, the p-value for computing the p-norm. Default is 2.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output array, for writing the result to. It must
            have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            An array with the vector norm downscaled to the max norm if needed.

        Examples
        --------
        With :class:`ivy.Container` instance method:

        >>> x = ivy.Container(a=ivy.array([0., 1., 2.]),b=ivy.array([3., 4., 5.]))
        >>> y = ivy.Container.static_clip_vector_norm(x, 2.0)
        >>> print(y)
        {
            a: ivy.array([0., 0.894, 1.79]),
            b: ivy.array([0.849, 1.13, 1.41])
        }

        """
        return ContainerBase.cont_multi_map_in_static_method(
            "clip_vector_norm",
            x,
            max_norm,
            p=p,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def clip_vector_norm(
        self: ivy.Container,
        max_norm: float,
        /,
        *,
        p: float = 2.0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.clip_vector_norm. This method
        simply wraps the function, and so the docstring for ivy.clip_vector_norm
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array
        max_norm
            float, the maximum value of the array norm.
        p
            optional float, the p-value for computing the p-norm. Default is 2.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output array, for writing the result to. It must
            have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            An array with the vector norm downscaled to the max norm if needed.

        Examples
        --------
        With :class:`ivy.Container` instance method:

        >>> x = ivy.Container(a=ivy.array([0., 1., 2.]),
        ...                   b=ivy.array([3., 4., 5.]))
        >>> y = x.clip_vector_norm(2.0, p=1.0)
        >>> print(y)
        {
            a: ivy.array([0., 0.667, 1.33]),
            b: ivy.array([0.5, 0.667, 0.833])
        }

        """
        return self.static_clip_vector_norm(
            self,
            max_norm,
            p=p,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_inplace_update(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        val: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.inplace_update. This method
        simply wraps the function, and so the docstring for ivy.inplace_update
        also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container to be updated inplace
        val
            value to update the input container with
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output array, for writing the result to. It must
            have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            An array with the vector norm downscaled to the max norm if needed.

        """
        # inplace update the leaves
        cont = x
        cont = ContainerBase.cont_multi_map_in_static_method(
            "inplace_update",
            cont,
            val,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )
        # inplace update the container
        x.cont_inplace_update(cont)
        return x

    def inplace_update(
        self: ivy.Container,
        val: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.inplace_update. This method
        simply wraps the function, and so the docstring for ivy.inplace_update
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container to be updated inplace
        val
            value to update the input container with
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output array, for writing the result to. It must
            have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            An array with the vector norm downscaled to the max norm if needed.

        """
        return self.static_inplace_update(
            self,
            val,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_inplace_decrement(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        val: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.inplace_decrement. This method
        simply wraps the function, and so the docstring for ivy.inplace_decrement
        also applies to this method with minimal changes.

        Parameters
        ----------
        x
            The input array to be decremented by the defined value.
        val
            The value of decrement.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains will
            be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied. Default
            is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Returns
        -------
        ret
            The array following an in-place decrement.

        Examples
        --------
        Decrement by a value
        >>> x = ivy.Container(a=ivy.array([0.5, -5., 30.]),b=ivy.array([0., -25., 50.]))
        >>> y = ivy.inplace_decrement(x, 1.5)
        >>> print(y)
        {
            a: ivy.array([-1., -6.5, 28.5]),
            b: ivy.array([-1.5, -26.5, 48.5])
        }

        Decrement by a Container
        >>> x = ivy.Container(a=ivy.array([0., 15., 30.]), b=ivy.array([0., 25., 50.]))
        >>> y = ivy.Container(a=ivy.array([0., 15., 30.]), b=ivy.array([0., 25., 50.]))
        >>> z = ivy.inplace_decrement(x, y)
        >>> print(z)
        {
            a: ivy.array([0., 0., 0.]),
            b: ivy.array([0., 0., 0.])
        }

        >>> x = ivy.Container(a=ivy.array([3., 7., 10.]), b=ivy.array([0., 75., 5.5]))
        >>> y = ivy.Container(a=ivy.array([2., 5.5, 7.]), b=ivy.array([0., 25., 2.]))
        >>> z = ivy.inplace_decrement(x, y)
        >>> print(z)
        {
            a: ivy.array([1., 1.5, 3.]),
            b: ivy.array([0., 50., 3.5])
        }

        """
        return ContainerBase.cont_multi_map_in_static_method(
            "inplace_decrement",
            x,
            val,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def inplace_decrement(
        self: ivy.Container,
        val: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.inplace_decrement. This method
        simply wraps the function, and so the docstring for ivy.inplace_decrement
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input container to apply an in-place decrement.
        val
            The value of decrement.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Returns
        -------
        ret
            A container with the array following the in-place decrement.

        Examples
        --------
        Using :class:`ivy.Container` instance method:
        >>> x = ivy.Container(a=ivy.array([-6.7, 2.4, -8.5]),
        ...                   b=ivy.array([1.5, -0.3, 0]),
        ...                   c=ivy.array([-4.7, -5.4, 7.5]))
        >>> y = x.inplace_decrement(2)
        >>> print(y)
        {
            a: ivy.array([-8.7, 0.4, -10.5]),
            b: ivy.array([-0.5, -2.3, -2]),
            c: ivy.array([-6.7, -7.4, 5.5])
        }

        """
        return self.static_inplace_decrement(
            self,
            val,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_inplace_increment(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        val: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.inplace_increment. This method
        simply wraps the function, and so the docstring for ivy.inplace_increment
        also applies to this method with minimal changes.

        Parameters
        ----------
        x
            The input array to be incremented by the defined value.
        val
            The value of increment.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Returns
        -------
        ret
            The array following an in-place increment.

        Examples
        --------
        Increment by a value
        >>> x = ivy.Container(a=ivy.array([0.5, -5., 30.]),b=ivy.array([0., -25., 50.]))
        >>> y = ivy.inplace_increment(x, 1.5)
        >>> print(y)
        {
            a: ivy.array([2., -3.5, 31.5]),
            b: ivy.array([1.5, -23.5, 51.5])
        }

        Increment by a Container
        >>> x = ivy.Container(a=ivy.array([0., 15., 30.]), b=ivy.array([0., 25., 50.]))
        >>> y = ivy.Container(a=ivy.array([0., 15., 30.]), b=ivy.array([0., 25., 50.]))
        >>> z = ivy.inplace_increment(x, y)
        >>> print(z)
        {
            a: ivy.array([0., 30., 60.]),
            b: ivy.array([0., 50., 100.])
        }

        >>> x = ivy.Container(a=ivy.array([3., 7., 10.]), b=ivy.array([0., 75., 5.5]))
        >>> y = ivy.Container(a=ivy.array([2., 5.5, 7.]), b=ivy.array([0., 25., 2.]))
        >>> z = ivy.inplace_increment(x, y)
        >>> print(z)
        {
            a: ivy.array([5., 12.5, 17.]),
            b: ivy.array([0., 100., 7.5])
        }

        """
        return ContainerBase.cont_multi_map_in_static_method(
            "inplace_increment",
            x,
            val,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def inplace_increment(
        self: ivy.Container,
        val: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.inplace_increment.
        This method wraps the function, and so the docstring for
        ivy.inplace_increment also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input container to apply an in-place increment.
        val
            The value of increment.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Returns
        -------
        ret
            A container with the array following the in-place increment.

        Examples
        --------
        Using :class:`ivy.Container` instance method:
        >>> x = ivy.Container(a=ivy.array([-6.7, 2.4, -8.5]),
        ...                   b=ivy.array([1.5, -0.3, 0]),
        ...                   c=ivy.array([-4.7, -5.4, 7.5]))
        >>> y = x.inplace_increment(2)
        >>> print(y)
        {
            a: ivy.array([-4.7, 4.4, -6.5]),
            b: ivy.array([3.5, 1.7, 2.]),
            c: ivy.array([-2.7, -3.4, 9.5])
        }

        """
        return self.static_inplace_increment(
            self,
            val,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_assert_supports_inplace(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.assert_supports_inplace. This
        method simply wraps the function, and so the docstring for
        ivy.assert_supports_inplace also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container to check for inplace support for.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Returns
        -------
        ret
            True if support, raises exception otherwise`
        """
        return ContainerBase.cont_multi_map_in_static_method(
            "assert_supports_inplace",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def assert_supports_inplace(
        self: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.assert_supports_inplace. This
        method simply wraps the function, and so the docstring for
        ivy.assert_supports_inplace also applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container to check for inplace support for.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Returns
        -------
        ret
            a container of whether or not inplace operations are supported for x and
            its leaves.
        """
        return self.static_assert_supports_inplace(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_all_equal(
        x1: ivy.Container,
        *xs: Iterable[Any],
        equality_matrix: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.all_equal. This method simply wraps
        the function, and so the docstring for ivy.all_equal also applies to this method
        with minimal changes.

        Parameters
        ----------
        x1
            input container.
        xs
            arrays or containers to be compared to ``x1``.
        equality_matrix
            Whether to return a matrix of equalities comparing each input with every
            other. Default is ``False``.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Returns
        -------
        ret
            Boolean, whether or not the inputs are equal, or matrix container of
            booleans if equality_matrix=True is set.

        Examples
        --------
        With one :class:`ivy.Container` input:

        >>> x1 = ivy.Container(a=ivy.array([1, 0, 1, 1]), b=ivy.array([1, -1, 0, 0]))
        >>> x2 = ivy.array([1, 0, 1, 1])
        >>> y = ivy.Container.static_all_equal(x1, x2, equality_matrix= False)
        >>> print(y)
        {
            a: ivy.array([True, True, True, True]),
            b: ivy.array([True, False, False, False])
        }

        With multiple :class:`ivy.Container` input:

        >>> x1 = ivy.Container(a=ivy.array([1, 0, 1, 1]),
        ...                    b=ivy.native_array([1, 0, 0, 1]))
        >>> x2 = ivy.Container(a=ivy.native_array([1, 0, 1, 1]),
        ...                    b=ivy.array([1, 0, -1, -1]))
        >>> y = ivy.Container.static_all_equal(x1, x2, equality_matrix= False)
        >>> print(y)
        {
            a: ivy.array([True, True, True, True]),
            b: ivy.array([True, True, False, False])
        }

        """
        return ContainerBase.cont_multi_map_in_static_method(
            "all_equal",
            x1,
            *xs,
            equality_matrix=equality_matrix,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def all_equal(
        self: ivy.Container,
        *xs: Iterable[Any],
        equality_matrix: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.all_equal.
        This method simply wraps the function, and so the docstring for
        ivy.all_equal also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container.
        xs
            arrays or containers to be compared to ``self``.
        equality_matrix
            Whether to return a matrix of equalities comparing each input with every
            other. Default is ``False``.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Returns
        -------
        ret
            Boolean, whether or not the inputs are equal, or matrix container of
            booleans if equality_matrix=True is set.

        Examples
        --------
        With one :class:`ivy.Container` instances:

        >>> x1 = ivy.Container(a=ivy.array([1, 0, 1, 1]), b=ivy.array([1, -1, 0, 0]))
        >>> x2 = ivy.array([1, 0, 1, 1])
        >>> y = x1.all_equal(x2, equality_matrix= False)
        >>> print(y)
        {
            a: true,
            b: false
        }

        >>> x1 = ivy.Container(a=ivy.array([1, 0, 1, 1]), b=ivy.array([1, -1, 0, 0]))
        >>> x2 = ivy.array([1, 0, 1, 1])
        >>> y = ivy.Container.static_all_equal(x1, x2, equality_matrix= False)
        >>> print(y)
        {
            a: true,
            b: false
        }

        With multiple :class:`ivy.Container` instances:

        >>> x1 = ivy.Container(a=ivy.native_array([1, 0, 0]),
        ...                    b=ivy.array([1, 2, 3]))
        >>> x2 = ivy.Container(a=ivy.native_array([1, 0, 1]),
        ...                    b=ivy.array([1, 2, 3]))
        >>> y = x1.all_equal(x2, equality_matrix= False)
        >>> print(y)
        {
            a: false,
            b: true
        }

        >>> x1 = ivy.Container(a=ivy.native_array([1, 0, 0]),
        ...                    b=ivy.array([1, 2, 3]))
        >>> x2 = ivy.Container(a=ivy.native_array([1, 0, 1]),
        ...                    b=ivy.array([1, 2, 3]))
        >>> y = ivy.Container.static_all_equal(x1, x2, equality_matrix= False)
        >>> print(y)
        {
            a: false,
            b: true
        }

        """
        return self.static_all_equal(
            self,
            *xs,
            equality_matrix=equality_matrix,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_fourier_encode(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        max_freq: Union[float, ivy.Array, ivy.NativeArray],
        /,
        *,
        num_bands: int = 4,
        linear: bool = False,
        flatten: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.fourier_encode. This method
        simply wraps the function, and so the docstring for ivy.fourier_encode
        also applies to this method with minimal changes.

        Parameters
        ----------
        x
            Input container to apply fourier_encode.
        max_freq
            The maximum frequency of the encoding.
        num_bands
            The number of frequency bands for the encoding. Default is 4.
        linear
            Whether to space the frequency bands linearly as opposed to geometrically.
            Default is ``False``.
        flatten
            Whether to flatten the position dimension into the batch dimension.
            Default is ``False``.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Returns
        -------
        ret
            New container with the final dimension expanded of arrays at its leaves,
            and the encodings stored in this channel.
        """
        return ContainerBase.cont_multi_map_in_static_method(
            "fourier_encode",
            x,
            max_freq,
            num_bands=num_bands,
            linear=linear,
            concat=True,
            flatten=flatten,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def fourier_encode(
        self: ivy.Container,
        max_freq: Union[float, ivy.Array, ivy.NativeArray],
        /,
        *,
        num_bands: int = 4,
        linear: bool = False,
        flatten: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.fourier_encode. This method
        simply wraps the function, and so the docstring for ivy.fourier_encode
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input container to apply fourier_encode at leaves.
        max_freq
            The maximum frequency of the encoding.
        num_bands
            The number of frequency bands for the encoding. Default is 4.
        linear
            Whether to space the frequency bands linearly as opposed to geometrically.
            Default is ``False``.
        flatten
            Whether to flatten the position dimension into the batch dimension.
            Default is ``False``.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        dtype
            Data type of the returned array. Default is ``None``.
        out
            Optional output container. Default is ``None``.

        Returns
        -------
        ret
            New container with the final dimension expanded of arrays at its leaves,
            and the encodings stored in this channel.
        """
        return self.static_fourier_encode(
            self,
            max_freq,
            num_bands=num_bands,
            linear=linear,
            flatten=flatten,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_gather(
        params: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        indices: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        axis: Optional[int] = -1,
        batch_dims: Optional[int] = 0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.gather. This method simply
        wraps the function, and so the docstring for ivy.gather also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            The container from which to gather values.
        indices
            The container or array which indicates the indices that will be
            gathered along the specified axis.
        axis
            The axis from which the indices will be gathered. Default is ``-1``.
        batch_dims
            optional int, lets you gather different items from each element of a batch.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains will
            be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied. Default
            is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional array, for writing the result to. It must have a shape
            that the inputs broadcast to.


        Returns
        -------
        ret
            New container with the values gathered at the specified indices
            along the specified axis.

        Examples
        --------
        With :class:`ivy.Container` input:

        >>> x = ivy.Container(a = ivy.array([0., 1., 2.]),
        ...                   b = ivy.array([4., 5., 6.]))
        >>> y = ivy.Container(a = ivy.array([0, 1]),
        ...                   b = ivy.array([1, 2]))
        >>> print(ivy.Container.static_gather(x, y))
        {
            a: ivy.array([0., 1.]),
            b: ivy.array([5., 6.])
        }

        With a mix of :class:`ivy.Array` and :class:`ivy.Container` inputs:

        >>> x = ivy.Container(a = ivy.array([0., 1., 2.]),
        ...                   b = ivy.array([4., 5., 6.]))
        >>> y = ivy.array([0, 1])
        >>> print(ivy.Container.static_gather(x, y))
        {
            a: ivy.array([0., 1.]),
            b: ivy.array([4., 5.])
        }

        """
        return ContainerBase.cont_multi_map_in_static_method(
            "gather",
            params,
            indices,
            axis=axis,
            batch_dims=batch_dims,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def gather(
        self: ivy.Container,
        indices: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        axis: Optional[int] = -1,
        batch_dims: Optional[int] = 0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.gather. This method simply
        wraps the function, and so the docstring for ivy.gather also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            The container from which to gather values.
        indices
            The container or array which indicates the indices that will be
            gathered along the specified axis.
        axis
            The axis from which the indices will be gathered. Default is ``-1``.
        batch_dims
            optional int, lets you gather different items from each element of a batch.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is
            False.
        out
            optional array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            New container with the values gathered at the specified indices
            along the specified axis.

        Examples
        --------
        >>> x = ivy.Container(a = ivy.array([0., 1., 2.]),
        ...                   b = ivy.array([4., 5., 6.]))
        >>> y = ivy.Container(a = ivy.array([0, 1]),
        ...                   b = ivy.array([1, 2]))
        >>> print(x.gather(y))
        {
            a: ivy.array([0., 1.]),
            b: ivy.array([5., 6.])
        }

        """
        return self.static_gather(
            self,
            indices,
            axis=axis,
            batch_dims=batch_dims,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_has_nans(
        self: ivy.Container,
        /,
        *,
        include_infs=True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        Determine whether arrays in the container contain any nans, as well as infs
        or -infs if specified.

        Parameters
        ----------
        x
            The container to check for nans.
        include_infs
            Whether to include infs and -infs in the check. Default is True.

        Returns
        -------
            Whether the container has any nans, applied either leafwise or across the
            entire container.

        """
        return ContainerBase.cont_multi_map_in_static_method(
            "has_nans",
            self,
            include_infs=include_infs,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def has_nans(
        self: ivy.Container,
        /,
        *,
        include_infs=True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        Determine whether arrays in the container contain any nans, as well as infs
        or -infs if specified.

        Parameters
        ----------
        include_infs
            Whether to include infs and -infs in the check. Default is True.

        Returns
        -------
            Whether the container has any nans, applied across the entire container.
        """
        return self.static_has_nans(
            self,
            include_infs=include_infs,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_scatter_nd(
        indices: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        updates: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        shape: Union[ivy.Array, ivy.NativeArray, ivy.Container] = None,
        *,
        reduction: str = "sum",
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.scatter_nd. This method simply wraps
        the function, and so the docstring for ivy.scatter_nd also applies to this
        method with minimal changes.

        Parameters
        ----------
        indices
            Index array or container.
        updates
            values to update input tensor with
        shape
            The shape of the result. Default is ``None``, in which case tensor argument
            must be provided.
        reduction
            The reduction method for the scatter, one of 'sum', 'min', 'max'
            or 'replace'
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ref
            New container of given shape, with the values updated at the indices.

        Examples
        --------
        scatter into an empty array

        >>> indices = ivy.Container(a=ivy.array([[5],[6],[7]]),
        ...                         b=ivy.array([[2],[3],[4]]))
        >>> updates = ivy.Container(a=ivy.array([50, 60, 70]),
        ...                         b=ivy.array([20, 30, 40]))
        >>> shape = ivy.Container(a=ivy.array([10]),
        ...                       b=ivy.array([10]))
        >>> z = ivy.Container.static_scatter_nd(indices, updates, shape=shape)
        >>> print(z)
        {
            a: ivy.array([0, 0, 0, 0, 0, 50, 60, 70, 0, 0]),
            b: ivy.array([0, 0, 20, 30, 40, 0, 0, 0, 0, 0])
        }

        scatter into a container

        >>> indices = ivy.Container(a=ivy.array([[5],[6],[7]]),
        ...          b=ivy.array([[2],[3],[4]]))
        >>> updates = ivy.Container(a=ivy.array([50, 60, 70]),
        ...                         b=ivy.array([20, 30, 40]))
        >>> z = ivy.Container(a=ivy.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        ...                   b=ivy.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
        >>> ivy.Container.static_scatter_nd(indices, updates,
        ...                                    reduction='replace', out = z)
        >>> print(z)
        {
            a: ivy.array([1, 2, 3, 4, 5, 50, 60, 70, 9, 10]),
            b: ivy.array([1, 2, 20, 30, 40, 6, 7, 8, 9, 10])
        }

        """
        return ContainerBase.cont_multi_map_in_static_method(
            "scatter_nd",
            indices,
            updates,
            shape=shape,
            reduction=reduction,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def scatter_nd(
        self: ivy.Container,
        updates: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        shape: Union[ivy.Array, ivy.NativeArray, ivy.Container] = None,
        *,
        reduction: str = "sum",
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.scatter_nd.
        This method simply wraps the function, and so the docstring
        for ivy.scatter_nd also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            Index array or container.
        updates
            values to update input tensor with
        shape
            The shape of the result. Default is ``None``, in which case tensor argument
            must be provided.
        reduction
            The reduction method for the scatter, one of 'sum', 'min', 'max'
            or 'replace'
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            New container of given shape, with the values updated at the indices.

        Examples
        --------
        scatter into an empty container

        >>> indices = ivy.Container(a=ivy.array([[4],[3],[6]]),
        ...                         b=ivy.array([[5],[1],[2]]))
        >>> updates = ivy.Container(a=ivy.array([100, 200, 200]),
        ...                         b=ivy.array([20, 30, 40]))
        >>> shape = ivy.Container(a=ivy.array([10]),
        ...                       b=ivy.array([10]))
        >>> z = indices.scatter_nd(updates, shape=shape)
        >>> print(z)
        {
            a: ivy.array([0, 0, 0, 200, 100, 0, 200, 0, 0, 0]),
            b: ivy.array([0, 30, 40, 0, 0, 20, 0, 0, 0, 0])
        }

        scatter into a container.

        >>> indices = ivy.Container(a=ivy.array([[5],[6],[7]]),
        ...                         b=ivy.array([[2],[3],[4]]))
        >>> updates = ivy.Container(a=ivy.array([50, 60, 70]),
        ...                         b=ivy.array([20, 30, 40]))
        >>> z = ivy.Container(a=ivy.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        ...                   b=ivy.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
        >>> indices.scatter_nd(updates,reduction='replace', out = z)
        >>> print(z)
        {
            a: ivy.array([1, 2, 3, 4, 5, 50, 60, 70, 9, 10]),
            b: ivy.array([1, 2, 20, 30, 40, 6, 7, 8, 9, 10])
        }
        """
        return self.static_scatter_nd(
            self,
            updates,
            shape=shape,
            reduction=reduction,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_scatter_flat(
        indices: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        updates: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        size: Optional[int] = None,
        reduction: str = "sum",
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.scatter_flat. This method simply
        wraps the function, and so the docstring for ivy.scatter_flat also applies to
        this method with minimal changes.

        Parameters
        ----------
        indices
            Index array or container.
        updates
            values to update input tensor with
        size
            The size of the result.
        reduction
            The reduction method for the scatter, one of 'sum', 'min', 'max'
            or 'replace'
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ref
            New container of given shape, with the values updated at the indices.
        """
        return ContainerBase.cont_multi_map_in_static_method(
            "scatter_flat",
            indices,
            updates,
            size=size,
            reduction=reduction,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def scatter_flat(
        self: ivy.Container,
        updates: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        size: Optional[int] = None,
        reduction: str = "sum",
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.scatter_flat.
        This method simply wraps the function, and so the docstring
        for ivy.scatter_flat also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            Index array or container.
        updates
            values to update input tensor with
        size
            The size of the result.
        reduction
            The reduction method for the scatter, one of 'sum', 'min', 'max'
            or 'replace'
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            New container of given shape, with the values updated at the indices.
        """
        return self.static_scatter_flat(
            self,
            updates,
            size=size,
            reduction=reduction,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_gather_nd(
        params: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        indices: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        batch_dims: Optional[int] = 0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """Gather slices from all container params into a arrays with shape specified by
        indices.

        Parameters
        ----------
        indices
            Index array.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains will
            be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied. Default
            is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Returns
        -------
            Container object with all sub-array dimensions gathered.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([[0., 10., 20.],[30.,40.,50.]]),
        ...                   b=ivy.array([[0., 100., 200.],[300.,400.,500.]]))
        >>> y = ivy.Container(a=ivy.array([1,0]),
        ...                   b=ivy.array([0]))
        >>> print(ivy.Container.static_gather_nd(x, y))
        {
            a: ivy.array(30.),
            b: ivy.array([0., 100., 200.])
        }

        """
        return ContainerBase.cont_multi_map_in_static_method(
            "gather_nd",
            params,
            indices,
            batch_dims=batch_dims,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def gather_nd(
        self: ivy.Container,
        indices: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        batch_dims: Optional[int] = 0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.gather_nd.
        This method simply wraps the function, and so the docstring
        for ivy.gather_nd also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            The container from which to gather values.
        indices
            Index array or container.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        device
            device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Same as
            ``x`` if None.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            New container of given shape, with the values gathered at the indices.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([[[0., 10.], [20.,30.]],
        ...                                [[40.,50.], [60.,70.]]]),
        ...                   b=ivy.array([[[0., 100.], [200.,300.]],
        ...                                [[400.,500.],[600.,700.]]]))
        >>> y = ivy.Container(a=ivy.array([1,0]),
        ...                   b=ivy.array([0]))
        >>> z = x.gather_nd(y)
        >>> print(z)
        {
            a: ivy.array([40., 50.]),
            b: ivy.array([[0., 100.],
                        [200., 300.]])
        }
        """
        return self.static_gather_nd(
            self,
            indices,
            batch_dims=batch_dims,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_einops_reduce(
        x: ivy.Container,
        pattern: str,
        reduction: str,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
        **axes_lengths: Dict[str, int],
    ) -> ivy.Container:
        """Perform einops reduce operation on each sub array in the container.

        Parameters
        ----------
        pattern
            Reduction pattern.
        reduction
            One of available reductions ('min', 'max', 'sum', 'mean', 'prod'), or
            callable.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains will
            be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied. Default
            is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        axes_lengths
            Any additional specifications for dimensions.

        Returns
        -------
            ivy.Container with each array having einops.reduce applied.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([[-4.47, 0.93, -3.34],
        ...                                [3.66, 24.29, 3.64]]),
        ...                   b=ivy.array([[4.96, 1.52, -10.67],
        ...                                [4.36, 13.96, 0.3]]))
        >>> reduced = ivy.Container.static_einops_reduce(x, 'a b -> a', 'mean')
        >>> print(reduced)
        {
            a: ivy.array([-2.29, 10.5]),
            b: ivy.array([-1.4, 6.21])
        }

        """
        return ContainerBase.cont_multi_map_in_static_method(
            "einops_reduce",
            x,
            pattern,
            reduction,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
            **axes_lengths,
        )

    def einops_reduce(
        self: ivy.Container,
        pattern: str,
        reduction: Union[str, Callable],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
        **axes_lengths: Dict[str, int],
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.einops_reduce. This method simply
        wraps the function, and so the docstring for ivy.einops_reduce also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            Input container to be reduced.
        pattern
            Reduction pattern.
        reduction
            One of available reductions ('min', 'max', 'sum', 'mean', 'prod'), or
            callable.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains will
            be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied. Default
            is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output container, for writing the result to. It must have a
            shape that the inputs broadcast to.
        axes_lengths
            Any additional specifications for dimensions.

        Returns
        -------
        ret
            New container with einops.reduce having been applied.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([[[5, 4, 3],
        ...                                 [11, 2, 9]],
        ...                                [[3, 5, 7],
        ...                                 [9, 7, 1]]]),
        ...                    b=ivy.array([[[9,7,6],
        ...                                  [5,2,1]],
        ...                                 [[4,1,2],
        ...                                  [2,3,6]],
        ...                                 [[1, 9, 6],
        ...                                  [0, 2, 1]]]))
        >>> reduced = x.einops_reduce('a b c -> a b', 'sum')
        >>> print(reduced)
        {
            a: ivy.array([[12, 22],
                        [15, 17]]),
            b: ivy.array([[22, 8],
                        [7, 11],
                        [16, 3]])
        }

        """
        return self.static_einops_reduce(
            self,
            pattern,
            reduction,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
            **axes_lengths,
        )

    @staticmethod
    def static_einops_repeat(
        x: ivy.Container,
        pattern: str,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
        **axes_lengths: Dict[str, int],
    ) -> ivy.Container:
        """Perform einops repeat operation on each sub array in the container.

        Parameters
        ----------
        pattern
            Rearrangement pattern.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains will
            be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied. Default
            is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        axes_lengths
            Any additional specifications for dimensions.
        **axes_lengths

        Returns
        -------
            ivy.Container with each array having einops.repeat applied.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([[30, 40], [50, 75]]),
        ...                   b=ivy.array([[1, 2], [4, 5]]))
        >>> repeated = ivy.Container.static_einops_repeat(
        ...    x, 'h w -> (tile h) w', tile=2)
        >>> print(repeated)
        {
            a: ivy.array([[30, 40],
                        [50, 75],
                        [30, 40],
                        [50, 75]]),
            b: ivy.array([[1, 2],
                        [4, 5],
                        [1, 2],
                        [4, 5]])
        }

        """
        return ContainerBase.cont_multi_map_in_static_method(
            "einops_repeat",
            x,
            pattern,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
            **axes_lengths,
        )

    def einops_repeat(
        self: ivy.Container,
        pattern: str,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
        **axes_lengths: Dict[str, int],
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.einops_repeat. This method simply
        wraps the function, and so the docstring for ivy.einops_repeat also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            Input array or container to be repeated.
        pattern
            Rearrangement pattern.
        axes_lengths
            Any additional specifications for dimensions.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output container, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
        ret
            New container with einops.repeat having been applied.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([[30, 40], [50, 75]]),
        ...                   b=ivy.array([[1, 2], [4, 5]]))
        >>> repeated = x.einops_repeat('h w ->  h  (w tile)', tile=2)
        >>> print(repeated)
        {
            a: ivy.array([[30, 30, 40, 40],
                          [50, 50, 75, 75]]),
            b: ivy.array([[1, 1, 2, 2],
                          [4, 4, 5, 5]])
        }

        """
        return self.static_einops_repeat(
            self,
            pattern,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
            **axes_lengths,
        )

    @staticmethod
    def static_value_is_nan(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        include_infs: bool = True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.value_is_nan. This method simply
        wraps the function, and so the docstring for ivy.value_is_nan also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input container.
        include_infs
            Whether to include infs and -infs in the check. Default is ``True``.
        key_chains
            The key-chains to apply or not apply the method to. Default is
            None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains will
            be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied. Default
            is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Returns
        -------
        ret
            Boolean as to whether the input value is a nan or not.

        Examples
        --------
        With :class:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([452]), b=ivy.array([float('inf')]))
        >>> y = ivy.Container.static_value_is_nan(x)
        >>> print(y)
        {
            a: false,
            b: true
        }

        With :class:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([float('nan')]), b=ivy.array([0]))
        >>> y = ivy.Container.static_value_is_nan(x)
        >>> print(y)
        {
            a: true,
            b: false
        }

        With :class:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([float('inf')]), b=ivy.array([22]))
        >>> y = ivy.Container.static_value_is_nan(x, include_infs=False)
        >>> print(y)
        {
            a: false,
            b: false
        }
        """
        return ContainerBase.cont_multi_map_in_static_method(
            "value_is_nan",
            x,
            include_infs=include_infs,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def value_is_nan(
        self: ivy.Container,
        /,
        *,
        include_infs: bool = True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.value_is_nan. This method simply
        wraps the function, and so the docstring for ivy.value_is_nan also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input container.
        include_infs
            Whether to include infs and -infs in the check. Default is ``True``.
        key_chains
            The key-chains to apply or not apply the method to. Default is
            None.
        to_apply
            If True, the method will be applied to key_chains, otherwise
            key_chains will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Returns
        -------
        ret
            Boolean as to whether the input value is a nan or not.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([425]), b=ivy.array([float('nan')]))
        >>> y = x.value_is_nan()
        >>> print(y)
        {
            a: false,
            b: true
        }

        >>> x = ivy.Container(a=ivy.array([float('inf')]), b=ivy.array([0]))
        >>> y = x.value_is_nan()
        >>> print(y)
        {
            a: true,
            b: false
        }

        >>> x = ivy.Container(a=ivy.array([float('inf')]), b=ivy.array([22]))
        >>> y = x.value_is_nan(include_infs=False)
        >>> print(y)
        {
            a: false,
            b: false
        }
        """
        return self.static_value_is_nan(
            self,
            include_infs=include_infs,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_to_numpy(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        copy: bool = True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.to_numpy. This method simply wraps
        the function, and so the docstring for ivy.to_numpy also applies to this method
        with minimal changes.

        Parameters
        ----------
        x
            input container.
        copy
            Whether to copy the input. Default is ``True``.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Returns
        -------
        ret
            a container of numpy arrays copying all the element of the container
            ``self``.

        Examples
        --------
        With one :class:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([1, 0, 1, 1]),
        ...                   b=ivy.array([1, -1, 0, 0]))
        >>> y = ivy.Container.static_to_numpy(x)
        >>> print(y)
        {
            a: array([1, 0, 1, 1], dtype=int32),
            b: array([1, -1, 0, 0], dtype=int32)
        }

        >>> x = ivy.Container(a=ivy.array([1., 0., 0., 1.]),
        ...                   b=ivy.native_array([1, 1, -1, 0]))
        >>> y = ivy.Container.static_to_numpy(x)
        >>> print(y)
        {
            a: array([1., 0., 0., 1.], dtype=float32),
            b: array([1, 1, -1, 0], dtype=int32)
        }

        """
        return ContainerBase.cont_multi_map_in_static_method(
            "to_numpy",
            x,
            copy=copy,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def to_numpy(
        self: ivy.Container,
        /,
        *,
        copy: bool = True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.to_numpy. This method simply wraps
        the function, and so the docstring for ivy.to_numpy also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input container.
        copy
            Whether to copy the input. Default is ``True``.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Returns
        -------
        ret
            a container of numpy arrays copying all the element of the container
            ``self``.

        Examples
        --------
        With one :class:`ivy.Container` instances:

        >>> x = ivy.Container(a=ivy.array([-1, 0, 1]), b=ivy.array([1, 0, 1, 1]))
        >>> y = x.to_numpy()
        >>> print(y)
        {
            a: array([-1, 0, 1], dtype=int32),
            b: array([1, 0, 1, 1], dtype=int32)
        }

        >>> x = ivy.Container(a=ivy.native_array([[-1, 0, 1], [-1, 0, 1], [1, 0, -1]]),
        ...                   b=ivy.native_array([[-1, 0, 0], [1, 0, 1], [1, 1, 1]]))
        >>> y = x.to_numpy()
        >>> print(y)
        {
            a: array([[-1, 0, 1],
                    [-1, 0, 1],
                    [1, 0, -1]], dtype=int32),
            b: array([[-1, 0, 0],
                    [1, 0, 1],
                    [1, 1, 1]], dtype=int32)
        }

        """
        return self.static_to_numpy(
            self,
            copy=copy,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_to_scalar(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.to_scalar. This method simply wraps
        the function, and so the docstring for ivy.to_scalar also applies to this method
        with minimal changes.

        Parameters
        ----------
        x
            input container.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Returns
        -------
        ret
            a container of scalar values copying all the element of the container
            ``x``.

        Examples
        --------
        With one :class:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([-1]), b=ivy.array([3]))
        >>> y = ivy.Container.static_to_scalar(x)
        >>> print(y)
        {
            a: -1,
            b: 3
        }
        """
        return ContainerBase.cont_multi_map_in_static_method(
            "to_scalar",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def to_scalar(
        self: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.to_scalar. This method simply wraps
        the function, and so the docstring for ivy.to_scalar also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input container.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Returns
        -------
        ret
            a container of scalar values copying all the element of the container
            ``self``.

        Examples
        --------
        With one :class:`ivy.Container` instance:


        >>> x = ivy.Container(a=ivy.array([1]), b=ivy.array([0]),
        ...                   c=ivy.array([-1]))
        >>> y = x.to_scalar()
        >>> print(y)
        {
            a: 1,
            b: 0,
            c: -1
        }
        """
        return self.static_to_scalar(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_to_list(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.to_list. This method simply wraps
        the function, and so the docstring for ivy.to_list also applies to this method
        with minimal changes.

        Parameters
        ----------
        x
            input container.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Returns
        -------
        ret
            A list representation of the input array ``x``.

        Examples
        --------
        With one :class:`ivy.Container` inputs:


        >>> x = ivy.Container(a=ivy.array([0, 1, 2]))
        >>> y = ivy.Container.static_to_list(x)
        >>> print(y)
        {a:[0,1,2]}

        """
        return ContainerBase.cont_multi_map_in_static_method(
            "to_list",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def to_list(
        self: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.to_list. This method simply wraps
        the function, and so the docstring for ivy.to_list also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input container.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Returns
        -------
        ret
            A list representation of the input array ``x``.

        Examples
        --------
        With one :class:`ivy.Container` instances:


        >>> x = ivy.Container(a=ivy.array([0, 1, 2]))
        >>> y = x.to_list()
        >>> print(y)
        {a:[0,1,2]}

        """
        return self.static_to_list(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_stable_divide(
        numerator: ivy.Container,
        denominator: Union[Number, ivy.Array, ivy.Container],
        /,
        *,
        min_denominator: Union[
            Number, ivy.Array, ivy.NativeArray, ivy.Container
        ] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.stable_divide. This method simply
        wraps the function, and so the docstring for ivy.stable_divide also applies
        to this method with minimal changes.

        Parameters
        ----------
        numerator
            Container of the numerators of the division.
        denominator
            Container of the denominators of the division.
        min_denominator
            Container of the minimum denominator to use,
            use global ivy._MIN_DENOMINATOR by default.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Returns
        -------
        ret
            A container of elements containing the new items following the numerically
            stable division.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.asarray([10., 15.]), b=ivy.asarray([20., 25.]))
        >>> y = ivy.Container.stable_divide(x, 0.5)
        >>> print(y)
        {
            a: ivy.array([20., 30.]),
            b: ivy.array([40., 50.])
        }

        >>> x = ivy.Container(a=1, b=10)
        >>> y = ivy.asarray([4, 5])
        >>> z = ivy.Container.stable_divide(x, y)
        >>> print(z)
        {
            a: ivy.array([0.25, 0.2]),
            b: ivy.array([2.5, 2.])
        }

        >>> x = ivy.Container(a=1, b=10)
        >>> y = np.array((4.5, 9))
        >>> z = ivy.Container.stable_divide(x, y)
        >>> print(z)
        {
            a: array([0.22222222, 0.11111111]),
            b: array([2.22222222, 1.11111111])
        }


        >>> x = ivy.Container(a=ivy.asarray([1., 2.]), b=ivy.asarray([3., 4.]))
        >>> y = ivy.Container(a=ivy.asarray([0.5, 2.5]), b=ivy.asarray([3.5, 0.4]))
        >>> z = ivy.Container.stable_divide(x, y)
        >>> print(z)
        {
            a: ivy.array([2., 0.8]),
            b: ivy.array([0.857, 10.])
        }

        >>> x = ivy.Container(a=ivy.asarray([1., 2.], [3., 4.]),
        ...                   b=ivy.asarray([5., 6.], [7., 8.]))
        >>> y = ivy.Container(a=ivy.asarray([0.5, 2.5]), b=ivy.asarray([3.5, 0.4]))
        >>> z = ivy.Container.stable_divide(x, y, min_denominator=2)
        >>> print(z)
        {
            a: ivy.array([0.4, 0.444]),
            b: ivy.array([0.909, 2.5])
        }
        """
        return ContainerBase.cont_multi_map_in_static_method(
            "stable_divide",
            numerator,
            denominator,
            min_denominator=min_denominator,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def stable_divide(
        self,
        denominator: Union[Number, ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        min_denominator: Union[
            Number, ivy.Array, ivy.NativeArray, ivy.Container
        ] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.stable_divide. This method
        simply wraps the function, and so the docstring for ivy.stable_divide
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container.
        denominator
            Container of the denominators of the division.
        min_denominator
            Container of the minimum denominator to use,
            use global ivy._MIN_DENOMINATOR by default.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Returns
        -------
        ret
            a container of numpy arrays copying all the element of the container
            ``self``.
            A container of elements containing the new items following the numerically
            stable division, using ``self`` as the numerator.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.asarray([3., 6.]), b=ivy.asarray([9., 12.]))
        >>> y = x.stable_divide(5)
        >>> print(y)
        {
            a: ivy.array([0.6, 1.2]),
            b: ivy.array([1.8, 2.4])
        }

        >>> x = ivy.Container(a=ivy.asarray([[2., 4.], [6., 8.]]),
        ...                   b=ivy.asarray([[10., 12.], [14., 16.]]))
        >>> z = x.stable_divide(2, min_denominator=2)
        >>> print(z)
        {
            a: ivy.array([[0.5, 1.],
                  [1.5, 2.]]),
            b: ivy.array([[2.5, 3.],
                  [3.5, 4.]])
        }

        >>> x = ivy.Container(a=ivy.asarray([3., 6.]), b=ivy.asarray([9., 12.]))
        >>> y = ivy.Container(a=ivy.asarray([6., 9.]), b=ivy.asarray([12., 15.]))
        >>> z = x.stable_divide(y)
        >>> print(z)
        {
            a: ivy.array([0.5, 0.667]),
            b: ivy.array([0.75, 0.8])
        }

        """
        return self.static_stable_divide(
            self,
            denominator,
            min_denominator=min_denominator,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_stable_pow(
        base: ivy.Container,
        exponent: Union[Number, ivy.Array, ivy.Container],
        /,
        *,
        min_base: float = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.stable_pow. This method
        simply wraps the function, and so the docstring for ivy.stable_pow also
        applies to this method with minimal changes.

        Parameters
        ----------
        base
            Container of the base.
        exponent
            Container of the exponent.
        min_base
            The minimum base to use, use global ivy._MIN_BASE by default.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise
            key_chains will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is
            False.

        Returns
        -------
        ret
            A container of elements containing the new items following the
            numerically stable power.

        """
        return ContainerBase.cont_multi_map_in_static_method(
            "stable_pow",
            base,
            exponent,
            min_base=min_base,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def stable_pow(
        self,
        exponent: Union[Number, ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        min_base: float = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.stable_pow. This method
        simply wraps the function, and so the docstring for ivy.stable_pow
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Container of the base.
        exponent
            Container of the exponent.
        min_base
            The minimum base to use, use global ivy._MIN_BASE by default.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise
            key_chains will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is
            False.

        Returns
        -------
        ret
            A container of elements containing the new items following the
            numerically stable power.

        """
        return self.static_stable_pow(
            self,
            exponent,
            min_base=min_base,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_einops_rearrange(
        x: ivy.Container,
        pattern: str,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
        **axes_lengths: Dict[str, int],
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.einops_rearrange. This method
        simply wraps the function, and so the docstring for ivy.einops_rearrange
        also applies to this method with minimal changes.

        Parameters
        ----------
        pattern
            Rearrangement pattern.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains will
            be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied. Default
            is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        axes_lengths
            Any additional specifications for dimensions.


        Returns
        -------
            ivy.Container with each array having einops.rearrange applied.

        """
        return ContainerBase.cont_multi_map_in_static_method(
            "einops_rearrange",
            x,
            pattern,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
            **axes_lengths,
        )

    def einops_rearrange(
        self: ivy.Container,
        pattern: str,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
        **axes_lengths: Dict[str, int],
    ):
        """
        ivy.Container instance method variant of ivy.einops_rearrange. This method
        simply wraps the function, and so the docstring for ivy.einops_rearrange
        also applies to this method with minimal changes.

        Parameters
        ----------
        pattern
            Rearrangement pattern.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains will
            be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied. Default
            is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        axes_lengths
            Any additional specifications for dimensions.
        **axes_lengths


        Returns
        -------
            ivy.Container with each array having einops.rearrange applied.

        """
        return self.static_einops_rearrange(
            self,
            pattern,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
            **axes_lengths,
        )

    @staticmethod
    def static_clip_matrix_norm(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        max_norm: float,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        p: float = 2.0,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.clip_matrix_norm. This method
        simply wraps the function, and so the docstring for ivy.clip_matrix_norm
        also applies to this method with minimal changes.

        Parameters
        ----------
        x
            Input array containing elements to clip.
        max_norm
            The maximum value of the array norm.
        p
            The p-value for computing the p-norm. Default is 2.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output array, for writing the result to. It must
            have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            An array with the matrix norm downscaled to the max norm if needed.

        Examples
        --------
        With :class:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([[0., 1., 2.]]),
        ...                   b=ivy.array([[3., 4., 5.]]))
        >>> y = ivy.Container.static_clip_matrix_norm(x, 2.0)
        >>> print(y)
        {
            a: ivy.array([[0., 0.894, 1.79]]),
            b: ivy.array([[0.849, 1.13, 1.41]])
        }

        """
        return ContainerBase.cont_multi_map_in_static_method(
            "clip_matrix_norm",
            x,
            max_norm,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            p=p,
            out=out,
        )

    def clip_matrix_norm(
        self: ivy.Container,
        max_norm: float,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        p: float = 2.0,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.clip_matrix_norm. This method
        simply wraps the function, and so the docstring for ivy.clip_matrix_norm
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input array containing elements to clip.
        max_norm
            The maximum value of the array norm.
        p
            The p-value for computing the p-norm. Default is 2.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output array, for writing the result to. It must
            have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            An array with the matrix norm downscaled to the max norm if needed.

        Examples
        --------
        With :class:`ivy.Container` instance method:

        >>> x = ivy.Container(a=ivy.array([[0., 1., 2.]]),
        ...                   b=ivy.array([[3., 4., 5.]]))
        >>> y = x.clip_matrix_norm(2.0, p=1.0)
        >>> print(y)
        {
            a: ivy.array([[0., 1., 2.]]),
            b: ivy.array([[1.2, 1.6, 2.]])
        }

        """
        return self.static_clip_matrix_norm(
            self,
            max_norm,
            p=p,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_supports_inplace_updates(
        x: Union[ivy.Dtype, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.supports_inplace. This method
        simply wraps the function, and so the docstring for ivy.supports_inplace
        also applies to this method with minimal changes.

        Parameters
        ----------
        x
            An ivy.Container.
        key_chains
            The key-chains to apply or not apply the method to.
            Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped.
            Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Returns
        -------
        ret
            An ivy.Container instance of bool values.
            True if nodes of x support in-place operations. False otherwise.

        Raises
        ------
        ValueError
            If a node(s) of the container isn't a class instance of ivy.Array or
            ivy.NativeArray, an exception will be raised.

        Examples
        --------
        With `ivy.Container` input and backend set as 'numpy':

        >>> x = ivy.Container(a = ivy.array(1.0), b=ivy.array(2))
        >>> ret = ivy.Container.static_supports_inplace_updates(x)
        >>> print(ret)
        {
            a: true,
            b: true
        }

        With `ivy.Container` input and backend set as 'tensorflow':

        >>> x = ivy.Container(a=ivy.array(ivy.array([2.0, 0.0])),
        ...                   b=ivy.array([0., 5.5, -8]))
        >>> ret = ivy.Container.static_supports_inplace_updates(x)
        >>> print(ret)
        {
            a: true,
            b: false
        }
        """
        return ContainerBase.cont_multi_map_in_static_method(
            "supports_inplace_updates",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def supports_inplace_updates(
        self: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.supports_inplace_updates.
        This method simply wraps the static function, and so the docstring for
        the static variant also applies to this method with minimal changes.

        Parameters
        ----------
        self
            An ivy.Container whose elements are data types supported by Ivy.
        key_chains
            The key-chains to apply or not apply the method to.
            Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped.
            Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Returns
        -------
        ret
            An ivy.Container instance of bool values.
            True if nodes of the Container support in-place operations. False otherwise.

        Examples
        --------
        With an `ivy.Container` instance and backend set as 'numpy':
        >>> x = ivy.Container(a = ivy.array(1.0), b=ivy.array(2))
        >>> ret = x.supports_inplace_updates()
        >>> print(ret)
        {
            a: false,
            b: false
        }

        With an `ivy.Container` instance and backend set as 'tensorflow':
        >>> x = ivy.Container(a=ivy.variable(ivy.array([2.0, 0.0])),
        ...                   b=ivy.array([0., 5.5, -8]))
        >>> ret = x.supports_inplace_updates()
        >>> print(ret)
        {
            a: false,
            b: false
        }
        """
        return ContainerWithGeneral.static_supports_inplace_updates(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_get_num_dims(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        as_array: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.get_num_dims. This method
        simply wraps the function, and so the docstring for ivy.get_num_dims
        also applies to this method with minimal changes.

        Parameters
        ----------
        x
            ivy.Container to infer the number of dimensions for
        as_array
            Whether to return the shape as a array, default False.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.


        Returns
        -------
        ret
            Shape of the array

        Examples
        --------
        >>> x = ivy.Container(b = ivy.asarray([[0.,1.,1.],[1.,0.,0.],[8.,2.,3.]]))
        >>> ivy.Container.static_get_num_dims(x)
        {
            b: 2
        }
        >>> x = ivy.Container(b = ivy.array([[[0,0,0],[0,0,0],[0,0,0]]
        ...                                    [[0,0,0],[0,0,0],[0,0,0]],
        ...                                    [[0,0,0],[0,0,0],[0,0,0]]]))
        >>> ivy.Container.static_get_num_dims(x)
        {
            b: 3
        }
        >>> x = ivy.Container(b = ivy.array([[[0,0,0],[0,0,0],[0,0,0]],
        ...                                    [[0,0,0],[0,0,0],[0,0,0]]]),
        ...                                    c = ivy.asarray([[0.,1.,1.],[8.,2.,3.]]))
        >>> ivy.Container.static_get_num_dims(x)
        {
            b: 3,
            c: 2
        }
        >>> ivy.Container.static_get_num_dims(x, as_array=True)
        {
            b: ivy.array(3),
            c: ivy.array(2)
        }
        """
        return ContainerBase.cont_multi_map_in_static_method(
            "get_num_dims",
            x,
            as_array=as_array,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def get_num_dims(
        self: ivy.Container,
        /,
        *,
        as_array: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.get_num_dims. This method
        simply wraps the function, and so the docstring for ivy.get_num_dims
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            ivy.Container to infer the number of dimensions for
        as_array
            Whether to return the shape as a array, default False.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.


        Returns
        -------
        ret
            Shape of the array

        Examples
        --------
        >>> a = ivy.Container(b = ivy.asarray([[0.,1.,1.],[1.,0.,0.],[8.,2.,3.]]))
        >>> a.get_num_dims()
        {
            b: 2
        }
        >>> a = ivy.Container(b = ivy.array([[[0,0,0],[0,0,0],[0,0,0]],
        ...                                    [[0,0,0],[0,0,0],[0,0,0]],
        ...                                    [[0,0,0],[0,0,0],[0,0,0]]]))
        >>> a.get_num_dims()
        {
            b: 3
        }
        >>> a = ivy.Container(b = ivy.array([[[0,0,0],[0,0,0],[0,0,0]],
        ...                                    [[0,0,0],[0,0,0],[0,0,0]]]),
        ...                                    c = ivy.asarray([[0.,1.,1.],[8.,2.,3.]]))
        >>> a.get_num_dims()
        {
            b: 3,
            c: 2
        }
        >>> a.get_num_dims(as_array=True)
        {
            b: ivy.array(3),
            c: ivy.array(2)
        }
        """
        return ContainerWithGeneral.static_get_num_dims(
            self,
            as_array=as_array,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )
