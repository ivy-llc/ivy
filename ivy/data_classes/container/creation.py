# global
from typing import Optional, Union, List, Tuple, Dict, Sequence
from numbers import Number
import numpy as np

# local
import ivy
from ivy.data_classes.container.base import ContainerBase


class _ContainerWithCreation(ContainerBase):
    @staticmethod
    def _static_arange(
        start: Number,
        /,
        stop: Optional[Number] = None,
        step: Number = 1,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.cont_multi_map_in_function(
            "arange",
            start,
            stop=stop,
            step=step,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            device=device,
            out=out,
        )

    @staticmethod
    def _static_asarray(
        x: Union[
            ivy.Array,
            ivy.NativeArray,
            List[Number],
            Tuple[Number],
            np.ndarray,
            ivy.Container,
        ],
        /,
        copy: Optional[bool] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.asarray. This method simply wraps the
        function, and so the docstring for ivy.asarray also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            input data, in any form that can be converted to an array. This includes
            lists, lists of tuples, tuples, tuples of tuples, tuples of lists and
            ndarrays.
        copy
            boolean indicating whether or not to copy the input array.
            If True, the function must always copy.
            If False, the function must never copy and must
            raise a ValueError in case a copy would be necessary.
            If None, the function must reuse existing memory buffer if possible
            and copy otherwise. Default: ``None``.
        dtype
            datatype, optional. Datatype is inferred from the input data.
        device
            device on which to place the created array. Default: ``None``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            An array interpretation of ``self``.

        Examples
        --------
        With :class:`ivy.Container` as input:
        >>> x = ivy.Container(a = [(1,2),(3,4),(5,6)], b = ((1,2,3),(4,5,6)))
        >>> ivy.asarray(x)
        {
            a: ivy.array([[1, 2],
                          [3, 4],
                          [5, 6]]),
            b: ivy.array([[1, 2, 3],
                          [4, 5, 6]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "asarray",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            copy=copy,
            dtype=dtype,
            device=device,
            out=out,
        )

    def asarray(
        self: ivy.Container,
        /,
        copy: Optional[bool] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self._static_asarray(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            copy=copy,
            dtype=dtype,
            device=device,
            out=out,
        )

    @staticmethod
    def _static_zeros(
        shape: Union[int, Sequence[int]],
        /,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    ) -> ivy.Container:
        return ContainerBase.cont_multi_map_in_function(
            "zeros",
            shape,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def _static_ones(
        shape: Union[int, Sequence[int]],
        /,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    ) -> ivy.Container:
        return ContainerBase.cont_multi_map_in_function(
            "ones",
            shape,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def _static_empty(
        shape: Union[int, Sequence[int]],
        /,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.cont_multi_map_in_function(
            "empty",
            shape,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            device=device,
            out=out,
        )

    @staticmethod
    def _static_full(
        shape: Union[ivy.Shape, ivy.NativeShape],
        fill_value: Union[float, bool],
        /,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        out: Optional[ivy.Container] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    ) -> ivy.Container:
        return ContainerBase.cont_multi_map_in_function(
            "full",
            shape,
            fill_value,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def _static_full_like(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        fill_value: Union[int, float],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.full_like. This method simply wraps
        the function, and so the docstring for ivy.full_like also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input container.
        fill_value
            Scalar fill value
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
        dtype
            output array data type. If ``dtype`` is `None`, the output array data type
            must be inferred from ``self``. Default: ``None``.
        device
            device on which to place the created array. If ``device`` is ``None``, the
            output array device must be inferred from ``self``. Default: ``None``.

        Returns
        -------
        ret
            an output container having the same data type as ``x`` and whose elements,
            relative to ``x``, are shifted.

        Examples
        --------
        With :class:`ivy.Container` input:

        >>> x = ivy.Container(a = ivy.array([1,2,3]) ,b = ivy.array([4,5,6]))
        >>> fill_value = 10
        >>> y = ivy.Container.static_full_like(fill_value)
        {
            a: ivy.array([10, 10, 10]),
            b: ivy.array([10, 10, 10])
        }

        >>> x = ivy.Container(a=ivy.array([1.2, 2.2324, 3.234]),
        ...                   b=ivy.array([4.123, 5.23, 6.23]))
        >>> fill_value = 15.0
        >>> y = ivy.Container.static_full_like(fill_value)
        >>> print(y)
        {
            a: ivy.array([15., 15., 15.]),
            b: ivy.array([15., 15., 15.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "full_like",
            x,
            fill_value=fill_value,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            device=device,
            out=out,
        )

    def full_like(
        self: ivy.Container,
        /,
        fill_value: Union[int, float],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.full_like. This method simply wraps
        the function, and so the docstring for ivy.full_like also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input container.
        fill_value
            Scalar fill value
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
        dtype
            output array data type. If ``dtype`` is `None`, the output array data type
            must be inferred from ``self``. Default: ``None``.
        device
            device on which to place the created array. If ``device`` is ``None``, the
            output array device must be inferred from ``self``. Default: ``None``.

        Returns
        -------
        ret
            an output container having the same data type as ``x`` and whose elements,
            relative to ``x``, are shifted.

        Examples
        --------
        With :class:`ivy.Container` input:

        >>> x = ivy.Container(a = ivy.array([1,2,3]) ,b = ivy.array([4,5,6]))
        >>> fill_value = 10
        >>> y = x.full_like(fill_value)
        {
            a: ivy.array([10, 10, 10]),
            b: ivy.array([10, 10, 10])
        }

        >>> x = ivy.Container(a=ivy.array([1.2,2.2324,3.234]),
        ...                   b=ivy.array([4.123,5.23,6.23]))
        >>> fill_value = 15.0
        >>> y = x.full_like(fill_value)
        >>> print(y)
        {
            a: ivy.array([15., 15., 15.]),
            b: ivy.array([15., 15., 15.])
        }
        """
        return self._static_full_like(
            self,
            fill_value,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def _static_ones_like(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.ones_like. This method simply wraps
        the function, and so the docstring for ivy.ones_like also applies to this method
        with minimal changes.

        Parameters
        ----------
        x
            input array from which to derive the output array shape.
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
            output array data type. If ``dtype`` is ``None``, the output array data type
            must be inferred from ``self``. Default  ``None``.
        device
            device on which to place the created array. If device is ``None``, the
            output array device must be inferred from ``self``. Default: ``None``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            a container having the same shape as ``self`` and filled with ones.
        """
        return ContainerBase.cont_multi_map_in_function(
            "ones_like",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            device=device,
            out=out,
        )

    def ones_like(
        self: ivy.Container,
        /,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.ones_like. This method simply wraps
        the function, and so the docstring for ivy.ones_like also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array from which to derive the output array shape.
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
            output array data type. If ``dtype`` is ``None``, the output array data type
            must be inferred from ``self``. Default  ``None``.
        device
            device on which to place the created array. If device is ``None``, the
            output array device must be inferred from ``self``. Default: ``None``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            a container having the same shape as ``self`` and filled with ones.
        """
        return self._static_ones_like(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            device=device,
            out=out,
        )

    @staticmethod
    def _static_zeros_like(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.zeros_like. This method simply wraps
        the function, and so the docstring for ivy.zeros_like also applies to this
        method with minimal changes.

        Parameters
        ----------
        x
            input array or container from which to derive the output container shape.
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
            output array data type. If ``dtype`` is ``None``, the output container
            data type must be inferred from ``self``. Default  ``None``.
        device
            device on which to place the created array. If device is ``None``, the
            output container device must be inferred from ``self``. Default: ``None``.
        out
            optional output container, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
        ret
            an container having the same shape as ``x`` and filled with ``zeros``.
        """
        return ContainerBase.cont_multi_map_in_function(
            "zeros_like",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            device=device,
            out=out,
        )

    def zeros_like(
        self: ivy.Container,
        /,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.zeros_like. This method simply
        wraps the function, and so the docstring for ivy.zeros_like also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            input array or container from which to derive the output container shape.
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
            output array data type. If ``dtype`` is ``None``, the output container
            data type must be inferred from ``self``. Default: ``None``.
        device
            device on which to place the created array. If device is ``None``, the
            output container device must be inferred from ``self``. Default: ``None``.
        out
            optional output container, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
        ret
            an container having the same shape as ``x`` and filled with ``zeros``.
        """
        return self._static_zeros_like(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def _static_tril(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        k: int = 0,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.cont_multi_map_in_function(
            "tril",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            k=k,
            out=out,
        )

    def tril(
        self: ivy.Container,
        /,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        k: int = 0,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self._static_tril(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            k=k,
            out=out,
        )

    @staticmethod
    def _static_triu(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        k: int = 0,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.cont_multi_map_in_function(
            "triu",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            k=k,
            out=out,
        )

    def triu(
        self: ivy.Container,
        /,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        k: int = 0,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self._static_triu(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            k=k,
            out=out,
        )

    @staticmethod
    def _static_empty_like(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.cont_multi_map_in_function(
            "empty_like",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            device=device,
            out=out,
        )

    def empty_like(
        self: ivy.Container,
        /,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self._static_empty_like(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            device=device,
            out=out,
        )

    @staticmethod
    def _static_eye(
        n_rows: int,
        n_cols: Optional[int] = None,
        /,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        k: int = 0,
        batch_shape: Optional[Union[int, Sequence[int]]] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.cont_multi_map_in_function(
            "eye",
            n_rows,
            n_cols,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            k=k,
            batch_shape=batch_shape,
            dtype=dtype,
            device=device,
            out=out,
        )

    @staticmethod
    def _static_linspace(
        start: Union[ivy.Array, ivy.NativeArray, float, ivy.Container],
        stop: Union[ivy.Array, ivy.NativeArray, float, ivy.Container],
        /,
        num: int,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        axis: Optional[int] = None,
        endpoint: bool = True,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.cont_multi_map_in_function(
            "linspace",
            start,
            stop,
            num,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            axis=axis,
            endpoint=endpoint,
            dtype=dtype,
            device=device,
            out=out,
        )

    def linspace(
        self: ivy.Container,
        stop: Union[ivy.Array, ivy.NativeArray, float, ivy.Container],
        /,
        num: int,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        axis: Optional[int] = None,
        endpoint: bool = True,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self._static_linspace(
            self,
            stop,
            num,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            axis=axis,
            endpoint=endpoint,
            dtype=dtype,
            device=device,
            out=out,
        )

    @staticmethod
    def _static_meshgrid(
        *arrays: Union[ivy.Array, ivy.NativeArray, List[Number], Tuple[Number]],
        sparse: bool = False,
        indexing: str = "xy",
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        return ContainerBase.cont_multi_map_in_function(
            "meshgrid",
            *arrays,
            sparse=sparse,
            indexing=indexing,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def meshgrid(
        self: ivy.Container,
        *arrays: Union[ivy.Array, ivy.NativeArray, List[Number], Tuple[Number]],
        sparse: bool = False,
        indexing: str = "xy",
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        return self._static_meshgrid(
            self,
            *arrays,
            sparse=sparse,
            indexing=indexing,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def _static_from_dlpack(
        x: Union[ivy.Array, ivy.NativeArray],
        /,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.cont_multi_map_in_function(
            "from_dlpack",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def from_dlpack(
        self: ivy.Container,
        /,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self._static_from_dlpack(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_copy_array(
        x: Union[ivy.Array, ivy.NativeArray],
        /,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        to_ivy_array: bool = True,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.cont_multi_map_in_function(
            "copy_array",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            to_ivy_array=to_ivy_array,
            out=out,
        )

    def copy_array(
        self: ivy.Container,
        /,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        to_ivy_array: bool = True,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self._static_copy_array(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            to_ivy_array=to_ivy_array,
            out=out,
        )

    @staticmethod
    def _static_native_array(
        x: Union[
            ivy.Array,
            ivy.NativeArray,
            List[Number],
            Tuple[Number],
            np.ndarray,
            ivy.Container,
        ],
        /,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    ) -> ivy.Container:
        return ContainerBase.cont_multi_map_in_function(
            "native_array",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            device=device,
        )

    def native_array(
        self: ivy.Container,
        /,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    ) -> ivy.Container:
        return self._static_native_array(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def _static_logspace(
        start: Union[ivy.Array, ivy.NativeArray, float, ivy.Container],
        stop: Union[ivy.Array, ivy.NativeArray, float, ivy.Container],
        /,
        num: int,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        base: float = 10.0,
        axis: int = 0,
        endpoint: bool = True,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.logspace. This method simply wraps
        the function, and so the docstring for ivy.logspace also applies to this method
        with minimal changes.

        Parameters
        ----------
        start
            Container for first value in the range in log space.
        stop
            Container for last value in the range in log space.
        num
            Number of values to generate.
        base
            The base of the log space. Default is 10.0
        axis
            Axis along which the operation is performed. Relevant only if values in
            start or stop containers are array-like. Default is 0.
        endpoint
            If True, stop is the last sample. Otherwise, it is not included. Default is
            True.
        dtype
            The data type of the output tensor. If None, the dtype of on_value is used
            or if that is None, the dtype of off_value is used, or if that is None,
            defaults to float32. Default is None.
        device
            device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Default
            is None.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to. Default is None.

        Returns
        -------
        ret
            a container having the same shape as ``start`` and filled with tensor of
            evenly-spaced values in log space.

        Examples
        --------
        >>> import ivy.container.creation.static_logspace as static_logspace
        >>> x = ivy.Container(a = 1, b = 0)
        >>> y = ivy.Container(a = 4, b = 1)
        >>> z = static_logspace(x, y, 4)
        {
            a: ivy.array([10.,  100.,  1000., 10000.]),
            b: ivy.array([ 1., 2.15443469, 4.64158883, 10.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "logspace",
            start,
            stop,
            num=num,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            base=base,
            axis=axis,
            endpoint=endpoint,
            dtype=dtype,
            device=device,
            out=out,
        )

    def logspace(
        self: ivy.Container,
        stop: Union[ivy.Array, ivy.NativeArray, float, ivy.Container],
        /,
        num: int,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        base: float = 10.0,
        axis: int = None,
        endpoint: bool = True,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.logspace. This method simply wraps
        the function, and so the docstring for ivy.logspace also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            Container for first value in the range in log space.
        stop
            Container for last value in the range in log space.
        num
            Number of values to generate.
        base
            The base of the log space. Default is 10.0
        axis
            Axis along which the operation is performed. Relevant only if values in
            start or stop containers are array-like. Default is 0.
        endpoint
            If True, stop is the last sample. Otherwise, it is not included. Default is
            True.
        dtype
            The data type of the output tensor. If None, the dtype of on_value is used
            or if that is None, the dtype of off_value is used, or if that is None,
            defaults to float32. Default is None.
        device
            device on which to create the array 'cuda:0', 'cuda:1', 'cpu' etc. Default
            is None.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to. Default is None.

        Returns
        -------
        ret
            a container having the same shape as ``self`` and filled with tensor of
            evenly-spaced values in log space.

        Examples
        --------
        >>> x = ivy.Container(a = 1, b = 0)
        >>> y = ivy.Container(a = 4, b = 1)
        >>> z = x.logspace(y, 4)
        {
            a: ivy.array([10.,  100.,  1000., 10000.]),
            b: ivy.array([ 1., 2.15443469, 4.64158883, 10.])
        }

        >>> x = ivy.Container(a = 1, b = 0)
        >>> y = ivy.Container(a = 4, b = 1)
        >>> z = ivy.logspace(x, y, 4)
        {
            a: ivy.array([10.,  100.,  1000., 10000.]),
            b: ivy.array([ 1., 2.15443469, 4.64158883, 10.])
        }

        >>> u = ivy.Container(c = 0, d = 0)
        >>> v = ivy.Container(c = 1, d = 2)
        >>> x = ivy.Container(a = 1, b = u)
        >>> y = ivy.Container(a = 4, b = v)
        >>> z = x.logspace(y, 4)
        {
            a: ivy.array([10.,  100.,  1000., 10000.]),
            b:  {
                    c: ivy.array([ 1., 2.15443469, 4.64158883, 10.])
                    d: ivy.array([ 1., 4.64158883, 21.5443469, 100.])
                }
        }
        """
        return self._static_logspace(
            self,
            stop,
            num=num,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            base=base,
            axis=axis,
            endpoint=endpoint,
            dtype=dtype,
            device=device,
            out=out,
        )

    @staticmethod
    def _static_one_hot(
        indices: ivy.Container,
        depth: int,
        /,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        on_value: Optional[Number] = None,
        off_value: Optional[Number] = None,
        axis: Optional[int] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Union[ivy.Device, ivy.NativeDevice] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.one_hot. This method simply wraps the
        function, and so the docstring for ivy.one_hot also applies to this method with
        minimal changes.

        Parameters
        ----------
        indices
            Indices for where the ones should be scattered *[batch_shape, dim]*
        depth
            Scalar defining the depth of the one-hot dimension.
        on_value
            Value to fill in output when indices[j] = i. If None, defaults to 1.
        off_value
            Value to fill in output when indices[j] != i. If None, defaults to 0.
        axis
            Axis to scatter on. The default is ``-1``, a new inner-most axis is created.
        dtype
            The data type of the output tensor. If None, defaults to the on_value dtype
            or the off_value dtype. If both are None, defaults to float32.
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
            container with tensors of zeros with the same shape and type as the inputs,
            unless dtype provided which overrides.
        
        Examples
        --------
        With :class:`ivy.Container` inputs:

        >>> x = ivy.Container(a=ivy.array([1, 2]), \
            b=ivy.array([3, 1]), c=ivy.array([2, 3]))
        >>> y = 5
        >>> z = ivy.Container.static_one_hot(x, y)
        >>> print(z)
        {
            a: ivy.array([[0., 1., 0., 0., 0.], 
                        [0., 0., 1., 0., 0.]]),
            b: ivy.array([[0., 0., 0., 1., 0.], 
                        [0., 1., 0., 0., 0.]]),
            c: ivy.array([[0., 0., 1., 0., 0.], 
                        [0., 0., 0., 1., 0.]])
        }

        >>> x = ivy.Container(a=ivy.array([1, 2]), \
            b=ivy.array([]), c=ivy.native_array([4]))
        >>> y = 5
        >>> z = ivy.Container.static_one_hot(x, y)
        >>> print(z)
        {
            a: ivy.array([[0., 1., 0., 0., 0.], 
                        [0., 0., 1., 0., 0.]]),
            b: ivy.array([], shape=(0, 5)),
            c: ivy.array([[0., 0., 0., 0., 1.]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "one_hot",
            indices,
            depth,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            on_value=on_value,
            off_value=off_value,
            axis=axis,
            dtype=dtype,
            device=device,
            out=out,
        )

    def one_hot(
        self: ivy.Container,
        depth: int,
        /,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        on_value: Optional[Number] = None,
        off_value: Optional[Number] = None,
        axis: Optional[int] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Union[ivy.Device, ivy.NativeDevice] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.one_hot. This method simply wraps
        the function, and so the docstring for ivy.one_hot also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            Indices for where the ones should be scattered *[batch_shape, dim]*
        depth
            Scalar defining the depth of the one-hot dimension.
        on_value
            Value to fill in output when indices[j] == i. If None, defaults to 1.
        off_value
            Value to fill in output when indices[j] != i. If None, defaults to 0.
        axis
            Axis to scatter on. The default is ``-1``, a new inner-most axis is created.
        dtype
            The dtype of the returned tensor. If None, defaults to the on_value dtype
            or the off_value dtype. If both are None, defaults to float32.
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

        Returns
        -------
        ret
            container with tensors of zeros with the same shape and type as the inputs,
            unless dtype provided which overrides.

        Examples
        --------
        With :class:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([1, 2]), \
             b=ivy.array([3, 1]), c=ivy.array([2, 3]))
        >>> y = 5
        >>> z = x.one_hot(y)
        >>> print(z)
        {
            a: ivy.array([[0., 1., 0., 0., 0.], 
                        [0., 0., 1., 0., 0.]]),
            b: ivy.array([[0., 0., 0., 1., 0.], 
                        [0., 1., 0., 0., 0.]]),
            c: ivy.array([[0., 0., 1., 0., 0.], 
                        [0., 0., 0., 1., 0.]])
        }

        >>> x = ivy.Container(a=ivy.array([1, 2]), \
             b=ivy.array([]), c=ivy.native_array([4]))
        >>> y = 5
        >>> z = x.one_hot(y)
        >>> print(z)
        {
            a: ivy.array([[0., 1., 0., 0., 0.], 
                        [0., 0., 1., 0., 0.]]),
            b: ivy.array([], shape=(0, 5)),
            c: ivy.array([[0., 0., 0., 0., 1.]])
        }
        """
        return self._static_one_hot(
            self,
            depth,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            on_value=on_value,
            off_value=off_value,
            axis=axis,
            dtype=dtype,
            device=device,
            out=out,
        )

    @staticmethod
    def static_frombuffer(
        buffer: ivy.Container,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = float,
        count: Optional[int] = -1,
        offset: Optional[int] = 0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        r"""
        ivy.Container static method variant of ivy.frombuffer. This method simply wraps
        the function, and so the docstring for ivy.frombuffer also applies to this
        method with minimal changes.

        Parameters
        ----------
        buffer
            An object that exposes the buffer interface.
        dtype
            Data-type of the returned array; default: float.
        count
            Number of items to read. -1 means all data in the buffer.
        offset
            Start reading the buffer from this offset (in bytes); default: 0.
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
        out
            1-dimensional array.

        Examples
        --------
        With :class:`ivy.Container` inputs:

        >>> x = ivy.Container(
        ...     a = b'\x00\x00\x00\x00\x00\x00\xf0?',
        ...     b = b'\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00@'
        ... )
        >>> y = ivy.Container.static_frombuffer(x)
        >>> print(y)
        {
            a: ivy.array([1.]),
            b: ivy.array([1., 2.])
        }

        >>> x = ivy.Container(
        ...     a = b'\x01\x02\x03\x04',
        ...     b = b'\x05\x04\x03\x03\x02'
        ... )
        >>> y = ivy.Container.static_frombuffer(x, dtype=ivy.int8, count=3, offset=1)
        >>> print(y)
        {
            a: ivy.array([2, 3, 4]),
            b: ivy.array([4, 3, 3])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "frombuffer",
            buffer,
            dtype=dtype,
            count=count,
            offset=offset,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def frombuffer(
        self: ivy.Container,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = float,
        count: Optional[int] = -1,
        offset: Optional[int] = 0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        r"""
        ivy.Container instance method variant of ivy.frombuffer. This method simply
        wraps the function, and so the docstring for ivy.frombuffer also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            An object that exposes the buffer interface.
        dtype
            Data-type of the returned array; default: float.
        count
            Number of items to read. -1 means all data in the buffer.
        offset
            Start reading the buffer from this offset (in bytes); default: 0.
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
        out
            1-dimensional array.

        Examples
        --------
        With :class:`ivy.Container` inputs:

        >>> x = ivy.Container(
        ...     a = b'\x00\x00\x00\x00\x00\x00\xf0?',
        ...     b = b'\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00@'
        ... )
        >>> y = ivy.Container.static_frombuffer(x)
        >>> print(y)
        {
            a: ivy.array([1.]),
            b: ivy.array([1., 2.])
        }

        >>> x = ivy.Container(
        ...     a = b'\x01\x02\x03\x04',
        ...     b = b'\x05\x04\x03\x03\x02'
        ... )
        >>> y = ivy.frombuffer(x, dtype=ivy.int8, count=3, offset=1)
        >>> print(y)
        {
            a: ivy.array([2, 3, 4]),
            b: ivy.array([4, 3, 3])
        }
        """
        return self.static_frombuffer(
            self,
            dtype=dtype,
            count=count,
            offset=offset,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_triu_indices(
        n_rows: int,
        n_cols: Optional[int] = None,
        k: int = 0,
        /,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[Tuple[ivy.Array]] = None,
    ) -> ivy.Container:
        return ContainerBase.cont_multi_map_in_function(
            "triu_indices",
            n_rows,
            n_cols,
            k,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            device=device,
            out=out,
        )

    def triu_indices(
        self: ivy.Container,
        n_rows: int,
        n_cols: Optional[int] = None,
        k: int = 0,
        /,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[Tuple[ivy.Array]] = None,
    ) -> ivy.Container:
        return self.static_triu_indices(
            self,
            n_rows,
            n_cols,
            k,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            device=device,
            out=out,
        )
