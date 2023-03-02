# global
from typing import Optional, Union, List, Tuple, Dict, Sequence
from numbers import Number
import numpy as np

# local
import ivy
from ivy.container.base import ContainerBase


class _ContainerWithCreation(ContainerBase):
    @staticmethod
    def static_arange(
        start: Number,
        /,
        stop: Optional[Number] = None,
        step: Number = 1,
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
            "arange",
            start,
            stop,
            step,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out,
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def static_asarray(
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
        out: Optional[ivy.Container] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    ) -> ivy.Container:
        return ContainerBase.cont_multi_map_in_function(
            "asarray",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
            copy=copy,
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def static_zeros(
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
    def static_ones(
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
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out,
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def static_full_like(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
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
        ivy.Container static method variant of ivy.full_like. This method simply wraps
        the function, and so the docstring for ivy.full_like also applies to this
        method with minimal changes.

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
            fill_value,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
            dtype=dtype,
            device=device,
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
        return self.static_full_like(
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
    def static_ones_like(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
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
        """
        ivy.Container static method variant of ivy.ones_like. This method simply
        wraps the function, and so the docstring for ivy.ones_like also applies
        to this method with minimal changes.

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
            out=out,
            dtype=dtype,
            device=device,
        )

    def ones_like(
        self: ivy.Container,
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
        """
        ivy.Container instance method variant of ivy.ones_like. This method simply
        wraps the function, and so the docstring for ivy.ones_like also applies
        to this method with minimal changes.

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
        return self.static_ones_like(
            self,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def static_zeros_like(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
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
        """
        ivy.Container static method variant of ivy.zeros_like. This method simply
        wraps the function, and so the docstring for ivy.zeros_like also applies
        to this method with minimal changes.

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
            out=out,
            dtype=dtype,
            device=device,
        )

    def zeros_like(
        self: ivy.Container,
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
        """
        ivy.Container instance method variant of ivy.zeros_like. This method simply
        wraps the function, and so the docstring for ivy.zeros_like also applies
        to this method with minimal changes.

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
        return self.static_zeros_like(
            self,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def static_tril(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        k: int = 0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.cont_multi_map_in_function(
            "tril",
            x,
            k,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out,
        )

    def tril(
        self: ivy.Container,
        /,
        k: int = 0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_tril(
            self,
            k,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
        )

    @staticmethod
    def static_triu(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        k: int = 0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.cont_multi_map_in_function(
            "triu",
            x,
            k,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out,
        )

    def triu(
        self: ivy.Container,
        /,
        k: int = 0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_triu(
            self,
            k,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
        )

    @staticmethod
    def static_empty_like(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
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
            "empty_like",
            x,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out,
            dtype=dtype,
            device=device,
        )

    def empty_like(
        self: ivy.Container,
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
        return self.static_empty_like(
            self,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def static_eye(
        n_rows: int,
        n_cols: Optional[int] = None,
        /,
        k: int = 0,
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
            "eye",
            n_rows,
            n_cols,
            k,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out,
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def static_linspace(
        start: Union[ivy.Array, ivy.NativeArray, float],
        stop: Union[ivy.Array, ivy.NativeArray, float],
        /,
        num: int,
        axis: Optional[int] = None,
        endpoint: bool = True,
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
            "linspace",
            start,
            stop,
            num,
            axis=axis,
            endpoint=endpoint,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
            dtype=dtype,
            device=device,
        )

    def linspace(
        self: ivy.Container,
        stop: Union[ivy.Array, ivy.NativeArray, float, ivy.Container],
        /,
        num: int,
        axis: Optional[int] = None,
        endpoint: bool = True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    ) -> ivy.Container:
        return self.static_linspace(
            self,
            stop,
            num,
            axis=axis,
            endpoint=endpoint,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def static_meshgrid(
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
            sparse,
            indexing,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
        )

    def meshgrid(
        self: ivy.Container,
        /,
        *arrays: Union[ivy.Array, ivy.NativeArray, List[Number], Tuple[Number]],
        sparse: bool = False,
        indexing: str = "xy",
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        return self.static_meshgrid(
            tuple([self] + list(arrays)),
            sparse,
            indexing,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
        )

    @staticmethod
    def static_from_dlpack(
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
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out,
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
        return self.static_from_dlpack(
            self,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
        )

    @staticmethod
    def static_native_array(
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
        out: Optional[ivy.Container] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    ) -> ivy.Container:
        return ContainerBase.cont_multi_map_in_function(
            "native_array",
            x,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out,
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
        out: Optional[ivy.Container] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    ) -> ivy.Container:
        return self.static_native_array(
            self,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def static_logspace(
        start: Union[ivy.Array, ivy.NativeArray, float, ivy.Container],
        stop: Union[ivy.Array, ivy.NativeArray, float, ivy.Container],
        /,
        num: int,
        base: float = 10.0,
        axis: int = 0,
        endpoint: bool = True,
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
        ivy.Container static method variant of ivy.logspace. This method simply
        wraps the function, and so the docstring for ivy.logspace also applies
        to this method with minimal changes.

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
            num,
            base=base,
            axis=axis,
            endpoint=endpoint,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            device=device,
            out=out,
        )

    def logspace(
        self: ivy.Container,
        stop: Union[ivy.Array, ivy.NativeArray, float, ivy.Container],
        /,
        num: int,
        *,
        base: float = 10.0,
        axis: int = None,
        endpoint: bool = True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.logspace. This method simply
        wraps the function, and so the docstring for ivy.logspace also applies
        to this method with minimal changes.

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
        return self.static_logspace(
            self,
            stop,
            num,
            base=base,
            axis=axis,
            endpoint=endpoint,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            device=device,
            out=out,
        )

    @staticmethod
    def static_one_hot(
        indices: ivy.Container,
        depth: int,
        /,
        *,
        on_value: Optional[Number] = None,
        off_value: Optional[Number] = None,
        axis: Optional[int] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.one_hot. This method
        simply wraps the function, and so the docstring for ivy.one_hot
        also applies to this method with minimal changes.

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
            on_value=on_value,
            off_value=off_value,
            axis=axis,
            dtype=dtype,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def one_hot(
        self: ivy.Container,
        depth: int,
        /,
        *,
        on_value: Optional[Number] = None,
        off_value: Optional[Number] = None,
        axis: Optional[int] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.one_hot. This method
        simply wraps the function, and so the docstring for ivy.one_hot
        also applies to this method with minimal changes.

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
        return self.static_one_hot(
            self,
            depth,
            on_value=on_value,
            off_value=off_value,
            axis=axis,
            dtype=dtype,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )
