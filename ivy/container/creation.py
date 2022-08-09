# global
from typing import Optional, Union, List, Tuple, Dict
from numbers import Number
import numpy as np

# local
import ivy
from ivy.container.base import ContainerBase


# noinspection PyMissingConstructor
class ContainerWithCreation(ContainerBase):
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
        return ContainerBase.multi_map_in_static_method(
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
        return ContainerBase.multi_map_in_static_method(
            "asarray",
            x,
            copy,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out,
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def static_zeros(
        shape: Union[int, Tuple[int], List[int]],
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
        return ContainerBase.multi_map_in_static_method(
            "zeros",
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
    def static_ones(
        shape: Union[int, Tuple[int], List[int]],
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
        return ContainerBase.multi_map_in_static_method(
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
        With ivy.Container input
        >>> x = ivy.Container(a = ivy.array([1,2,3]) ,b = ivy.array([4,5,6]))
        >>> fill_value = 10
        >>> y = ivy.Container.full_like(fill_value)
        {
            a: ivy.array([10, 10, 10]),
            b: ivy.array([10, 10, 10])
        }

        >>> x = ivy.Container(a=ivy.array([1.2,2.2324,3.234]), \
                               b=ivy.array([4.123,5.23,6.23]))
        >>> fill_value = 15.0
        >>> y = ivy.Container.full_like(fill_value)
        >>> print(y)
        {
            a: ivy.array([15., 15., 15.]),
            b: ivy.array([15., 15., 15.])
        }
        """
        return ContainerBase.multi_map_in_static_method(
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
        With ivy.Container input
        >>> x = ivy.Container(a = ivy.array([1,2,3]) ,b = ivy.array([4,5,6]))
        >>> fill_value = 10
        >>> y = x.full_like(fill_value)
        {
            a: ivy.array([10, 10, 10]),
            b: ivy.array([10, 10, 10])
        }

        >>> x = ivy.Container(a=ivy.array([1.2,2.2324,3.234]), \
                               b=ivy.array([4.123,5.23,6.23]))
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
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
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
        return ContainerBase.multi_map_in_static_method(
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
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
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
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
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
        return ContainerBase.multi_map_in_static_method(
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
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
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
        return ContainerBase.multi_map_in_static_method(
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
        return ContainerBase.multi_map_in_static_method(
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
        return ContainerBase.multi_map_in_static_method(
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
        k: Optional[int] = 0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
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
        axis: int = None,
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
        return ContainerBase.multi_map_in_static_method(
            "linspace",
            start,
            stop,
            num,
            axis,
            endpoint,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def static_meshgrid(
        *arrays: Union[ivy.Array, ivy.NativeArray, List[Number], Tuple[Number]],
        indexing: Optional[str] = "xy",
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "meshgrid",
            *arrays,
            indexing,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out,
        )

    def meshgrid(
        self: ivy.Container,
        /,
        *arrays: Union[ivy.Array, ivy.NativeArray, List[Number], Tuple[Number]],
        indexing: Optional[str] = "xy",
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.handle_inplace(
            self.map(
                lambda x_: ivy.meshgrid([x_._data] + list(arrays))
                if ivy.is_array(x_)
                else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
            out,
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
        return ContainerBase.multi_map_in_static_method(
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
        return ContainerBase.multi_map_in_static_method(
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
        start: Union[ivy.Array, ivy.NativeArray, float],
        stop: Union[ivy.Array, ivy.NativeArray, float],
        /,
        num: int,
        base: float = 10.0,
        axis: int = None,
        endpoint: bool = True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice]] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "logspace",
            start,
            stop,
            num,
            base,
            axis,
            endpoint,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out,
            device=device,
        )
