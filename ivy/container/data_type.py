# global
from typing import Optional, Union, List, Dict, Tuple, Callable

# local
import ivy
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class ContainerWithDataTypes(ContainerBase):
    @staticmethod
    def static_astype(
        x: ivy.Container,
        dtype: ivy.Dtype,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        copy: bool = True,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """Copies an array to a specified data type irrespective of :ref:`type-promotion`
        rules.

        .. note::
        Casting floating-point ``NaN`` and ``infinity`` values to integral data types is
        not specified and is implementation-dependent.

        .. note::
        When casting a boolean input array to a numeric data type, a value of ``True``
        must cast to a numeric value equal to ``1``, and a value of ``False`` must cast
        to a numeric value equal to ``0``.

        When casting a numeric input array to ``bool``, a value of ``0`` must cast to
        ``False``, and a non-zero value must cast to ``True``.

        Parameters
        ----------
        x
            array to cast.
        dtype
            desired data type.
        copy
            specifies whether to copy an array when the specified ``dtype`` matches the data
            type of the input array ``x``. If ``True``, a newly allocated array must always
            be returned. If ``False`` and the specified ``dtype`` matches the data type of
            the input array, the input array must be returned; otherwise, a newly allocated
            must be returned. Default: ``True``.
        out
            optional output array, for writing the result to. It must have a shape that the
            inputs broadcast to.

        Returns
        -------
        ret
            an array having the specified data type. The returned array must have the same
            shape as ``x``.

        Examples
        --------

        >>> c = ivy.Container(a=ivy.array([False,True,True]), b=ivy.array([3.14, 2.718, 1.618]))
        >>> ivy.Container.static_astype(c, int)
        {
            a: ivy.array([0, 1, 1]),
            b: ivy.array([3, 2, 1])
        }
        """
    
        return ContainerBase.multi_map_in_static_method(
            "astype",
            x,
            dtype,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            copy=copy,
            out=out,
        )

    def astype(
        self: ivy.Container,
        dtype: ivy.Dtype,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        copy: bool = True,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """Copies an array to a specified data type irrespective of :ref:`type-promotion`
        rules.

        .. note::
        Casting floating-point ``NaN`` and ``infinity`` values to integral data types is
        not specified and is implementation-dependent.

        .. note::
        When casting a boolean input array to a numeric data type, a value of ``True``
        must cast to a numeric value equal to ``1``, and a value of ``False`` must cast
        to a numeric value equal to ``0``.

        When casting a numeric input array to ``bool``, a value of ``0`` must cast to
        ``False``, and a non-zero value must cast to ``True``.

        Parameters
        ----------
        x
            array to cast.
        dtype
            desired data type.
        copy
            specifies whether to copy an array when the specified ``dtype`` matches the data
            type of the input array ``x``. If ``True``, a newly allocated array must always
            be returned. If ``False`` and the specified ``dtype`` matches the data type of
            the input array, the input array must be returned; otherwise, a newly allocated
            must be returned. Default: ``True``.
        out
            optional output array, for writing the result to. It must have a shape that the
            inputs broadcast to.

        Returns
        -------
        ret
            an array having the specified data type. The returned array must have the same
            shape as ``x``.

        Examples
        --------

        Using :code:`ivy.Container` instance method:
        
        >>> x = ivy.Container(a=ivy.array([False,True,True]), \
                                b=ivy.array([3.14, 2.718, 1.618]))
        >>> x.astype(ivy.int32)
        {
            a: ivy.array([0, 1, 1]),
            b: ivy.array([1, 0, 1])
        }

        """
        return self.static_astype(
            self,
            dtype,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            copy=copy,
            out=out,
        )

    @staticmethod
    def static_broadcast_arrays(
        *arrays: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "broadcast_arrays",
            *arrays,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def broadcast_arrays(
        self: ivy.Container,
        *arrays: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        return self.static_broadcast_arrays(
            self,
            *arrays,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_broadcast_to(
        x: ivy.Container,
        shape: Tuple[int, ...],
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "broadcast_to",
            x,
            shape,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def broadcast_to(
        self: ivy.Container,
        shape: Tuple[int, ...],
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_broadcast_to(
            self,
            shape,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_can_cast(
        from_: ivy.Container,
        to: ivy.Dtype,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        `ivy.Container` static method variant of `ivy.can_cast`. This method simply
        wraps the function, and so the docstring for `ivy.can_cast` also applies to
        this method with minimal changes.

        Parameters
        ----------
        from_
            input container from which to cast.
        to
            desired data type.
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

        Returns
        -------
        ret
            ``True`` if the cast can occur according to :ref:`type-promotion` rules;
            otherwise, ``False``.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), \
            b=ivy.array([3, 4, 5]))
        >>> print(x.a.dtype, x.b.dtype)
        float32 int32

        >>> print(ivy.Container.static_can_cast(x, 'int64'))
        {
            a: false,
            b: true
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "can_cast",
            from_,
            to,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def can_cast(
        self: ivy.Container,
        to: ivy.Dtype,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        `ivy.Container` instance method variant of `ivy.can_cast`. This method simply
        wraps the function, and so the docstring for `ivy.can_cast` also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input container from which to cast.
        to
            desired data type.
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

        Returns
        -------
        ret
            ``True`` if the cast can occur according to :ref:`type-promotion` rules;
            otherwise, ``False``.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), \
            b=ivy.array([3, 4, 5]))
        >>> print(x.a.dtype, x.b.dtype)
        float32 int32

        >>> print(x.can_cast('int64'))
        {
            a: false,
            b: true
        }
        """
        return self.static_can_cast(
            self, to, key_chains, to_apply, prune_unapplied, map_sequences
        )

    @staticmethod
    def static_dtype(
        x: ivy.Container,
        as_native: Optional[bool] = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "dtype",
            x,
            as_native,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def dtype(
        self: ivy.Container,
        as_native: Optional[bool] = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_dtype(
            self,
            as_native,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_default_float_dtype(
        input=None,
        float_dtype: Optional[Union[ivy.FloatDtype, ivy.NativeDtype]] = None,
        as_native: Optional[bool] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "default_float_dtype",
            input,
            float_dtype,
            as_native,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_function_supported_dtypes(
        fn: Callable,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "function_supported_dtypes",
            fn,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_function_unsupported_dtypes(
        fn: Callable,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "function_unsupported_dtypes",
            fn,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_finfo(
        type: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "finfo",
            type,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def finfo(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        return self.static_finfo(
            self, key_chains, to_apply, prune_unapplied, map_sequences
        )

    @staticmethod
    def static_iinfo(
        type: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "iinfo",
            type,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def iinfo(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        return self.static_iinfo(
            self, key_chains, to_apply, prune_unapplied, map_sequences
        )

    @staticmethod
    def static_is_bool_dtype(
        dtype_in: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "is_bool_dtype",
            dtype_in,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def is_bool_dtype(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        return self.static_is_bool_dtype(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_is_float_dtype(
        dtype_in: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "is_float_dtype",
            dtype_in,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def is_float_dtype(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        return self.static_is_int_dtype(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_is_int_dtype(
        dtype_in: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "is_int_dtype",
            dtype_in,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def is_int_dtype(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        return self.static_is_int_dtype(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_is_uint_dtype(
        dtype_in: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "is_uint_dtype",
            dtype_in,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def is_uint_dtype(
        self: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        return self.static_is_uint_dtype(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_result_type(
        *arrays_and_dtypes: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "result_type",
            *arrays_and_dtypes,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def result_type(
        self: ivy.Container,
        *arrays_and_dtypes: ivy.Container,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        return self.static_result_type(
            self,
            *arrays_and_dtypes,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )
