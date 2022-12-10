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
        """Copies an array to a specified data type irrespective of
        :ref:`type-promotion` rules.

        .. note::
        Casting floating-point ``NaN`` and ``infinity`` values to integral data types
        is not specified and is implementation-dependent.

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
            specifies whether to copy an array when the specified ``dtype`` matches
            the data type of the input array ``x``. If ``True``, a newly allocated
            array must always be returned. If ``False`` and the specified ``dtype``
            matches the data type of the input array, the input array must be returned;
            otherwise, a newly allocated must be returned. Default: ``True``.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            an array having the specified data type. The returned array must have
            the same shape as ``x``.

        Examples
        --------
        >>> c = ivy.Container(a=ivy.array([False,True,True]),
        ...                   b=ivy.array([3.14, 2.718, 1.618]))
        >>> ivy.Container.static_astype(c, ivy.int32)
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
        """Copies an array to a specified data type irrespective of
        :ref:`type-promotion` rules.

        .. note::
        Casting floating-point ``NaN`` and ``infinity`` values to integral data types
        is not specified and is implementation-dependent.

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
            specifies whether to copy an array when the specified ``dtype`` matches
            the data type of the input array ``x``. If ``True``, a newly allocated
            array must always be returned. If ``False`` and the specified ``dtype``
            matches the data type of the input array, the input array must be returned;
            otherwise, a newly allocated must be returned. Default: ``True``.
        out
            optional output array, for writing the result to. It must have a shape
             that the inputs broadcast to.

        Returns
        -------
        ret
            an array having the specified data type. The returned array must have
            the same shape as ``x``.

        Examples
        --------
        Using :class:`ivy.Container` instance method:

        >>> x = ivy.Container(a=ivy.array([False,True,True]),
        ...                   b=ivy.array([3.14, 2.718, 1.618]))
        >>> print(x.astype(ivy.int32))
        {
            a: ivy.array([0, 1, 1]),
            b: ivy.array([3, 2, 1])
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
        *arrays: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        `ivy.Container` static method variant of `ivy.broadcast_arrays`.
        This method simply wraps the function,
        and so the docstring for `ivy.broadcast_arrays`
        also applies to this method with minimal changes.

        Parameters
        ----------
        arrays
            an arbitrary number of arrays to-be broadcasted.
            Each array must have the same shape.
            And Each array must have the same dtype as its
            corresponding input array.
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
            A list of containers containing broadcasted arrays

        Examples
        --------
        With :class:`ivy.Container` inputs:

        >>> x1 = ivy.Container(a=ivy.array([1, 2]), b=ivy.array([3, 4]))
        >>> x2 = ivy.Container(a=ivy.array([-1.2, 0.4]), b=ivy.array([0, 1]))
        >>> y = ivy.Container.static_broadcast_arrays(x1, x2)
        >>> print(y)
        [{
            a: ivy.array([1, 2]),
            b: ivy.array([3, 4])
        }, {
            a: ivy.array([-1.2, 0.4]),
            b: ivy.array([0, 1])
        }]

        With mixed :class:`ivy.Container` and :class:`ivy.Array` inputs:

        >>> x1 = ivy.Container(a=ivy.array([4, 5]), b=ivy.array([2, -1]))
        >>> x2 = ivy.array([0.2, 3.])
        >>> y = ivy.Container.static_broadcast_arrays(x1, x2)
        >>> print(y)
        [{
            a: ivy.array([4, 5]),
            b: ivy.array([2, -1])
        }, {
            a: ivy.array([0.2, 3.]),
            b: ivy.array([0.2, 3.])
        }]
        """
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
        *arrays: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        `ivy.Container` instance method variant of `ivy.broadcast_arrays`.
        This method simply wraps the function,
        and so the docstring for `ivy.broadcast_arrays`
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            A container to be broadcatsed against other input arrays.
        arrays
            an arbitrary number of containers having arrays to-be broadcasted.
            Each array must have the same shape.
            Each array must have the same dtype as its corresponding input array.
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


        Examples
        --------
        With :class:`ivy.Container` inputs:

        >>> x1 = ivy.Container(a=ivy.array([1, 2]), b=ivy.array([3, 4]))
        >>> x2 = ivy.Container(a=ivy.array([-1.2, 0.4]), b=ivy.array([0, 1]))
        >>> y = x1.broadcast_arrays(x2)
        >>> print(y)
        [{
            a: ivy.array([1, 2]),
            b: ivy.array([3, 4])
        }, {
            a: ivy.array([-1.2, 0.4]),
            b: ivy.array([0, 1])
        }]

        With mixed :class:`ivy.Container` and :class:`ivy.Array` inputs:

        >>> x1 = ivy.Container(a=ivy.array([4, 5]), b=ivy.array([2, -1]))
        >>> x2 = ivy.zeros(2)
        >>> y = x1.broadcast_arrays(x2)
        >>> print(y)
        [{
            a: ivy.array([4, 5]),
            b: ivy.array([2, -1])
        }, {
            a: ivy.array([0., 0.]),
            b: ivy.array([0., 0.])
        }]
        """
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
        """
        `ivy.Container` static method variant of `ivy.broadcast_to`.
        This method simply wraps the function, and so the docstring
        for `ivy.broadcast_to` also applies to this
        method with minimal changes.

        Parameters
        ----------
        x
            input array to be broadcasted.
        shape
            desired shape to be broadcasted to.
        out
            Optional array to store the broadcasted array.

        Returns
        -------
        ret
            Returns the broadcasted array of shape 'shape'

        Examples
        --------
        With :class:`ivy.Container` static method:

        >>> x = ivy.Container(a=ivy.array([1]),
        ...                   b=ivy.array([2]))
        >>> y = ivy.Container.static_broadcast_to(x,(3, 1))
        >>> print(y)
        {
            a: ivy.array([1],
                         [1],
                         [1]),
            b: ivy.array([2],
                         [2],
                         [2])
        }
        """
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
        """
        `ivy.Container` instance method variant of `ivy.broadcast_to`.
        This method simply wraps the function, and so the docstring
        for `ivy.broadcast_to` also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            input array to be broadcasted.
        shape
            desired shape to be broadcasted to.
        out
            Optional array to store the broadcasted array.

        Returns
        -------
        ret
            Returns the broadcasted array of shape 'shape'

        Examples
        --------
        With :class:`ivy.Container` instance method:

        >>> x = ivy.Container(a=ivy.array([0, 0.5]),
        ...                   b=ivy.array([4, 5]))
        >>> y = x.broadcast_to((3,2))
        >>> print(y)
        {
            a: ivy.array([[0., 0.5],
                          [0., 0.5],
                          [0., 0.5]]),
            b: ivy.array([[4, 5],
                          [4, 5],
                          [4, 5]])
        }

        """
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
            ``True`` if the cast can occur according to :ref:`type-promotion` rules;
            otherwise, ``False``.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0., 1., 2.]),
        ...                   b=ivy.array([3, 4, 5]))
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
            ``True`` if the cast can occur according to :ref:`type-promotion` rules;
            otherwise, ``False``.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0., 1., 2.]),
        ...                   b=ivy.array([3, 4, 5]))
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
        as_native: bool = False,
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
        as_native: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1, 2, 3]), b=ivy.array([2, 3, 4]))
        >>> y = x.dtype()
        >>> print(y)
        {
            a: int32,
            b: int32
        }
        """
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
        *,
        input: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
        float_dtype: Optional[Union[ivy.FloatDtype, ivy.NativeDtype]] = None,
        as_native: Optional[bool] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "default_float_dtype",
            input=input,
            float_dtype=float_dtype,
            as_native=as_native,
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
        /,
        *,
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
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        return self.static_finfo(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_iinfo(
        type: ivy.Container,
        /,
        *,
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
        """
        Examples
        --------
        >>> c = ivy.Container(x=ivy.array([-9,1800,89], dtype=ivy.int16),
        ...                   y=ivy.array([76,-81,16], dtype=ivy.int32))
        >>> c.iinfo()
        {
            x: iinfo(min=-32768, max=32767, dtype=int16),
            y: iinfo(min=-2147483648, max=2147483647, dtype=int32)
        }
        """
        return self.static_iinfo(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_is_bool_dtype(
        dtype_in: ivy.Container,
        /,
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
        /,
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
        /,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        `ivy.Container` static method variant of `is_float_dtype`. This method
        simply wraps this function, so the docstring of `is_float_dtype`
        roughly applies to this method.

        Parameters
        ----------
        dtype_in : ivy.Container
            The input to check for float dtype.

        key_chains : Optional[Union[List[str], Dict[str, str]]]
            The key chains to use when mapping over the input.

        to_apply : bool
            Whether to apply the mapping over the input.

        prune_unapplied : bool
            Whether to prune the keys that were not applied.

        map_sequences : bool
            Boolean indicating whether to map method
            to sequences (list, tuple). Default is ``False``.

        Returns
        -------
        ret : bool
            Boolean indicating whether the input has float dtype.

        Examples
        --------
        >>> x = ivy.static_is_float_dtype(ivy.float32)
        >>> print(x)
        True

        >>> x = ivy.static_is_float_dtype(ivy.int64)
        >>> print(x)
        False

        >>> x = ivy.static_is_float_dtype(ivy.int32)
        >>> print(x)
        False

        >>> x = ivy.static_is_float_dtype(ivy.bool)
        >>> print(x)
        False

        >>> arr = ivy.array([1.2, 3.2, 4.3], dtype=ivy.float32)
        >>> print(arr.is_float_dtype())
        True

        >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([3, 4, 5]))
        >>> print(x.a.dtype, x.b.dtype)
        float32 int32
        """
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
        /,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        `ivy.Container` instance method variant of `ivy.is_float_dtype`.
        This method simply wraps the function,
        and so the docstring for `ivy.is_float_dtype`
        also applies to this method with minimal changes.

        Parameters
        ----------
        self : ivy.Container
            The `ivy.Container` instance to call `ivy.is_float_dtype` on.

        key_chains : Union[List[str], Dict[str, str]]
            The key-chains to apply or not apply the method to.
            Default is ``None``.

        to_apply : bool
            Boolean indicating whether to apply the
            method to the key-chains. Default is ``False``.

        prune_unapplied : bool
            Boolean indicating whether to prune the
            key-chains that were not applied. Default is ``False``.

        map_sequences : bool
            Boolean indicating whether to map method
            to sequences (list, tuple). Default is ``False``.

        Returns
        -------
        ret : bool
            Boolean of whether the input is of a float dtype.

        Examples
        --------
        >>> x = ivy.is_float_dtype(ivy.float32)
        >>> print(x)
        True

        >>> x = ivy.is_float_dtype(ivy.int64)
        >>> print(x)
        False

        >>> x = ivy.is_float_dtype(ivy.int32)
        >>> print(x)
        False

        >>> x = ivy.is_float_dtype(ivy.bool)
        >>> print(x)
        False

        >>> arr = ivy.array([1.2, 3.2, 4.3], dtype=ivy.float32)
        >>> print(arr.is_float_dtype())
        True

        >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([3, 4, 5]))
        >>> print(x.a.dtype, x.b.dtype)
        float32 int32
        """
        return self.static_is_float_dtype(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_is_int_dtype(
        dtype_in: ivy.Container,
        /,
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
        /,
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
        """
        `ivy.Container` static method variant of `ivy.result_type`. This method simply
        wraps the function, and so the docstring for `ivy.result_type` also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input container from which to cast.
        arrays_and_dtypes
            an arbitrary number of input arrays and/or dtypes.
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
            the dtype resulting from an operation involving the input arrays and dtypes.

        Examples
        --------
        >>> x = ivy.Container(a = ivy.array([0, 1, 2]),
        ...                   b = ivy.array([3., 4., 5.]))
        >>> print(x.a.dtype, x.b.dtype)
        int32 float32

        >>> print(ivy.Container.static_result_type(x, ivy.float64))
        {
            a: float64,
            b: float32
        }
        """
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
        """
        `ivy.Container` instance method variant of `ivy.result_type`. This method simply
        wraps the function, and so the docstring for `ivy.result_type` also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input container from which to cast.
        arrays_and_dtypes
            an arbitrary number of input arrays and/or dtypes.
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
            the dtype resulting from an operation involving the input arrays and dtypes.

        Examples
        --------
        >>> x = ivy.Container(a = ivy.array([3, 3, 3]))
        >>> print(x.a.dtype)
        int32

        >>> y = ivy.Container(b = ivy.float64)
        >>> print(x.result_type(y))
        {
            a: {
                b: float64
            }
        }
        """
        return self.static_result_type(
            self,
            *arrays_and_dtypes,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )
