# global
from typing import Optional, Union, List, Dict, Tuple
from numbers import Number

# local
import ivy
from ivy.data_classes.container.base import ContainerBase


class _ContainerWithElementWiseExperimental(ContainerBase):
    @staticmethod
    def static_sinc(
        x: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.sinc. This method simply wraps the
        function, and so the docstring for ivy.sinc also applies to this method with
        minimal changes.

        Parameters
        ----------
        x
            input container whose elements are each expressed in radians.
            Should have a floating-point data type.
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
            a container containing the sinc of each element in ``x``. The returned
            container must have a floating-point data type determined by
            :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0.5, 1.5, 2.5]),
        ...                   b=ivy.array([3.5, 4.5, 5.5]))
        >>> y = ivy.Container.static_sinc(x)
        >>> print(y)
        {
            a: ivy.array([0.636, -0.212, 0.127]),
            b: ivy.array([-0.090, 0.070, -0.057])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "sinc",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def sinc(
        self: ivy.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.sinc. This method simply wraps the
        function, and so the docstring for ivy.sinc also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            input container whose elements are each expressed in radians.
            Should have a floating-point data type.
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
            a container containing the sinc of each element in ``self``.
            The returned container must have a floating-point data type
            determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0.5, 1.5, 2.5]),
        ...                   b=ivy.array([3.5, 4.5, 5.5]))
        >>> y = x.sinc()
        >>> print(y)
        {
            a: ivy.array([0.637,-0.212,0.127]),
            b: ivy.array([-0.0909,0.0707,-0.0579])
        }
        """
        return self.static_sinc(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_fmod(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.fmod. This method simply wraps the
        function, and so the docstring for ivy.fmod also applies to this method with
        minimal changes.

        Parameters
        ----------
        x1
            container with the first input arrays.
        x2
            container with the second input arrays
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with element-wise remainder of divisions.

        Examples
        --------
        >>> x1 = ivy.Container(a=ivy.array([2, 3, 4]),\
                               b=ivy.array([ivy.nan, 0, ivy.nan]))
        >>> x2 = ivy.Container(a=ivy.array([1, 5, 2]),\
                               b=ivy.array([0, ivy.nan, ivy.nan]))
        >>> ivy.Container.static_fmod(x1, x2)
        {
            a: ivy.array([ 0,  3,  0])
            b: ivy.array([ nan,  nan,  nan])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "fmod",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def fmod(
        self: ivy.Container,
        x2: ivy.Container,
        /,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.fmod. This method simply wraps the
        function, and so the docstring for ivy.fmod also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            container with the first input arrays.
        x2
            container with the second input arrays
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with element-wise remainder of divisions.

        Examples
        --------
        >>> x1 = ivy.Container(a=ivy.array([2, 3, 4]),\
                               b=ivy.array([ivy.nan, 0, ivy.nan]))
        >>> x2 = ivy.Container(a=ivy.array([1, 5, 2]),\
                               b=ivy.array([0, ivy.nan, ivy.nan]))
        >>> x1.fmod(x2)
        {
            a: ivy.array([ 0,  3,  0])
            b: ivy.array([ nan,  nan,  nan])
        }
        """
        return self.static_fmod(self, x2, out=out)

    @staticmethod
    def static_fmax(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.fmax. This method simply wraps the
        function, and so the docstring for ivy.fmax also applies to this method with
        minimal changes.

        Parameters
        ----------
        x1
            container with the first input arrays.
        x2
            container with the second input arrays
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with element-wise maximums.

        Examples
        --------
        >>> x1 = ivy.Container(a=ivy.array([2, 3, 4]),\
                               b=ivy.array([ivy.nan, 0, ivy.nan]))
        >>> x2 = ivy.Container(a=ivy.array([1, 5, 2]),\
                               b=ivy.array([0, ivy.nan, ivy.nan]))
        >>> ivy.Container.static_fmax(x1, x2)
        {
            a: ivy.array([ 2.,  5.,  4.])
            b: ivy.array([ 0,  0,  nan])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "fmax",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def fmax(
        self: ivy.Container,
        x2: ivy.Container,
        /,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.fmax. This method simply wraps the
        function, and so the docstring for ivy.fmax also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            container with the first input arrays.
        x2
            container with the second input arrays
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with element-wise maximums.

        Examples
        --------
        >>> x1 = ivy.Container(a=ivy.array([2, 3, 4]),\
                               b=ivy.array([ivy.nan, 0, ivy.nan]))
        >>> x2 = ivy.Container(a=ivy.array([1, 5, 2]),\
                               b=ivy.array([0, ivy.nan, ivy.nan]))
        >>> x1.fmax(x2)
        {
            a: ivy.array([ 2.,  5.,  4.])
            b: ivy.array([ 0,  0,  nan])
        }
        """
        return self.static_fmax(self, x2, out=out)

    @staticmethod
    def static_float_power(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container, float, list, tuple],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container, float, list, tuple],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.float_power. This method simply wraps
        the function, and so the docstring for ivy.float_power also applies to this
        method with minimal changes.

        Parameters
        ----------
        x1
            container with the base input arrays.
        x2
            container with the exponent input arrays
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with base arrays raised to the powers
            of exponents arrays, element-wise .

        Examples
        --------
        >>> x1 = ivy.Container(a=ivy.array([1, 2, 3]),\
                               b=ivy.array([2, 10]))
        >>> x2 = ivy.Container(a=ivy.array([1, 3, 1]), b=0)
        >>> ivy.Container.static_float_power(x1, x2)
        {
            a: ivy.array([1,  8,  3])
            b: ivy.array([1, 1])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "float_power",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def float_power(
        self: ivy.Container,
        x2: ivy.Container,
        /,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.float_power. This method simply
        wraps the function, and so the docstring for ivy.float_power also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            container with the base input arrays.
        x2
            container with the exponent input arrays
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with base arrays raised to the powers
            of exponents arrays, element-wise .

        Examples
        --------
        >>> x1 = ivy.Container(a=ivy.array([1, 2, 3]),\
                               b=ivy.array([2, 10]))
        >>> x2 = ivy.Container(a=ivy.array([1, 3, 1]), b=0)
        >>> x1.float_power(x2)
        {
            a: ivy.array([1,  8,  3])
            b: ivy.array([1, 1])
        }
        """
        return self.static_float_power(self, x2, out=out)

    @staticmethod
    def static_copysign(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container, Number],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container, Number],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.copysign. This method simply wraps
        the function, and so the docstring for ivy.copysign also applies to this method
        with minimal changes.

        Parameters
        ----------
        x1
            Container, Array, or scalar to change the sign of
        x2
            Container, Array, or scalar from which the new signs are applied
            Unsigned zeroes are considered positive.
        out
            optional output Container, for writing the result to.

        Returns
        -------
        ret
            x1 with the signs of x2.
            This is a scalar if both x1 and x2 are scalars.

        Examples
        --------
        >>> x1 = ivy.Container(a=ivy.array([0,1,2]), b=ivy.array(-1))
        >>> x2 = ivy.Container(a=-1, b=ivy.array(10))
        >>> ivy.Container.static_copysign(x1, x2)
        {
            a: ivy.array([-0., -1., -2.]),
            b: ivy.array(1.)
        }
        >>> ivy.Container.static_copysign(23, x1)
        {
            a: ivy.array([23., 23., 23.]),
            b: ivy.array(-23.)
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "copysign",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def copysign(
        self: ivy.Container,
        x2: ivy.Container,
        /,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.copysign. This method simply wraps
        the function, and so the docstring for ivy.copysign also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            Container to change the sign of
        x2
            Container from which the new signs are applied
            Unsigned zeroes are considered positive.
        out
            optional output Container, for writing the result to.

        Returns
        -------
        ret
            x1 with the signs of x2.
            This is a scalar if both x1 and x2 are scalars.

        Examples
        --------
        >>> x1 = ivy.Container(a=ivy.array([0,1,2]), b=ivy.array(-1))
        >>> x2 = ivy.Container(a=-1, b=ivy.array(10))
        >>> x1.copysign(x2)
        {
            a: ivy.array([-0., -1., -2.]),
            b: ivy.array(1.)
        }
        >>> x1.copysign(-1)
        {
            a: ivy.array([-0., -1., -2.]),
            b: ivy.array(-1.)
        }
        """
        return self.static_copysign(self, x2, out=out)

    @staticmethod
    def static_count_nonzero(
        a: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        axis: Optional[Union[int, Tuple[int, ...], ivy.Container]] = None,
        keepdims: Union[bool, ivy.Container] = False,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype, ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.count_nonzero. This method simply
        wraps the function, and so the docstring for ivy.count_nonzero also applies to
        this method with minimal changes.

        Parameters
        ----------
        a
            container with the base input arrays.
        axis
            optional axis or tuple of axes along which to count non-zeros. Default is
            None, meaning that non-zeros will be counted along a flattened
            version of the input array.
        keepdims
            optional, if this is set to True, the axes that are counted are left in the
            result as dimensions with size one. With this option, the result
            will broadcast correctly against the input array.
        dtype
            optional output dtype. Default is of type integer.
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
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including number of non-zero values in the array along a
            given axis. Otherwise, container with the total number of non-zero
            values in the array is returned.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([[0, 1, 2, 3],[4, 5, 6, 7]]),\
                        b=ivy.array([[[0,1],[2,3]],[[4,5],[6,7]]]))
        >>> ivy.Container.static_count_nonzero(x)
        {
            a: ivy.array(7),
            b: ivy.array(7)
        }
        >>> x = ivy.Container(a=ivy.array([[0, 1, 2, 3],[4, 5, 6, 7]]),\
                        b=ivy.array([[[0,1],[2,3]],[[4,5],[6,7]]]))
        >>> ivy.Container.static_count_nonzero(x, axis=0)
        {
            a: ivy.array([1, 2, 2, 2]),
            b: ivy.array([[1, 2],
                          [2, 2]])
        }
        >>> x = ivy.Container(a=ivy.array([[0, 1, 2, 3],[4, 5, 6, 7]]),\
                        b=ivy.array([[[0,1],[2,3]],[[4,5],[6,7]]]))
        >>> ivy.Container.static_count_nonzero(x, axis=(0,1), keepdims=True)
        {
            a: ivy.array([[7]]),
            b: ivy.array([[[3, 4]]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "count_nonzero",
            a,
            axis=axis,
            keepdims=keepdims,
            dtype=dtype,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def count_nonzero(
        self: ivy.Container,
        /,
        *,
        axis: Optional[Union[int, Tuple[int, ...], ivy.Container]] = None,
        keepdims: Union[bool, ivy.Container] = False,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype, ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.count_nonzero. This method simply
        wraps the function, and so the docstring for ivy.count_nonzero also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            container with the base input arrays.
        axis
            optional axis or tuple of axes along which to count non-zeros. Default is
            None, meaning that non-zeros will be counted along a flattened
            version of the input array.
        keepdims
            optional, if this is set to True, the axes that are counted are left in the
            result as dimensions with size one. With this option, the result
            will broadcast correctly against the input array.
        dtype
            optional output dtype. Default is of type integer.
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
            Default is ``False``
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including number of non-zero values in the array along a
            given axis. Otherwise, container with the total number of non-zero
            values in the array is returned.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([[0, 1, 2, 3],[4, 5, 6, 7]]),\
                        b=ivy.array([[[0,1],[2,3]],[[4,5],[6,7]]]))
        >>> x.count_nonzero()
        {
            a: ivy.array(7),
            b: ivy.array(7)
        }
        >>> x = ivy.Container(a=ivy.array([[0, 1, 2, 3],[4, 5, 6, 7]]),\
                        b=ivy.array([[[0,1],[2,3]],[[4,5],[6,7]]]))
        >>> x.count_nonzero(axis=0)
        {
            a: ivy.array([1, 2, 2, 2]),
            b: ivy.array([[1, 2],
                          [2, 2]])
        }
        >>> x = ivy.Container(a=ivy.array([[0, 1, 2, 3],[4, 5, 6, 7]]),\
                        b=ivy.array([[[0,1],[2,3]],[[4,5],[6,7]]]))
        >>> x.count_nonzero(axis=(0,1), keepdims=True)
        {
            a: ivy.array([[7]]),
            b: ivy.array([[[3, 4]]])
        }
        """
        return self.static_count_nonzero(
            self,
            axis=axis,
            keepdims=keepdims,
            dtype=dtype,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_nansum(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        axis: Optional[Union[tuple, int, ivy.Container]] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype, ivy.Container]] = None,
        keepdims: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.nansum. This method simply wraps the
        function, and so the docstring for ivy.nansum also applies to this method with
        minimal changes.

        Parameters
        ----------
        x
            Input array.
        axis
            Axis or axes along which the sum is computed.
            The default is to compute the sum of the flattened array.
        dtype
            The type of the returned array and of the accumulator in
            which the elements are summed. By default, the dtype of input is used.
        keepdims
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one.
        out
            Alternate output array in which to place the result.
            The default is None.

        Returns
        -------
        ret
            A new array holding the result is returned unless out is specified,
            in which it is returned.

        Examples
        --------
        With one :class:`ivy.Container` input:
        >>> x = ivy.Container(a=ivy.array([[10, 7, 4], [3, 2, 1]]),\
                b=ivy.array([[1, 4, 2], [ivy.nan, ivy.nan, 0]]))
        >>> ivy.Container.static_nansum(x)
        {
            a: 27,
            b: 7.0
        }
        >>> ivy.Container.static_nansum(x, axis=0)
        {
            a: ivy.array([13, 9, 5]),
            b: ivy.array([1., 4., 2.])
        }
        >>> ivy.Container.static_nansum(x, axis=1)
        {
            a: ivy.array([21, 6]),
            b: ivy.array([7., 0.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "nansum",
            x,
            axis=axis,
            dtype=dtype,
            keepdims=keepdims,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def nansum(
        self: ivy.Container,
        /,
        *,
        axis: Optional[Union[tuple, int, ivy.Container]] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype, ivy.Container]] = None,
        keepdims: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.nansum. This method simply wraps
        the function, and so the docstring for ivy.nansum also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            Input container including arrays.
        axis
            Axis or axes along which the sum is computed.
            The default is to compute the sum of the flattened array.
        dtype
            The type of the returned array and of the accumulator in
            which the elements are summed. By default, the dtype of input is used.
        keepdims
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one.
        out
            Alternate output array in which to place the result.
            The default is None.

        Returns
        -------
        ret
            A new array holding the result is returned unless out is specified,
            in which it is returned.

        Examples
        --------
        With one :class:`ivy.Container` input:
        >>> x = ivy.Container(a=ivy.array([[10, 7, 4], [3, 2, 1]]),\
                b=ivy.array([[1, 4, 2], [ivy.nan, ivy.nan, 0]]))
        >>> x.nansum(axis=0)
        {
            a: ivy.array([13, 9, 5]),
            b: ivy.array([1., 4., 2.])
        }
        >>> x.nansum(axis=1)
        {
            a: ivy.array([21, 6]),
            b: ivy.array([7., 0.])
        }
        """
        return self.static_nansum(
            self, axis=axis, dtype=dtype, keepdims=keepdims, out=out
        )

    @staticmethod
    def static_isclose(
        a: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        b: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        rtol: Union[float, ivy.Container] = 1e-05,
        atol: Union[float, ivy.Container] = 1e-08,
        equal_nan: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.isclose. This method simply wraps the
        function, and so the docstring for ivy.isclose also applies to this method with
        minimal changes.

        Parameters
        ----------
        a
            Input container containing first input array.
        b
            Input container containing second input array.
        rtol
            The relative tolerance parameter.
        atol
            The absolute tolerance parameter.
        equal_nan
            Whether to compare NaN's as equal. If True, NaN's in a will be
            considered equal to NaN's in b in the output array.
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
            Alternate output array in which to place the result.
            The default is None.

        Returns
        -------
        ret
            A new array holding the result is returned unless out is specified,
            in which it is returned.

        Examples
        --------
        With one :class:`ivy.Container` input:
        >>> x = ivy.Container(a=ivy.array([1.0, ivy.nan]),\
                b=ivy.array([1.0, ivy.nan]))
        >>> y = ivy.Container(a=ivy.array([1.0, ivy.nan]),\
                b=ivy.array([1.0, ivy.nan]))
        >>> ivy.Container.static_isclose(x, y)
        {
            a: ivy.array([True, False]),
            b: ivy.array([True, False])
        }
        >>> ivy.Container.static_isclose(x, y, equal_nan=True)
        {
            a: ivy.array([True, True]),
            b: ivy.array([True, True])
        }
        >>> x = ivy.Container(a=ivy.array([1.0, 2.0]),\
                b=ivy.array([1.0, 2.0]))
        >>> y = ivy.Container(a=ivy.array([1.0, 2.001]),\
                b=ivy.array([1.0, 2.0]))
        >>> ivy.Container.static_isclose(x, y, atol=0.0)
        {
            a: ivy.array([True, False]),
            b: ivy.array([True, True])
        }
        >>> ivy.Container.static_isclose(x, y, rtol=0.01, atol=0.0)
        {
            a: ivy.array([True, True]),
            b: ivy.array([True, True])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "isclose",
            a,
            b,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def isclose(
        self: ivy.Container,
        b: ivy.Container,
        /,
        *,
        rtol: Union[float, ivy.Container] = 1e-05,
        atol: Union[float, ivy.Container] = 1e-08,
        equal_nan: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.isclose. This method simply wraps
        the function, and so the docstring for ivy.isclose also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            Input container containing first input array.
        b
            Input container containing second input array.
        rtol
            The relative tolerance parameter.
        atol
            The absolute tolerance parameter.
        equal_nan
            Whether to compare NaN's as equal. If True, NaN's in a will be
            considered equal to NaN's in b in the output array.
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
            Alternate output array in which to place the result.
            The default is None.

        Returns
        -------
        ret
            A new array holding the result is returned unless out is specified,
            in which it is returned.

        Examples
        --------
        With one :class:`ivy.Container` input:
        >>> x = ivy.Container(a=ivy.array([1.0, ivy.nan]),\
                b=ivy.array([1.0, ivy.nan]))
        >>> y = ivy.Container(a=ivy.array([1.0, ivy.nan]),\
                b=ivy.array([1.0, ivy.nan]))
        >>> x.isclose(y)
        {
            a: ivy.array([True, False]),
            b: ivy.array([True, False])
        }
        >>> x.isclose(y, equal_nan=True)
        {
            a: ivy.array([True, True]),
            b: ivy.array([True, True])
        }
        >>> x = ivy.Container(a=ivy.array([1.0, 2.0]),\
                b=ivy.array([1.0, 2.0]))
        >>> y = ivy.Container(a=ivy.array([1.0, 2.001]),\
                b=ivy.array([1.0, 2.0]))
        >>> x.isclose(y, atol=0.0)
        {
            a: ivy.array([True, False]),
            b: ivy.array([True, True])
        }
        >>> x.isclose(y, rtol=0.01, atol=0.0)
        {
            a: ivy.array([True, True]),
            b: ivy.array([True, True])
        }
        """
        return self.static_isclose(
            self,
            b,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_signbit(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container, float, int, list, tuple],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.signbit. This method simply wraps the
        function, and so the docstring for ivy.signbit also applies to this method with
        minimal changes.

        Parameters
        ----------
        x
            input container with array-like items.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with element-wise signbit of input arrays.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1, -2, 3]),\
                               b=-5)
        >>> ivy.Container.static_signbit(x)
        {
            a: ivy.array([False, True, False])
            b: ivy.array([True])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "signbit",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def signbit(
        self: ivy.Container,
        /,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.signbit. This method simply wraps
        the function, and so the docstring for ivy.signbit also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input container with array-like items.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with element-wise signbit of input arrays.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1, -2, 3]),\
                               b=-5)
        >>> x.signbit()
        {
            a: ivy.array([False, True, False])
            b: ivy.array([True])
        }
        """
        return self.static_signbit(self, out=out)

    @staticmethod
    def static_hypot(
        x1: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.hypot. This method simply wraps the
        function, and so the docstring for ivy.hypot also applies to this method with
        minimal changes.

        Parameters
        ----------
        x1
            Input container containing first input array.
        x2
            Input container containing second input array.
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
            Alternate output array in which to place the result.
            The default is None.

        Returns
        -------
        ret
            container including the hypot function computed element-wise

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([2.0]),\
        ...                         b=ivy.array([3.0]))
        >>> y = ivy.Container(a=ivy.array([3.0]),\
                                    b=ivy.array([4.0]))
        >>> ivy.Container.static_hypot(x, y)
        {
            a: ivy.array([3.6055]),
            b: ivy.array([5.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "hypot",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def hypot(
        self: ivy.Container,
        x2: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.hypot. This method simply wraps the
        function, and so the docstring for ivy.hypot also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            Input container containing first input array.
        x2
            Input container containing second input array.
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
            Alternate output array in which to place the result.
            The default is None.

        Returns
        -------
        ret
            container including the hypot function computed element-wise

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([2.0]),\
        ...                         b=ivy.array([3.0]))
        >>> y = ivy.Container(a=ivy.array([3.0]),\
                                    b=ivy.array([4.0]))
        >>> x.hypot(y)
        {
            a: ivy.array([3.6055]),
            b: ivy.array([5.])
        }
        """
        return self.static_hypot(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_allclose(
        x1: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        rtol: Union[float, ivy.Container] = 1e-05,
        atol: Union[float, ivy.Container] = 1e-08,
        equal_nan: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.allclose. This method simply wraps
        the function, and so the docstring for ivy.allclose also applies to this method
        with minimal changes.

        Parameters
        ----------
        x1
            Input container containing first input array.
        x2
            Input container containing second input array.
        rtol
            The relative tolerance parameter.
        atol
            The absolute tolerance parameter.
        equal_nan
            Whether to compare NaN's as equal. If True, NaN's in x1 will be
            considered equal to NaN's in x2 in the output array.
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
            Alternate output array in which to place the result.
            The default is None.

        Returns
        -------
        ret
            A new container holding the result is returned unless out is specified,
            in which it is returned.

        Examples
        --------
        >>> x1 = ivy.Container(a=ivy.array([1., 2., 3.]),\
        ...                         b=ivy.array([1., 2., 3.]))
        >>> x2 = ivy.Container(a=ivy.array([1., 2., 3.]),\
        ...                         b=ivy.array([1., 2., 3.]))
        >>> y = ivy.Container.static_allclose(x1, x2)
        >>> print(y)
        {
            a: ivy.array(True),
            b: ivy.array(True)
        }

        >>> x1 = ivy.Container(a=ivy.array([1., 2., 3.]),\
        ...                         b=ivy.array([1., 2., 3.]))
        >>> x2 = ivy.Container(a=ivy.array([1., 2., 3.0003]),\
        ...                         b=ivy.array([1.0006, 2., 3.]))
        >>> y = ivy.Container.static_allclose(x1, x2, rtol=1e-3)
        >>> print(y)
        {
            a: ivy.array(True),
            b: ivy.array(True)
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "allclose",
            x1,
            x2,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def allclose(
        self: ivy.Container,
        x2: ivy.Container,
        /,
        *,
        rtol: Union[float, ivy.Container] = 1e-05,
        atol: Union[float, ivy.Container] = 1e-08,
        equal_nan: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.allclose. This method simply wraps
        the function, and so the docstring for ivy.allclose also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            Input container containing first input array.
        x2
            Input container containing second input array.
        rtol
            The relative tolerance parameter.
        atol
            The absolute tolerance parameter.
        equal_nan
            Whether to compare NaN's as equal. If True, NaN's in x1 will be
            considered equal to NaN's in x2 in the output array.
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
            Alternate output array in which to place the result.
            The default is None.

        Returns
        -------
        ret
            A new container holding the result is returned unless out is specified,
            in which it is returned.

        Examples
        --------
        >>> x1 = ivy.Container(a=ivy.array([1., 2., 3.]), b=ivy.array([1., 2., 3.]))
        >>> x2 = ivy.Container(a=ivy.array([1., 2., 3.]), b=ivy.array([1., 2., 3.]))
        >>> y = x1.allclose(x2)
        >>> print(y)
        {
            a: ivy.array(True),
            b: ivy.array(True)
        }

        >>> x1 = ivy.Container(a=ivy.array([1., 2., 3.]),\
        ...                         b=ivy.array([1., 2., 3.]))
        >>> x2 = ivy.Container(a=ivy.array([1., 2., 3.0003]),\
        ...                         b=ivy.array([1.0006, 2., 3.]))
        >>> y = x1.allclose(x2, rtol=1e-3)
        >>> print(y)
        {
            a: ivy.array(True),
            b: ivy.array(True)
        }
        """
        return self.static_allclose(
            self,
            x2,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_diff(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        n: Union[int, ivy.Container] = 1,
        axis: Union[int, ivy.Container] = -1,
        prepend: Optional[
            Union[ivy.Array, ivy.NativeArray, int, list, tuple, ivy.Container]
        ] = None,
        append: Optional[
            Union[ivy.Array, ivy.NativeArray, int, list, tuple, ivy.Container]
        ] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.diff. This method simply wraps the
        function, and so the docstring for ivy.diff also applies to this method with
        minimal changes.

        Parameters
        ----------
        x
            input container with array-like items.
        n
            The number of times values are differenced. If zero, the input is returned
            as-is.
        axis
            The axis along which the difference is taken, default is the last axis.
        prepend,append
            Values to prepend/append to x along given axis prior to performing the
            difference. Scalar values are expanded to arrays with length 1 in the
            direction of axis and the shape of the input array in along all other
            axes. Otherwise the dimension and shape must match x except along axis.
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
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with the n-th discrete difference along
            the given axis.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1, 2, 4, 7, 0]),
                              b=ivy.array([1, 2, 4, 7, 0]))
        >>> ivy.Container.static_diff(x)
        {
            a: ivy.array([ 1,  2,  3, -7]),
            b: ivy.array([ 1,  2,  3, -7])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "diff",
            x,
            n=n,
            axis=axis,
            prepend=prepend,
            append=append,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def diff(
        self: ivy.Container,
        /,
        *,
        n: Union[int, ivy.Container] = 1,
        axis: Union[int, ivy.Container] = -1,
        prepend: Optional[
            Union[ivy.Array, ivy.NativeArray, int, list, tuple, ivy.Container]
        ] = None,
        append: Optional[
            Union[ivy.Array, ivy.NativeArray, int, list, tuple, ivy.Container]
        ] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.diff. This method simply wraps the
        function, and so the docstring for ivy.diff also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            input container with array-like items.
        n
            The number of times values are differenced. If zero, the input is returned
            as-is.
        axis
            The axis along which the difference is taken, default is the last axis.
        prepend,append
            Values to prepend/append to x along given axis prior to performing the
            difference. Scalar values are expanded to arrays with length 1 in the
            direction of axis and the shape of the input array in along all other
            axes. Otherwise the dimension and shape must match x except along axis.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with the n-th discrete difference along the
            given axis.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1, 2, 4, 7, 0]),
                              b=ivy.array([1, 2, 4, 7, 0]))
        >>> x.diff()
        {
            a: ivy.array([1, 2, 3, -7]),
            b: ivy.array([1, 2, 3, -7])
        }
        """
        return self.static_diff(
            self, n=n, axis=axis, prepend=prepend, append=append, out=out
        )

    @staticmethod
    def static_fix(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.fix. This method simply wraps the
        function, and so the docstring for ivy.fix also applies to this method with
        minimal changes.

        Parameters
        ----------
        x
            input container with array items.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with element-wise rounding of
            input arrays elements.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([2.1, 2.9, -2.1]),\
                               b=ivy.array([3.14]))
        >>> ivy.Container.static_fix(x)
        {
            a: ivy.array([ 2.,  2., -2.])
            b: ivy.array([ 3.0 ])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "fix",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def fix(
        self: ivy.Container,
        /,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.fix. This method simply wraps the
        function, and so the docstring for ivy.fix also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            input container with array items.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with element-wise rounding of
            input arrays elements.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([2.1, 2.9, -2.1]),\
                               b=ivy.array([3.14]))
        >>> x.fix()
        {
            a: ivy.array([ 2.,  2., -2.])
            b: ivy.array([ 3.0 ])
        }
        """
        return self.static_fix(self, out=out)

    @staticmethod
    def static_nextafter(
        x1: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.nextafter. This method simply wraps
        the function, and so the docstring for ivy.nextafter also applies to this method
        with minimal changes.

        Parameters
        ----------
        x1
            Input container containing first input arrays.
        x2
            Input container containing second input arrays.
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
            Alternate output array in which to place the result.
            The default is None.

        Returns
        -------
        ret
            container including the next representable values of
            input container's arrays, element-wise

        Examples
        --------
        >>> x1 = ivy.Container(a=ivy.array([1.0e-50, 2.0e+50]),\
        ...                         b=ivy.array([2.0, 1.0])
        >>> x2 = ivy.Container(a=ivy.array([5.5e-30]),\
        ...                         b=ivy.array([-2.0]))
        >>> ivy.Container.static_nextafter(x1, x2)
        {
            a: ivy.array([1.4013e-45., 3.4028e+38]),
            b: ivy.array([5.5e-30])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "nextafter",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def nextafter(
        self: ivy.Container,
        x2: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.nextafter. This method simply wraps
        the function, and so the docstring for ivy.nextafter also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            Input container containing first input array.
        x2
            Input container containing second input array.
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
            Alternate output array in which to place the result.
            The default is None.

        Returns
        -------
        ret
            container including the next representable values of
            input container's arrays, element-wise

        Examples
        --------
        >>> x1 = ivy.Container(a=ivy.array([1.0e-50, 2.0e+50]),\
        ...                         b=ivy.array([2.0, 1.0])
        >>> x2 = ivy.Container(a=ivy.array([5.5e-30]),\
        ...                         b=ivy.array([-2.0]))
        >>> x1.nextafter(x2)
        {
            a: ivy.array([1.4013e-45., 3.4028e+38]),
            b: ivy.array([5.5e-30])
        }
        """
        return self.static_nextafter(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_zeta(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        q: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.zeta. This method simply wraps the
        function, and so the docstring for ivy.zeta also applies to this method with
        minimal changes.

        Parameters
        ----------
        x
            Input container containing first input arrays.
        q
            Input container containing second input arrays.
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
            Alternate output array in which to place the result.
            The default is None.

        Returns
        -------
        ret
            container including the zeta function computed element-wise

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([5.0, 3.0]),\
        ...                         b=ivy.array([2.0, 1.0])
        >>> q = ivy.Container(a=ivy.array([2.0]),\
        ...                         b=ivy.array([5.0]))
        >>> ivy.Container.static_zeta(x1, x2)
        {
            a: ivy.array([0.0369, 0.2021]),
            b: ivy.array([0.0006, 0.0244])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "zeta",
            x,
            q,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def zeta(
        self: ivy.Container,
        q: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.zeta. This method simply wraps the
        function, and so the docstring for ivy.zeta also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            Input container containing first input array.
        q
            Input container containing second input array.
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
            Alternate output array in which to place the result.
            The default is None.

        Returns
        -------
        ret
            container including the zeta function computed element-wise

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([5.0, 3.0]),\
        ...                         b=ivy.array([2.0, 1.0])
        >>> q = ivy.Container(a=ivy.array([2.0]),\
        ...                         b=ivy.array([5.0]))
        >>> x.zeta(q)
        {
            a: ivy.array([0.0369, 0.2021]),
            b: ivy.array([0.0006, 0.0244])
        }
        """
        return self.static_zeta(
            self,
            q,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_gradient(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        spacing: Union[int, list, tuple, ivy.Container] = 1,
        edge_order: Union[int, ivy.Container] = 1,
        axis: Optional[Union[int, list, tuple, ivy.Container]] = None,
    ) -> ivy.Container:
        return ContainerBase.cont_multi_map_in_function(
            "gradient",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            spacing=spacing,
            edge_order=edge_order,
            axis=axis,
        )

    def gradient(
        self: ivy.Container,
        /,
        *,
        spacing: Union[int, list, tuple, ivy.Container] = 1,
        edge_order: Union[int, ivy.Container] = 1,
        axis: Optional[Union[int, list, tuple, ivy.Container]] = None,
    ) -> ivy.Container:
        """
        Calculate gradient of x with respect to (w.r.t.) spacing.

        Parameters
        ----------
            x
                input array representing outcomes of the function
                spacing
                if not given, indices of x will be used
                if scalar indices of x will be scaled with this value
                if array gradient of x w.r.t. spacing
            edge_order
                1 or 2, for 'frist order' and 'second order' estimation
                of boundary values of gradient respectively.
            axis
                dimension(s) to approximate the gradient over.
                By default, partial gradient is computed in every dimension


        Returns
        -------
        ret
            Array with values computed from gradient function from
            inputs

        Examples
        --------
        >>> coordinates = ivy.Container(
        >>>     a=(ivy.array([-2., -1., 1., 4.]),),
        >>>     b=(ivy.array([2., 1., -1., -4.]),)
        >>> )
        >>> values = ivy.Container(
        >>>     a=ivy.array([4., 1., 1., 16.]),
        >>>     b=ivy.array([4., 1., 1., 16.])
        >>> )
        >>> ivy.gradient(values, spacing=coordinates)
        {
            a: ivy.array([-3., -2., 2., 5.]),
            b: ivy.array([3., 2., -2., -5.])
        }

        >>> values = ivy.Container(
        >>>     a=ivy.array([[1, 2, 4, 8], [10, 20, 40, 80]]),
        >>>     b=ivy.array([[-1, -2, -4, -8], [-10, -20, -40, -80]])
        >>> )
        >>> ivy.gradient(values)
        [{
            a: ivy.array([[9., 18., 36., 72.],
                          [9., 18., 36., 72.]]),
            b: ivy.array([[-9., -18., -36., -72.],
                          [-9., -18., -36., -72.]])
        }, {
            a: ivy.array([[1., 1.5, 3., 4.],
                          [10., 15., 30., 40.]]),
            b: ivy.array([[-1., -1.5, -3., -4.],
                          [-10., -15., -30., -40.]])
        }]

        >>> values = ivy.Container(
        >>>     a=ivy.array([[1, 2, 4, 8], [10, 20, 40, 80]]),
        >>>     b=ivy.array([[-1, -2, -4, -8], [-10, -20, -40, -80]])
        >>> )
        >>> ivy.gradient(values, spacing=2.0)
        [{
            a: ivy.array([[4.5, 9., 18., 36.],
                          [4.5, 9., 18., 36.]]),
            b: ivy.array([[-4.5, -9., -18., -36.],
                          [-4.5, -9., -18., -36.]])
        }, {
            a: ivy.array([[0.5, 0.75, 1.5, 2.],
                          [5., 7.5, 15., 20.]]),
            b: ivy.array([[-0.5, -0.75, -1.5, -2.],
                          [-5., -7.5, -15., -20.]])
        }]

        >>> values = ivy.Container(
        >>>     a=ivy.array([[1, 2, 4, 8], [10, 20, 40, 80]]),
        >>>     b=ivy.array([[-1, -2, -4, -8], [-10, -20, -40, -80]])
        >>> )
        >>> ivy.gradient(values, axis=1)
        {
            a: ivy.array([[1., 1.5, 3., 4.],
                          [10., 15., 30., 40.]]),
            b: ivy.array([[-1., -1.5, -3., -4.],
                          [-10., -15., -30., -40.]])
        }

        >>> values = ivy.Container(
        >>>     a=ivy.array([[1, 2, 4, 8], [10, 20, 40, 80]]),
        >>>     b=ivy.array([[-1, -2, -4, -8], [-10, -20, -40, -80]])
        >>> )
        >>> ivy.gradient(values, spacing = [3., 2.])
        [{
            a: ivy.array([[3., 6., 12., 24.],
                          [3., 6., 12., 24.]]),
            b: ivy.array([[-3., -6., -12., -24.],
                          [-3., -6., -12., -24.]])
        }, {
            a: ivy.array([[0.5, 0.75, 1.5, 2.],
                          [5., 7.5, 15., 20.]]),
            b: ivy.array([[-0.5, -0.75, -1.5, -2.],
                          [-5., -7.5, -15., -20.]])
        }]

        >>> coords = ivy.Container(
        >>>    a=(ivy.array([0, 2]), ivy.array([0, 3, 6, 9])),
        >>>    b=(ivy.array([0, -2]), ivy.array([0, -3, -6, -9]))
        >>>)
        >>> values = ivy.Container(
        >>>     a=ivy.array([[1, 2, 4, 8], [10, 20, 40, 80]]),
        >>>     b=ivy.array([[-1, -2, -4, -8], [-10, -20, -40, -80]])
        >>>)
        >>> ivy.gradient(values, spacing = coords)
        [{
            a: ivy.array([[4.5, 9., 18., 36.],
                          [4.5, 9., 18., 36.]]),
            b: ivy.array([[4.5, 9., 18., 36.],
                          [4.5, 9., 18., 36.]])
        }, {
            a: ivy.array([[0.33333333, 0.5, 1., 1.33333333],
                          [3.33333333, 5., 10., 13.33333333]]),
            b: ivy.array([[0.33333333, 0.5, 1., 1.33333333],
                          [3.33333333, 5., 10., 13.33333333]])
        }]
        """
        return self.static_gradient(
            self, spacing=spacing, edge_order=edge_order, axis=axis
        )

    @staticmethod
    def static_xlogy(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        y: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.xlogy. This method simply wraps the
        function, and so the docstring for ivy.xlogy also applies to this method with
        minimal changes.

        Parameters
        ----------
        x
            Input container containing first input arrays.
        y
            Input container containing second input arrays.
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
            Alternate output array in which to place the result.
            The default is None.

        Returns
        -------
        ret
            container including the next representable values of
            input container's arrays, element-wise

        Examples
        --------
        >>> x = ivy.Container(a=ivy.zeros(3)),\
        ...                         b=ivy.array([1.0, 2.0, 3.0]))
        >>> y = ivy.Container(a=ivy.array([-1.0, 0.0, 1.0]),\
        ...                         b=ivy.array([3.0, 2.0, 1.0]))
        >>> ivy.Container.static_xlogy(x, y)
        {
            a: ivy.array([0.0, 0.0, 0.0]),
            b: ivy.array([1.0986, 1.3863, 0.0000])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "xlogy",
            x,
            y,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def xlogy(
        self: ivy.Container,
        y: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.xlogy. This method simply wraps the
        function, and so the docstring for ivy.xlogy also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            Input container containing first input array.
        y
            Input container containing second input array.
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
            Alternate output array in which to place the result.
            The default is None.

        Returns
        -------
        ret
            container including the next representable values of
            input container's arrays, element-wise

        Examples
        --------
        >>> x = ivy.Container(a=ivy.zeros(3)),\
        ...                         b=ivy.array([1.0, 2.0, 3.0]))
        >>> y = ivy.Container(a=ivy.array([-1.0, 0.0, 1.0]),\
        ...                         b=ivy.array([3.0, 2.0, 1.0]))
        >>> x.xlogy(y)
        {
            a: ivy.array([0.0, 0.0, 0.0]),
            b: ivy.array([1.0986, 1.3863, 0.0000])
        }
        """
        return self.static_xlogy(
            self,
            y,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_binarizer(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        threshold: Union[float, ivy.Container] = 0,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        Map the values of the input tensor to either 0 or 1, element-wise, based on the
        outcome of a comparison against a threshold value.

        Parameters
        ----------
        self
            input container. Should have a real-valued floating-point data type.
        threshold
            Values greater than this are
            mapped to 1, others to 0.
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
            Binarized output data
        """
        return ContainerBase.cont_multi_map_in_function(
            "binarizer",
            x,
            threshold=threshold,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def binarizer(
        self: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        *,
        threshold: Union[float, ivy.Container] = 0,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        Map the values of the input tensor to either 0 or 1, element-wise, based on the
        outcome of a comparison against a threshold value.

        Parameters
        ----------
        threshold
            Values greater than this are
            mapped to 1, others to 0.
        key_chains
            The keychains to apply or not apply the method to. Default is ``None``.
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
            Binarized output data
        """
        return self.static_binarizer(
            self,
            threshold=threshold,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_conj(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.conj. This method simply wraps the
        function, and so the docstring for ivy.conj also applies to this method with
        minimal changes.

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
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing output array(s) of the same
            dtype as the input array(s) with the complex conjugates of
            the complex values present in the input array. If x is a
            container of scalar(s) then a container of scalar(s)
            will be returned.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([-1+5j, 0-0j, 1.23j]),
        ...                   b=ivy.array([7.9, 0.31+3.3j, -4.2-5.9j]))
        >>> z = ivy.Container.static_conj(x)
        >>> print(z)
        {
            a: ivy.array([-1-5j, 0+0j, -1.23j]),
            b: ivy.array([7.9, 0.31-3.3j, -4.2+5.9j])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "conj",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def conj(
        self: ivy.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.conj. This method simply wraps the
        function, and so the docstring for ivy.conj also applies to this method with
        minimal changes.

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
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing output array(s) of the same dtype
            as the input array(s) with the complex conjugates of the
            complex values present in the input array.
            If x is a container of scalar(s) then a container of
            scalar(s) will be returned.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([-1j, 0.335+2.345j, 1.23+7j]),\
                          b=ivy.array([0.0, 1.2+3.3j, 1+0j]))
        >>> x.conj()
        {
            a: ivy.array([1j, 0.335-2345j, 1.23-7j]),
            b: ivy.array([0.0, 1.2-3.3j, 1-0j])
        }
        """
        return self.static_conj(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_ldexp(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.ldexp. This method simply wraps the
        function, and so the docstring for ivy.ldexp also applies to this method with
        minimal changes.

        Parameters
        ----------
        x1
            The container whose arrays should be multiplied by 2**i.
        x2
            The container whose arrays should be used to multiply x by 2**i.
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
            optional output container, for writing the result to.

        Returns
        -------
        ret
            container including x1 * 2**x2.

        Examples
        --------
        With one :class:`ivy.Container` input:
        >>> x1 = ivy.Container(a=ivy.array([1, 2, 3]), b=ivy.array([1, 5, 10]))
        >>> x2 = ivy.Container(a=ivy.array([1, 2, 3]), b=ivy.array([1, 5, 10]))
        >>> ivy.Container.static_ldexp(x1, x2)
        {
            a: ivy.array([2, 8, 24]),
            b: ivy.array([2, 160, 10240])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "ldexp",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def ldexp(
        self: ivy.Container,
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.ldexp. This method simply wraps the
        function, and so the docstring for ivy.ldexp also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            The container whose arrays should be multiplied by 2**x2.
        x2
            The container whose arrays should be used to multiply x1 by 2**x2.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            container including x1 * 2**x2.

        Examples
        --------
        With one :class:`ivy.Container` input:
        >>> x1 = ivy.Container(a=ivy.array([1, 2, 3]), b=ivy.array([1, 5, 10]))
        >>> x2 = ivy.Container(a=ivy.array([1, 2, 3]), b=ivy.array([1, 5, 10]))
        >>> x1.ldexp(x2)
        {
            a: ivy.array([2, 8, 24]),
            b: ivy.array([2, 160, 10240])
        }
        """
        return self.static_ldexp(self, x2, out=out)

    @staticmethod
    def static_lerp(
        input: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        end: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        weight: Union[ivy.Array, ivy.NativeArray, float, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.lerp. This method simply wraps the
        function, and so the docstring for ivy.lerp also applies to this method with
        minimal changes.

        Parameters
        ----------
        input
            The container whose arrays should be used as parameter: input
        end
            The container whose arrays should be used as parameter: end
        weight
            The container whose arrays or scalar should be used as parameter: weight
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
            optional output container, for writing the result to.

        Returns
        -------
        ret
            container including  input + ((end - input) * weight)

        Examples
        --------
        With one :class:`ivy.Container` input:
        >>> input = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([3., 4., 5.]))
        >>> end = ivy.array([10.])
        >>> weight = 1.1
        >>> y = ivy.Container.static_lerp(input, end, weight)
        >>> print(y)
        {
            a: ivy.array([11., 10.90000057, 10.80000019]),
            b: ivy.array([10.70000076, 10.60000038, 10.5])
        }
        >>> input = ivy.Container(a=ivy.array([10.1, 11.1]), b=ivy.array([10, 11]))
        >>> end = ivy.Container(a=ivy.array([5]))
        >>> weight = ivy.Container(a=0.5)
        >>> y = ivy.Container.static_lerp(input, end, weight)
        >>> print(y)
        {
            a: ivy.array([7.55000019, 8.05000019]),
            b: {
                a: ivy.array([7.5, 8.])
            }
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "lerp",
            input,
            end,
            weight,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def lerp(
        self: ivy.Container,
        end: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        weight: Union[ivy.Array, ivy.NativeArray, float, ivy.Container],
        /,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.lerp. This method simply wraps the
        function, and so the docstring for ivy.lerp also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            The container whose arrays should be used as parameter: input
        end
            The container whose arrays should be used as parameter: end
        weight
            The container whose arrays or scalar should be used as parameter: weight
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            container including  input + ((end - input) * weight)

        Examples
        --------
        With one :class:`ivy.Container` input:
        >>> input = ivy.Container(a=ivy.array([1, 2, 3]), b=ivy.array([1, 5, 10]))
        >>> end = ivy.Container(a=ivy.array([10, 10, 10]), b=ivy.array([20, 20, 20]))
        >>> weight = ivy.Container(a=ivy.array(0.5), b=ivy.array([0.4, 0.5, 0.6]))
        >>> input.lerp(end, weight)
        {
            a: ivy.array([5.5, 6., 6.5]),
            b: ivy.array([8.60000038, 12.5, 16.])
        }
        """
        return self.static_lerp(self, end, weight, out=out)

    @staticmethod
    def static_frexp(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.frexp. This method simply wraps the
        function, and so the docstring for ivy.frexp also applies to this method with
        minimal changes.

        Parameters
        ----------
        x
            The container whose arrays should be split into mantissa and exponent.
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
            optional output container, for writing the result to.

        Returns
        -------
        ret
            container including the mantissa and exponent of x.

        Examples
        --------
        With one :class:`ivy.Container` input:
        >>> x = ivy.Container(a=ivy.array([1, 2, 3]), b=ivy.array([1, 5, 10]))
        >>> ivy.Container.static_frexp(x)
        {
            a: (ivy.array([0.5, 0.5, 0.75]), ivy.array([1, 1, 2])),
            b: (ivy.array([0.5, 0.625, 0.625]), ivy.array([1, 3, 4]))
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "frexp",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def frexp(
        self: ivy.Container,
        /,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.frexp. This method simply wraps the
        function, and so the docstring for ivy.frexp also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            The container whose arrays should be split into mantissa and exponent.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            container including the mantissa and exponent of x.

        Examples
        --------
        With one :class:`ivy.Container` input:
        >>> x = ivy.Container(a=ivy.array([1, 2, 3]),\
                                            b=ivy.array([1, 5, 10]))
        >>> x.frexp()
        {
            a: (ivy.array([0.5, 0.5, 0.75]), ivy.array([1, 1, 2])),
            b: (ivy.array([0.5, 0.625, 0.625]), ivy.array([1, 3, 4]))
        }
        """
        return self.static_frexp(self, out=out)

    @staticmethod
    def static_modf(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.modf. This method simply wraps the
        function, and so the docstring for ivy.modf also applies to this method with
        minimal changes.

        Parameters
        ----------
        x
            The container whose arrays should be split into
            the fractional and integral parts.
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
            optional output container, for writing the result to.

        Returns
        -------
        ret
            container including the fractional and integral parts of x.

        Examples
        --------
        With one :class:`ivy.Container` input:
        >>> x = ivy.Container(a=ivy.array([1.2, 2.7, 3.9]),
        >>> b = ivy.array([-1.5, 5.3, -10.7]))
        >>> ivy.Container.static_modf(x)
        {
            a: (ivy.array([0.2, 0.7, 0.9]), ivy.array([1.0, 2.0, 3.0])),
            b: (ivy.array([-0.5, 0.3, -0.7]), ivy.array([-1.0, 5.0, -10.0]))
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "modf",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def modf(
        self: ivy.Container,
        /,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        r"""
        ivy.Container instance method variant of ivy.modf. This method simply wraps the
        function, and so the docstring for ivy.modf also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            The container whose arrays should be split into
            the fractional and integral parts.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            container including the fractional and integral parts of x.

        Examples
        --------
        With one :class:`ivy.Container` input:
        >>> x = ivy.Container(a=ivy.array([1.2, 2.7, 3.9]),
        >>> b = ivy.array([-1.5, 5.3, -10.7]))
        >>> x.modf()
        {
            a: (ivy.array([0.2, 0.7, 0.9]), ivy.array([1.0, 2.0, 3.0])),
            b: (ivy.array([-0.5, 0.3, -0.7]), ivy.array([-1.0, 5.0, -10.0]))
        }
        """
        return self.static_modf(self, out=out)

    @staticmethod
    def static_digamma(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.digamma. This method simply wraps the
        function, and so the docstring for ivy.digamma also applies to this method with
        minimal changes.

        Note
        ----
        The Ivy version only accepts real-valued inputs.

        Parameters
        ----------
        x
            Input container containing input arrays.
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
            Alternate output array in which to place the result.
            The default is None.

        Returns
        -------
        ret
            container including the digamma function computed element-wise
        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1, 0.5]),\
        ...                         b=ivy.array([-2.0, 3.0]))
        >>> ivy.Container.static_digamma(x)
        {
            a: ivy.array([-0.57721537, -1.96351004]),
            b: ivy.array([nan, 0.92278427])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "digamma",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def digamma(
        self: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.digamma. This method simply wraps
        the function, and so the docstring for ivy.digamma also applies to this method
        with minimal changes.

        Note
        ----
        The Ivy version only accepts real-valued inputs.

        Parameters
        ----------
        self
            Input container containing input arrays.
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
            Alternate output array in which to place the result.
            The default is None.

        Returns
        -------
        ret
            container including the digamma function computed element-wise
        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1, 0.5]), b=ivy.array([2.0, 3.0])
        >>> x.digamma()
        {
            a: ivy.array([-0.5772, -1.9635]),
            b: ivy.array([0.4228, 0.9228])
        }
        """
        return self.static_digamma(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_amax(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.amax. This method simply wraps the
        function, and so the docstring for ivy.amax also applies to this method with
        minimal changes.

        Parameters
        ----------
        x
            The container whose arrays should be processed.
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
            optional output container, for writing the result to.

        Returns
        -------
        ret
            container including the maximum values along the specified axis.

        Examples
        --------
        With one :class:`ivy.Container` input:
        >>> x = ivy.Container(a=ivy.array([1, 2, 3]),
        >>>                   b=ivy.array([4, 5, 6]))
        >>> ivy.Container.static_amax(x)
        {
            a: 3,
            b: 6
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "amax",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def amax(
        self: ivy.Container,
        /,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.amax. This method simply wraps the
        function, and so the docstring for ivy.amax also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            The container whose arrays should be processed.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            container including the maximum values along the specified axis.

        Examples
        --------
        With one :class:`ivy.Container` input:
        >>> x = ivy.Container(a=ivy.array([1, 2, 3]),
        >>>                   b=ivy.array([4, 5, 6]))
        >>> x.amax()
        {
            a: 3,
            b: 6
        }
        """
        return self.static_amax(self, out=out)
