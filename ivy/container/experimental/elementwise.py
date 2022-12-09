# global
from typing import Optional, Union, List, Dict, Tuple
from numbers import Number

# local
import ivy
from ivy.container.base import ContainerBase


class ContainerWithElementWiseExperimental(ContainerBase):
    @staticmethod
    def static_sinc(
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
        ivy.Container static method variant of ivy.sinc. This method simply
        wraps the function, and so the docstring for ivy.sinc also
        applies to this method with minimal changes.

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
        return ContainerBase.multi_map_in_static_method(
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
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.sinc. This method simply
        wraps the function, and so the docstring for ivy.sinc also
        applies to this method with minimal changes.

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
    def static_lcm(
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
        ivy.Container static method variant of ivy.lcm. This method simply wraps the
        function, and so the docstring for ivy.lcm also applies to this method
        with minimal changes.

        Parameters
        ----------
        x1
            first input container.
        x2
            second input container.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            a container containing the element-wise least common multiples
            of the arrays contained in x1 and x2.

        Examples
        --------
        >>> x1=ivy.Container(a=ivy.array([2, 3, 4]),
        ...                  b=ivy.array([6, 54, 62, 10]))
        >>> x2=ivy.Container(a=ivy.array([5, 8, 15]),
        ...                  b=ivy.array([32, 40, 25, 13]))
        >>> ivy.Container.lcm(x1, x2)
        {
            a: ivy.array([10, 21, 60]),
            b: ivy.array([96, 1080, 1550, 130])
        }

        """
        return ContainerBase.multi_map_in_static_method(
            "lcm",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def lcm(
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
        ivy.Container instance method variant of ivy.lcm. This method simply wraps the
        function, and so the docstring for ivy.lcm also applies to this method
        with minimal changes.

        Parameters
        ----------
        x1
            first input container.
        x2
            second input container.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            a container containing the the element-wise least common multiples
            of the arrays contained in x1 and x2.

        Examples
        --------
        >>> x1=ivy.Container(a=ivy.array([2, 3, 4]),
        ...                  b=ivy.array([6, 54, 62, 10]))
        >>> x2=ivy.Container(a=ivy.array([5, 8, 15]),
        ...                  b=ivy.array([32, 40, 25, 13]))
        >>> x1.lcm(x2)
        {
            a: ivy.array([10, 24, 60]),
            b: ivy.array([96, 1080, 1550, 130])
        }

        """
        return self.static_lcm(
            self,
            x2,
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
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.fmod. This method simply wraps
        the function, and so the docstring for ivy.fmod also applies to this method
        with minimal changes.

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
        return ContainerBase.multi_map_in_static_method(
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
        """ivy.Container instance method variant of ivy.fmod. This method simply
        wraps the function, and so the docstring for ivy.fmod also applies to this
        method with minimal changes.

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
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.fmax. This method simply wraps
        the function, and so the docstring for ivy.fmax also applies to this method
        with minimal changes.

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
        return ContainerBase.multi_map_in_static_method(
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
        """ivy.Container instance method variant of ivy.fmax. This method simply
        wraps the function, and so the docstring for ivy.fmax also applies to this
        method with minimal changes.

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
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
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
        return ContainerBase.multi_map_in_static_method(
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
        """ivy.Container instance method variant of ivy.float_power. This method simply
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
    def static_exp2(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container, float, list, tuple],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.exp2. This method simply wraps
        the function, and so the docstring for ivy.exp2 also applies to this
        method with minimal changes.

        Parameters
        ----------
        x
            container with the base input arrays.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with element-wise 2 to the power
            of input arrays elements.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1, 2, 3]),\
                               b=[5, 6, 7])
        >>> ivy.Container.static_exp2(x)
        {
            a: ivy.array([2.,  4.,  8.])
            b: ivy.array([32., 64., 128.])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "exp2",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def exp2(
        self: ivy.Container,
        /,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.exp2. This method simply
        wraps the function, and so the docstring for ivy.exp2 also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            container with the base input arrays.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with element-wise 2 to the power
            of input array elements.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1, 2, 3]),\
                               b=[5, 6, 7])
        >>> x.exp2()
        {
            a: ivy.array([2.,  4.,  8.])
            b: ivy.array([32., 64., 128.])
        }
        """
        return self.static_exp2(self, out=out)

    @staticmethod
    def static_copysign(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container, Number],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container, Number],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.copysign. This method simply wraps
        the function, and so the docstring for ivy.copysign also applies to this
        method with minimal changes.

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
        return ContainerBase.multi_map_in_static_method(
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
        """ivy.Container instance method variant of ivy.copysign. This method simply
        wraps the function, and so the docstring for ivy.copysign also applies to
        this method with minimal changes.

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
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: Optional[bool] = False,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.count_nonzero. This method simply
        wraps the function, and so the docstring for ivy.count_nonzero also applies
        to this method with minimal changes.

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
        return ContainerBase.multi_map_in_static_method(
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
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: Optional[bool] = False,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.count_nonzero. This method
        simply wraps the function, and so the docstring for ivy.count_nonzero also
        applies to this method with minimal changes.

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
        axis: Optional[Union[tuple, int]] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        keepdims: Optional[bool] = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.nansum. This method simply wraps
        the function, and so the docstring for ivy.nansum also applies to this method
        with minimal changes.

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
        return ContainerBase.multi_map_in_static_method(
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
        axis: Optional[Union[tuple, int]] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        keepdims: Optional[bool] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.nansum. This method simply
        wraps the function, and so the docstring for ivy.nansum also applies to this
        method with minimal changes.

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
    def static_gcd(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container, int, list, tuple],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container, int, list, tuple],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.gcd. This method simply wraps
        the function, and so the docstring for ivy.gcd also applies to this
        method with minimal changes.

        Parameters
        ----------
        x1
            first input container with array-like items.
        x2
            second input container with array-like items.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with element-wise gcd of input arrays.

        Examples
        --------
        >>> x1 = ivy.Container(a=ivy.array([1, 2, 3]),\
                               b=ivy.array([1, 2, 3]))
        >>> x2 = ivy.Container(a=ivy.array([5, 6, 7]),\
                               b=10)
        >>> ivy.Container.static_gcd(x1, x2)
        {
            a: ivy.array([1.,  1.,  3.])
            b: ivy.array([1., 2., 1.])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "gcd",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def gcd(
        self: ivy.Container,
        x2: ivy.Container,
        /,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.gcd. This method simply
        wraps the function, and so the docstring for ivy.gcd also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            first input container with array-like items.
        x2
            second input container with array-like items.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with element-wise gcd of input arrays.

        Examples
        --------
        >>> x1 = ivy.Container(a=ivy.array([1, 2, 3]),\
                               b=ivy.array([1, 2, 3]))
        >>> x2 = ivy.Container(a=ivy.array([5, 6, 7]),\
                               b=10)
        >>> x1.gcd(x2)
        {
            a: ivy.array([1.,  1.,  3.])
            b: ivy.array([1., 2., 1.])
        }
        """
        return self.static_gcd(self, x2, out=out)

    @staticmethod
    def static_isclose(
        a: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        b: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        rtol: Optional[float] = 1e-05,
        atol: Optional[float] = 1e-08,
        equal_nan: Optional[bool] = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.isclose. This method simply wraps
        the function, and so the docstring for ivy.isclose also applies to this method
        with minimal changes.

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
        return ContainerBase.multi_map_in_static_method(
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
        rtol: Optional[float] = 1e-05,
        atol: Optional[float] = 1e-08,
        equal_nan: Optional[bool] = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.isclose. This method simply
        wraps the function, and so the docstring for ivy.isclose also applies to this
        method with minimal changes.

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
    def static_isposinf(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container, float, list, tuple],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.isposinf. This method simply wraps
        the function, and so the docstring for ivy.isposinf also applies to this
        method with minimal changes.

        Parameters
        ----------
        x
            container with the base input arrays.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including a boolean array with values
            True where the corresponding element of the input
            is positive infinity and values False where the
            element of the input is not positive infinity.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1, ivy.inf, -ivy.inf]),\
                                b=ivy.array([5, ivy.inf, ivy.inf]))
        >>> ivy.Container.static_isposinf(x)
        {
            a: ivy.array([False, True, False]),
            b: ivy.array([False, True, True])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "isposinf",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def isposinf(
        self: ivy.Container,
        /,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.isposinf. This method simply
        wraps the function, and so the docstring for ivy.isposinf also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            container with the base input arrays.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Returns container including a boolean array with values
            True where the corresponding element of the input
            is positive infinity and values False where the
            element of the input is not positive infinity.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1, ivy.inf, -ivy.inf]),\
                               b=ivy.array([5, ivy.inf, ivy.inf]))
        >>> x.isposinf()
        {
            a: ivy.array([False, True, False]),
            b: ivy.array([False, True, True])
        }
        """
        return self.static_isposinf(self, out=out)

    @staticmethod
    def static_isneginf(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container, float, list, tuple],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.isneginf. This method simply wraps
        the function, and so the docstring for ivy.isneginf also applies to this
        method with minimal changes.

        Parameters
        ----------
        x
            container with the base input arrays.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including a boolean array with values
            True where the corresponding element of the input
            is negative infinity and values False where the
            element of the input is not negative infinity.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1, ivy.inf, -ivy.inf]),\
                                b=ivy.array([5, -ivy.inf, -ivy.inf]))
        >>> ivy.Container.static_isneginf(x)
        {
            a: ivy.array([False, False, True]),
            b: ivy.array([False, True, True])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "isneginf",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def isneginf(
        self: ivy.Container,
        /,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.isneginf. This method simply
        wraps the function, and so the docstring for ivy.isneginf also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            container with the base input arrays.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Returns container including a boolean array with values
            True where the corresponding element of the input
            is negative infinity and values False where the
            element of the input is not negative infinity.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1, ivy.inf, -ivy.inf]),\
                               b=ivy.array([5, -ivy.inf, -ivy.inf]))
        >>> x.isneginf()
        {
            a: ivy.array([False, False, True]),
            b: ivy.array([False, True, True])
        }
        """
        return self.static_isneginf(self, out=out)

    @staticmethod
    def static_nan_to_num(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        copy: Optional[bool] = True,
        nan: Optional[Union[float, int]] = 0.0,
        posinf: Optional[Union[float, int]] = None,
        neginf: Optional[Union[float, int]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.nan_to_num. This method simply wraps
        the function, and so the docstring for ivy.nan_to_num also applies to this
        method with minimal changes.

        Parameters
        ----------
        x
            Input container with array items.
        copy
            Whether to create a copy of x (True) or to replace values in-place (False).
            The in-place operation only occurs if casting to an array does not require
            a copy. Default is True.
        nan
            Value to be used to fill NaN values. If no value is passed then NaN values
            will be replaced with 0.0.
        posinf
            Value to be used to fill positive infinity values. If no value is passed
            then positive infinity values will be replaced with a very large number.
        neginf
            Value to be used to fill negative infinity values.
            If no value is passed then negative infinity values
            will be replaced with a very small (or negative) number.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with replaced non-finite elements.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1, 2, 3, nan]),\
                               b=ivy.array([1, 2, 3, inf]))
        >>> ivy.Container.static_nan_to_num(x, posinf=5e+100)
        {
            a: ivy.array([1.,  1.,  3.,  0.0])
            b: ivy.array([1., 2., 1.,  5e+100])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "nan_to_num",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            copy=copy,
            nan=nan,
            posinf=posinf,
            neginf=neginf,
            out=out,
        )

    def nan_to_num(
        self: ivy.Container,
        /,
        *,
        copy: Optional[bool] = True,
        nan: Optional[Union[float, int]] = 0.0,
        posinf: Optional[Union[float, int]] = None,
        neginf: Optional[Union[float, int]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.nan_to_num. This method simply
        wraps the function, and so the docstring for ivy.nan_to_num also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            Input container with array items.
        copy
            Whether to create a copy of x (True) or to replace values in-place (False).
            The in-place operation only occurs if casting to an array does not require
            a copy. Default is True.
        nan
            Value to be used to fill NaN values. If no value is passed then NaN values
            will be replaced with 0.0.
        posinf
            Value to be used to fill positive infinity values. If no value is passed
            then positive infinity values will be replaced with a very large number.
        neginf
            Value to be used to fill negative infinity values.
            If no value is passed then negative infinity values
            will be replaced with a very small (or negative) number.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with replaced non-finite elements.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1, 2, 3, nan]),\
                               b=ivy.array([1, 2, 3, inf]))
        >>> x.nan_to_num(posinf=5e+100)
        {
            a: ivy.array([1.,  1.,  3.,  0.0])
            b: ivy.array([1., 2., 1.,  5e+100])
        }
        """
        return self.static_nan_to_num(
            self, copy=copy, nan=nan, posinf=posinf, neginf=neginf, out=out
        )

    @staticmethod
    def static_logaddexp2(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container, float, list, tuple],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container, float, list, tuple],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.logaddexp2. This method simply wraps
        the function, and so the docstring for ivy.logaddexp2 also applies to this
        method with minimal changes.

        Parameters
        ----------
        x1
            first input container with array-like items.
        x2
            second input container with array-like items.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with element-wise logaddexp2 of input arrays.

        Examples
        --------
        >>> x1 = ivy.Container(a=ivy.array([1, 2, 3]),\
                               b=ivy.array([1, 2, 3]))
        >>> x2 = ivy.Container(a=ivy.array([4, 5, 6]),\
                               b=5)
        >>> ivy.Container.static_logaddexp2(x1, x2)
        {
            a: ivy.array([4.169925, 5.169925, 6.169925])
            b: ivy.array([5.08746284, 5.169925  , 5.32192809])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "logaddexp2",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def logaddexp2(
        self: ivy.Container,
        x2: ivy.Container,
        /,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.logaddexp2. This method simply
        wraps the function, and so the docstring for ivy.logaddexp2 also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            first input container with array-like items.
        x2
            second input container with array-like items.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with element-wise logaddexp2 of input arrays.

        Examples
        --------
        >>> x1 = ivy.Container(a=ivy.array([1, 2, 3]),\
                               b=ivy.array([1, 2, 3]))
        >>> x2 = ivy.Container(a=ivy.array([4, 5, 6]),\
                               b=5)
        >>> x1.logaddexp2(x2)
        {
            a: ivy.array([4.169925, 5.169925, 6.169925])
            b: ivy.array([5.08746284, 5.169925  , 5.32192809])
        }
        """
        return self.static_logaddexp2(self, x2, out=out)

    @staticmethod
    def static_signbit(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container, float, int, list, tuple],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.signbit. This method simply wraps
        the function, and so the docstring for ivy.signbit also applies to this
        method with minimal changes.

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
        return ContainerBase.multi_map_in_static_method(
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
        """ivy.Container instance method variant of ivy.signbit. This method simply
        wraps the function, and so the docstring for ivy.signbit also applies to
        this method with minimal changes.

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
    def static_allclose(
        x1: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        rtol: Optional[float] = 1e-05,
        atol: Optional[float] = 1e-08,
        equal_nan: Optional[bool] = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Array] = None,
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
            a: true,
            b: true
        }

        >>> x1 = ivy.Container(a=ivy.array([1., 2., 3.]),\
        ...                         b=ivy.array([1., 2., 3.]))
        >>> x2 = ivy.Container(a=ivy.array([1., 2., 3.0003]),\
        ...                         b=ivy.array([1.0006, 2., 3.]))
        >>> y = ivy.Container.static_allclose(x1, x2, rtol=1e-3)
        >>> print(y)
        {
            a: true,
            b: true
        }
        """
        return ContainerBase.multi_map_in_static_method(
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
        rtol: Optional[float] = 1e-05,
        atol: Optional[float] = 1e-08,
        equal_nan: Optional[bool] = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.allclose. This method simply
        wraps the function, and so the docstring for ivy.allclose also applies to this
        method with minimal changes.

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
            a: true,
            b: true
        }

        >>> x1 = ivy.Container(a=ivy.array([1., 2., 3.]),\
        ...                         b=ivy.array([1., 2., 3.]))
        >>> x2 = ivy.Container(a=ivy.array([1., 2., 3.0003]),\
        ...                         b=ivy.array([1.0006, 2., 3.]))
        >>> y = x1.allclose(x2, rtol=1e-3)
        >>> print(y)
        {
            a: true,
            b: true
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
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container, int, list, tuple],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.diff. This method simply wraps
        the function, and so the docstring for ivy.diff also applies to this
        method with minimal changes.

        Parameters
        ----------
        x
            input container with array-like items.
        
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with the n-th discrete difference along
            the given axis.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1, 2, 4, 7, 0]),\
                               b=ivy.array([1, 2, 4, 7, 0]))
        >>> ivy.Container.static_diff(x)
        {
            a: ivy.array([ 1,  2,  3, -7])
            b: ivy.array([ 1,  2,  3, -7])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "diff",
            x,
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
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.diff. This method simply
        wraps the function, and so the docstring for ivy.diff also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input container with array-like items.
        
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with the n-th discrete difference along the
            given axis.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([1, 2, 4, 7, 0]),\
                               b=ivy.array([1, 2, 4, 7, 0]))
        >>> ivy.Container.static_diff(x)
        {
            a: ivy.array([ 1,  2,  3, -7])
            b: ivy.array([ 1,  2,  3, -7])
        }
        """
        return self.static_diff(self, out=out)

    @staticmethod
    def static_fix(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.fix. This method simply wraps
        the function, and so the docstring for ivy.fix also applies to this
        method with minimal changes.

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
        return ContainerBase.multi_map_in_static_method(
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
        """ivy.Container instance method variant of ivy.fix. This method simply
        wraps the function, and so the docstring for ivy.fix also applies to
        this method with minimal changes.

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
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Array] = None,
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
        return ContainerBase.multi_map_in_static_method(
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
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.nextafter. This method simply
        wraps the function, and so the docstring for ivy.nextafter also applies to this
        method with minimal changes.

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
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.zeta. This method simply wraps
        the function, and so the docstring for ivy.zeta also applies to this method
        with minimal changes.

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
        return ContainerBase.multi_map_in_static_method(
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
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.zeta. This method simply
        wraps the function, and so the docstring for ivy.zeta also applies to this
        method with minimal changes.

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
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        spacing: Optional[Union[int, list, tuple]] = 1,
        edge_order: Optional[int] = 1,
        axis: Optional[Union[int, list, tuple]] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
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
        spacing: Optional[Union[int, list, tuple]] = 1,
        edge_order: Optional[int] = 1,
        axis: Optional[Union[int, list, tuple]] = None,
    ) -> ivy.Container:
        """Calculates gradient of x with respect to (w.r.t.) spacing
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
