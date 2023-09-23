# global
from typing import Optional, Union, List, Dict, Sequence

# local
import ivy
from ivy.data_classes.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


class _ContainerWithStatistical(ContainerBase):
    def min(
        self: ivy.Container,
        /,
        *,
        axis: Optional[Union[int, Sequence[int], ivy.Container]] = None,
        keepdims: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.min. This method simply wraps the
        function, and so the docstring for ivy.min also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            Input container. Should have a real-valued data type.
        axis
            axis or axes along which minimum values must be computed.
            By default, the minimum value must be computed over the
            entire array. If a tuple of integers, minimum values must
            be computed over multiple axes. Default: ``None``.
        keepdims
            optional boolean, if ``True``, the reduced axes
            (dimensions) must be included in the result as
            singleton dimensions, and, accordingly, the result
            must be compatible with the input array
            (see :ref:`broadcasting`). Otherwise, if ``False``, the
            reduced axes (dimensions) must not be included in the
            result. Default: ``False``.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            if the minimum value was computed over the entire array,
            a zero-dimensional array containing the minimum value;
            otherwise, a non-zero-dimensional array containing the
            minimum values. The returned array must have the same data type
            as ``x``.

        Examples
        --------
        With :class:`ivy.Container` input:

        >> > x = ivy.Container(a=ivy.array([1, 2, 3]), \
                               b=ivy.array([2, 3, 4]))
        >> > z = x.min()
        >> > print(z)
        {
            a: ivy.array(1),
            b: ivy.array(2)
        }

        >>> x = ivy.Container(a=ivy.array([[1, 2, 3],[-1,0,2]]),
        ...                   b=ivy.array([[2, 3, 4], [0, 1, 2]]))
        >>> z = x.min(axis=1)
        >>> print(z)
        {
            a:ivy.array([1,-1]),
            b:ivy.array([2,0])
        }
        """
        return self.cont_handle_inplace(
            self.cont_map(
                lambda x_, _: (
                    ivy.min(x_, axis=axis, keepdims=keepdims)
                    if ivy.is_array(x_)
                    else x_
                ),
                key_chains=key_chains,
                to_apply=to_apply,
                prune_unapplied=prune_unapplied,
                map_sequences=map_sequences,
            ),
            out=out,
        )

    def max(
        self: ivy.Container,
        /,
        *,
        axis: Optional[Union[int, Sequence[int], ivy.Container]] = None,
        keepdims: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.max. This method simply wraps the
        function, and so the docstring for ivy.max also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            Input container. Should have a real-valued data type.
        axis
            axis or axes along which max values must be computed.
            By default, the maximum value must be computed over
            the entire array. If a tuple of integers, maximum values
            must be computed over multiple axes. Default: ``None``.
        keepdims
            optional boolean, if ``True``, the reduced axes (dimensions)
            must be included in the result as singleton dimensions,
            and, accordingly, the result must be compatible with the
            input array (see :ref:`broadcasting`). Otherwise, if ``False``,
            the reduced axes (dimensions) must not be included in the
            result. Default: ``False``.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            if the maximum value was computed over the entire array, a zero-dimensional
            array containing the maximum value; otherwise, a non-zero-dimensional array
            containing the maximum values. The returned array must have the same
            data type as ``x``.

        Examples
        --------
        With :class:`ivy.Container` input:

        >> > x = ivy.Container(a=ivy.array([1, 2, 3]), \
                               b=ivy.array([2, 3, 4]))
        >> > z = x.max()
        >> > print(z)
        {
            a: ivy.array(3),
            b: ivy.array(4)
        }
        >>> x = ivy.Container(a=ivy.array([[1, 2, 3],[-1,0,2]]),
        ...                   b=ivy.array([[2, 3, 4], [0, 1, 2]]))
        >>> z = x.max(axis=1)
        >>> print(z)
        {
            a: ivy.array([3, 2]),
            b: ivy.array([4, 2])
        }
        """
        return self.cont_handle_inplace(
            self.cont_map(
                lambda x_, _: (
                    ivy.max(x_, axis=axis, keepdims=keepdims)
                    if ivy.is_array(x_)
                    else x_
                ),
                key_chains=key_chains,
                to_apply=to_apply,
                prune_unapplied=prune_unapplied,
                map_sequences=map_sequences,
            ),
            out=out,
        )

    def mean(
        self: ivy.Container,
        /,
        *,
        axis: Optional[Union[int, Sequence[int], ivy.Container]] = None,
        keepdims: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.mean. This method simply wraps the
        function, and so the docstring for ivy.mean also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            input container. Should have a floating-point data type.
        axis
            axis or axes along which arithmetic means must be computed. By default,
            the mean must be computed over the entire array. If a Sequence of
            integers, arithmetic means must be computed over multiple axes.
            Default: ``None``.
        keepdims
            bool, if ``True``, the reduced axes (dimensions) must be included in the
            result as singleton dimensions, and, accordingly, the result must be
            compatible with the input array (see :ref:`broadcasting`). Otherwise,
            if ``False``, the reduced axes (dimensions) must not be included in
            the result. Default: ``False``.
        key_chains
            The key-chains to apply or not apply the method to.
            Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains,
            otherwise key_chains will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was
            not applied. Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output, for writing the result to.
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret
           container, if the arithmetic mean was computed over the entire array,
           a zero-dimensional array containing the arithmetic mean;
           otherwise, a non-zero-dimensional array containing the arithmetic
           means. The returned array must have the same data type as ``self``.

        Examples
        --------
        With :class:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([3., 4., 5.]))
        >>> y = x.mean()
        >>> print(y)
        {
            a: ivy.array(1.),
            b: ivy.array(4.)
        }

        >>> x = ivy.Container(a=ivy.array([0.1, 1.1]), b=ivy.array([0.1, 1.1, 2.1]))
        >>> y = x.mean(keepdims=True)
        >>> print(y)
        {
            a: ivy.array([0.60000002]),
            b: ivy.array([1.10000002])
        }

        >>> x = ivy.Container(a=ivy.array([[0.1, 1.1]]), b=ivy.array([[2., 4.]]))
        >>> y = x.mean(axis=1, keepdims=True)
        >>> print(y)
        {
            a: ivy.array([[0.60000002]]),
            b: ivy.array([[3.]])
        }

        >>> x = ivy.Container(a=ivy.array([-1., 0., 1.]), b=ivy.array([1.1, 0.2, 1.4]))
        >>> x.mean(out=x)
        >>> print(x)
        {
            a: ivy.array(0.),
            b: ivy.array(0.9)
        }

        >>> x = ivy.Container(a=ivy.array([0., -1., 1.]), b=ivy.array([1., 1., 1.]))
        >>> y = ivy.Container(a=ivy.array(0.), b=ivy.array(0.))
        >>> x.mean(out=y)
        >>> print(y)
        {
            a: ivy.array(0.),
            b: ivy.array(1.)
        }

        >>> x = ivy.Container(a=ivy.array([[0., 1., 2.], [3., 4., 5.]]),
        ...                   b=ivy.array([[3., 4., 5.], [6., 7., 8.]]))
        >>> x.mean(axis=0, out=x)
        >>> print(x)
        {
            a: ivy.array([1.5, 2.5, 3.5]),
            b: ivy.array([4.5, 5.5, 6.5])
        }

        >>> x = ivy.Container(a=ivy.array([[1., 1., 1.], [2., 2., 2.]]),
        ...                   b=ivy.array([[3., 3., 3.], [4., 4., 4.]]))
        >>> y = ivy.mean(x, axis=1)
        >>> print(y)
        {
            a: ivy.array([1., 2.]),
            b: ivy.array([3., 4.])
        }
        """
        return self.cont_handle_inplace(
            self.cont_map(
                lambda x_, _: (
                    ivy.mean(x_, axis=axis, keepdims=keepdims)
                    if ivy.is_array(x_)
                    else x_
                ),
                key_chains=key_chains,
                to_apply=to_apply,
                prune_unapplied=prune_unapplied,
                map_sequences=map_sequences,
            ),
            out=out,
        )

    def var(
        self: ivy.Container,
        /,
        *,
        axis: Optional[Union[int, Sequence[int], ivy.Container]] = None,
        correction: Union[int, float, ivy.Container] = 0.0,
        keepdims: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.var. This method simply wraps the
        function, and so the docstring for ivy.var also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            input container. Should have a floating-point data type.
        axis
            axis or axes along which variances must be computed. By default, the
            variance must be computed over the entire array for each array in the input
            container. If a tuple of integers, variances must be computed over
            multiple axes. Default: ``None``.
        correction
            degrees of freedom adjustment. Setting this parameter to a value other than
            0 has the effect of adjusting the divisor during the calculation of the
            variance according to N-c where N corresponds to the total number of
            elements over which the variance is computed and c corresponds to the
            provided degrees of freedom adjustment. When computing the variance of a
            population, setting this parameter to 0 is the standard choice (i.e.,
            the provided array contains data constituting an entire population).
            When computing the unbiased sample variance, setting this parameter to 1
            is the standard choice (i.e., the provided array contains data sampled from
            a larger population; this is commonly referred to as Bessel's correction).
            Default: ``0``.
        keepdims
            if True, the reduced axes (dimensions) must be included in the result as
            singleton dimensions, and, accordingly, the result must be compatible
            with the input array (see Broadcasting). Otherwise, if False, the
            reduced axes (dimensions) must not be included in the result.
            Default: ``False``.
            input array. Should have a floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to.
            Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains,
            otherwise key_chains will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not
            applied. Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
        ret
            a container contianing different arrays depends on parameters. see below
            for the types of arrays in the returned container if the variance was
            computed over the entire array, a zero-dimensional array containing the
            variance; otherwise, a non-zero-dimensional array containing the variances.
            The returned container must have the same data type as self.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0.0, 1.0, 2.0]),
        ...                   b=ivy.array([3.0, 4.0, 5.0]))
        >>> y = x.var()
        >>> print(y)
        {
            a: ivy.array(0.6666667),
            b: ivy.array(0.6666667)
        }

        >>> x = ivy.Container(a=ivy.array([0.0, 1.0, 2.0]),
        ...                   b=ivy.array([3.0, 4.0, 5.0]))
        >>> y = ivy.Container(a=ivy.array(0.), b=ivy.array(0.))
        >>> x.var(out=y)
        >>> print(y)
        {
            a: ivy.array(0.6666667),
            b: ivy.array(0.6666667)
        }

        >>> x = ivy.Container(a=ivy.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]),
        ...                   b=ivy.array([[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]))
        >>> y = ivy.Container(a=ivy.array([0., 0., 0.]), b=ivy.array([0., 0., 0.]))
        >>> x.var(axis=0, out=y)
        >>> print(y)
        {
            a: ivy.array([2.25, 2.25, 2.25]),
            b: ivy.array([2.25, 2.25, 2.25])
        }
        """
        return self.cont_handle_inplace(
            self.cont_map(
                lambda x_, _: (
                    ivy.var(x_, axis=axis, correction=correction, keepdims=keepdims)
                    if ivy.is_array(x_)
                    else x_
                ),
                key_chains=key_chains,
                to_apply=to_apply,
                prune_unapplied=prune_unapplied,
                map_sequences=map_sequences,
            ),
            out=out,
        )

    @staticmethod
    def _static_var(
        x: ivy.Container,
        /,
        *,
        axis: Optional[Union[int, Sequence[int], ivy.Container]] = None,
        correction: Union[int, float, ivy.Container] = 0.0,
        keepdims: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.var. This method simply wraps the
        function, and so the docstring for ivy.var also applies to this method with
        minimal changes.

        Parameters
        ----------
        x
            input array. Should have a floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to.
            Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains,
            otherwise key_chains will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was
            not applied. Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output, for writing the result to.
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret
           if the variance was computed over the entire array,
           a zero-dimensional array containing the variance;
           otherwise, a non-zero-dimensional array containing the
           variances. The returned array must have the same data
           type as x.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0.1, 0.2, 0.9]),
        ...                   b=ivy.array([0.7, 0.1, 0.9]))
        >>> y = ivy.Container.static_var(x)
        >>> print(y)
        {
            a:ivy.array(0.12666667),
            b:ivy.array(0.11555555)
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "var",
            x,
            key_chains=key_chains,
            axis=axis,
            correction=correction,
            keepdims=keepdims,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_prod(
        x: ivy.Container,
        /,
        *,
        axis: Optional[Union[int, Sequence[int], ivy.Container]] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype, ivy.Container]] = None,
        keepdims: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ):
        """
        ivy.Container static method variant of ivy.prod. This method simply wraps the
        function, and so the docstring for ivy.prod also applies to this method with
        minimal changes.

        Parameters
        ----------
        x
            input container. Should have a floating-point data type.
        axis
            axis or axes along which products must be computed. By
            default, the product must be computed over the entire
            array. If a tuple of integers, products must be
            computed over multiple axes. Default: ``None``.
        keepdims
            bool, if True, the reduced axes (dimensions) must be
            included in the result as singleton dimensions, and,
            accordingly, the result must be compatible with the
            input array (see Broadcasting). Otherwise, if False,
            the reduced axes (dimensions) must not be included
            in the result. Default: ``False``.
        dtype
            data type of the returned array.
        out
            optional output array, for writing the result to.
        key_chains
            The key-chains to apply or not apply the method to.
            Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains,
            otherwise key_chains will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was
            not applied. Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output, for writing the result to.
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            container, if the product was computed over the entire
            array, a zero-dimensional array containing the product;
            otherwise, a non-zero-dimensional array containing the
            products. The returned array must have the same data type
            as ``self``.

        Examples
        --------
        With :class:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([3., 4., 5.]))
        >>> y = ivy.Container.static_prod(x)
        >>> print(y)
        {
            a: ivy.array(0.),
            b: ivy.array(60.)
        }

        >>> x = ivy.Container(a=ivy.array([0.1, 1.1]), b=ivy.array([0.1, 1.1, 2.1]))
        >>> y = ivy.Container.static_prod(x, keepdims=True)
        >>> print(y)
        {
            a: ivy.array([0.11000001]),
            b: ivy.array([0.23100001])
        }

        >>> x = ivy.Container(a=ivy.array([[2, 1]]), b=ivy.array([[2, 3]]))
        >>> y = ivy.Container.static_prod(x, axis=1, keepdims=True)
        >>> print(y)
        {
            a: ivy.array([[2]]),
            b: ivy.array([[6]])
        }

        >>> x = ivy.Container(a=ivy.array([-1, 0, 1]), b=ivy.array([1.1, 0.2, 1.4]))
        >>> ivy.Container.static_prod(x, out=x)
        >>> print(x)
        {
            a: ivy.array(0),
            b: ivy.array(0.30800003)
        }

        >>> x = ivy.Container(a=ivy.array([0., -1., 1.]), b=ivy.array([1., 1., 1.]))
        >>> y = ivy.Container(a=ivy.array(0.), b=ivy.array(0.))
        >>> ivy.Container.static_prod(x, out=y)
        >>> print(y)
        {
            a: ivy.array(-0.),
            b: ivy.array(1.)
        }

        >>> x = ivy.Container(a=ivy.array([[0., 1., 2.], [3., 4., 5.]]),
        ...                   b=ivy.array([[3., 4., 5.], [6., 7., 8.]]))
        >>> ivy.Container.static_prod(x, axis=0, out=x)
        >>> print(x)
        {
            a: ivy.array([0., 4., 10.]),
            b: ivy.array([18., 28., 40.])
        }

        >>> x = ivy.Container(a=ivy.array([[1., 1., 1.], [2., 2., 2.]]),
        ...                   b=ivy.array([[3., 3., 3.], [4., 4., 4.]]))
        >>> y = ivy.Container.static_prod(x, axis=1)
        >>> print(y)
        {
            a: ivy.array([1., 8.]),
            b: ivy.array([27., 64.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "prod",
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

    def prod(
        self: ivy.Container,
        /,
        *,
        axis: Optional[Union[int, Sequence[int], ivy.Container]] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype, ivy.Container]] = None,
        keepdims: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.prod. This method simply wraps the
        function, and so the docstring for ivy.prod also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            input container. Should have a floating-point data type.
        axis
            axis or axes along which products must be computed. By
            default, the product must be computed over the entire
            array. If a tuple of integers, products must be
            computed over multiple axes. Default: ``None``.
        keepdims
            bool, if True, the reduced axes (dimensions) must be
            included in the result as singleton dimensions, and,
            accordingly, the result must be compatible with the
            input array (see Broadcasting). Otherwise, if False,
            the reduced axes (dimensions) must not be included
            in the result. Default: ``False``.
        dtype
            data type of the returned array.
        out
            optional output array, for writing the result to.
        key_chains
            The key-chains to apply or not apply the method to.
            Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains,
            otherwise key_chains will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was
            not applied. Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output, for writing the result to.
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            container, if the product was computed over the entire
            array, a zero-dimensional array containing the product;
            otherwise, a non-zero-dimensional array containing the
            products. The returned array must have the same data type
            as ``self``.

        Examples
        --------
        With :class:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([3., 4., 5.]))
        >>> y = x.prod()
        >>> print(y)
        {
            a: ivy.array(0.),
            b: ivy.array(60.)
        }

        >>> x = ivy.Container(a=ivy.array([0.1, 1.1]), b=ivy.array([0.1, 1.1, 2.1]))
        >>> y = x.prod(keepdims=True)
        >>> print(y)
        {
            a: ivy.array([0.11000001]),
            b: ivy.array([0.23100001])
        }

        >>> x = ivy.Container(a=ivy.array([[2, 1]]), b=ivy.array([[2, 3]]))
        >>> y = x.prod(axis=1, keepdims=True)
        >>> print(y)
        {
            a: ivy.array([[2]]),
            b: ivy.array([[6]])
        }

        >>> x = ivy.Container(a=ivy.array([-1, 0, 1]), b=ivy.array([1.1, 0.2, 1.4]))
        >>> y = ivy.Container(a=ivy.array(0.), b=ivy.array(0.))
        >>> x.prod(out=y)
        >>> print(y)
        {
            a: ivy.array(0),
            b: ivy.array(0.30800003)
        }

        >>> x = ivy.Container(a=ivy.array([0., -1., 1.]), b=ivy.array([1., 1., 1.]))
        >>> y = ivy.Container(a=ivy.array(0.), b=ivy.array(0.))
        >>> x.prod(out=y)
        >>> print(y)
        {
            a: ivy.array(-0.),
            b: ivy.array(1.)
        }

        >>> x = ivy.Container(a=ivy.array([[0., 1., 2.], [3., 4., 5.]]),
        ...                   b=ivy.array([[3., 4., 5.], [6., 7., 8.]]))
        >>> y = ivy.Container(a=ivy.zeros(3), b=ivy.zeros(3))
        >>> x.prod(axis=0, out=y)
        >>> print(y)
        {
            a: ivy.array([0., 4., 10.]),
            b: ivy.array([18., 28., 40.])
        }

        >>> x = ivy.Container(a=ivy.array([[1., 1., 1.], [2., 2., 2.]]),
        ...                   b=ivy.array([[3., 3., 3.], [4., 4., 4.]]))
        >>> y = x.prod(axis=1)
        >>> print(y)
        {
            a: ivy.array([1., 8.]),
            b: ivy.array([27., 64.])
        }
        """
        return self._static_prod(
            self,
            axis=axis,
            dtype=dtype,
            keepdims=keepdims,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def _static_sum(
        x: ivy.Container,
        /,
        *,
        axis: Optional[Union[int, Sequence[int], ivy.Container]] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype, ivy.Container]] = None,
        keepdims: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.cont_multi_map_in_function(
            "sum",
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

    def sum(
        self: ivy.Container,
        /,
        *,
        axis: Optional[Union[int, Sequence[int], ivy.Container]] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype, ivy.Container]] = None,
        keepdims: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self._static_sum(
            self,
            axis=axis,
            dtype=dtype,
            keepdims=keepdims,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def std(
        self: ivy.Container,
        /,
        *,
        axis: Optional[Union[int, Sequence[int], ivy.Container]] = None,
        correction: Union[int, float, ivy.Container] = 0.0,
        keepdims: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.std. This method simply wraps the
        function, and so the docstring for ivy.std also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            input container.
        axis
            axis or axes along which standard deviation must be computed.
            By default, the product must be computed over the entire
            array. If a tuple of integers, products must be
            computed over multiple axes. Default: ``None``.
        correction
            degrees of freedom adjustment. Setting this parameter to a
            value other than ``0`` has the effect of adjusting the
            divisor during the calculation of the standard deviation
            according to ``N-c`` where ``N`` corresponds to the total
            number of elements over which the standard deviation is
            computed and ``c`` corresponds to the provided degrees of
            freedom adjustment. When computing the standard deviation
            of a population, setting this parameter to ``0`` is the
            standard choice (i.e., the provided array contains data
            constituting an entire population). When computing
            the corrected sample standard deviation, setting this
            parameter to ``1`` is the standard choice (i.e., the
            provided array contains data sampled from a larger
            population; this is commonly referred to as Bessel's
            correction). Default: ``0``.
        keepdims
            bool, if True, the reduced axes (dimensions) must be
            included in the result as singleton dimensions, and,
            accordingly, the result must be compatible with the
            input array (see Broadcasting). Otherwise, if False,
            the reduced axes (dimensions) must not be included
            in the result. Default: ``False``.
        out
            optional output array, for writing the result to.
        key_chains
            The key-chains to apply or not apply the method to.
            Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains,
            otherwise key_chains will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was
            not applied. Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output, for writing the result to.
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            container, if the standard deviation was computed over the
            entire array, a zero-dimensional array containing the
            standard deviation; otherwise, a non-zero-dimensional array
            containing the respectve standard deviations. The returned
            array must have the same data type as ``self``.

        Examples
        --------
        With :class:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([0., 2.]), b=ivy.array([-4., 5.]))
        >>> y = x.std()
        >>> print(y)
        {
            a: ivy.array(1.),
            b: ivy.array(4.5)
        }

        >>> x = ivy.Container(a=ivy.array([0.1, 1.1]), b=ivy.array([0.1, 1.1, 2.1]))
        >>> y = x.std(keepdims=True)
        >>> print(y)
        {
            a: ivy.array([0.5]),
            b: ivy.array([0.81649649])
        }

        >>> x = ivy.Container(a=ivy.array([[2., 1.]]), b=ivy.array([[2., -2.]]))
        >>> y = x.std(axis=1, keepdims=True)
        >>> print(y)
        {
            a: ivy.array([[0.5]]),
            b: ivy.array([[2.]])
        }

        >>> x = ivy.Container(a=ivy.array([-1., 1., 1.]), b=ivy.array([1.1, 0.2, 1.4]))
        >>> x.std(out=x)
        >>> print(x)
        {
            a: ivy.array(0.94280904),
            b: ivy.array(0.509902)
        }

        >>> x = ivy.Container(a=ivy.array([0., -2., 1.]), b=ivy.array([1., 1., 1.]))
        >>> y = ivy.Container(a=ivy.array(0.), b=ivy.array(0.))
        >>> x.std(out=y)
        >>> print(y)
        {
            a: ivy.array(1.2472192),
            b: ivy.array(0.)
        }

        >>> x = ivy.Container(a=ivy.array([[-1., 1., 2.], [2., 2., 2.]]),
        ...                   b=ivy.array([[3., 0., -3.], [4., 1., 4.]]))
        >>> y = x.std(axis=1)
        >>> print(y)
        {
            a: ivy.array([1.2472192, 0.]),
            b: ivy.array([2.44948983, 1.41421354])
        }
        """
        return self.cont_handle_inplace(
            self.cont_map(
                lambda x_, _: (
                    ivy.std(x_, axis=axis, correction=correction, keepdims=keepdims)
                    if ivy.is_array(x_)
                    else x_
                ),
                key_chains=key_chains,
                to_apply=to_apply,
                prune_unapplied=prune_unapplied,
                map_sequences=map_sequences,
            ),
            out=out,
        )

    # Extra #
    # ----- #

    @staticmethod
    def _static_cumsum(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        axis: Union[int, ivy.Container] = 0,
        exclusive: Union[bool, ivy.Container] = False,
        reverse: Union[bool, ivy.Container] = False,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype, ivy.Container]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.cumsum. This method simply wraps the
        function, and so the docstring for ivy.cumsum also applies to this method with
        minimal changes.

        Parameters
        ----------
        x
            Input array or container to apply cumsum.
        axis
            Axis along which the cumulative sum is computed. Default is ``0``.
        exclusive
            Whether to perform cumsum exclusively. Default is ``False``.
        reverse
            Whether to perform the cumsum from last to first element in the selected
            axis. Default is ``False`` (from first to last element)
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
            Container whose leaves hold the result of applying cumsum
            at each original leaf arrays along the specified axis.

        Examples
        --------
        With :class:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([[1, 2, 3], [2, 4, 5]]),
        ...                   b=ivy.array([[4, 5, 6], [2, 3, 1 ]]))
        >>> y = ivy.Container.static_cumsum(x, axis=0)
        >>> print(y)
        {
            a: ivy.array([[1, 2, 3],
                          [3, 6, 8]]),
            b: ivy.array([[4, 5, 6],
                          [6, 8, 7]])
        }

        >>> x = ivy.Container(a=ivy.array([[1, 3, 5]]),
        ...                   b=ivy.array([[3, 5, 7]]))
        >>> y = ivy.Container.static_cumsum(x, axis=0,
        ...                      exclusive=False, reverse=True, dtype='float32')
        >>> print(y)
        {
            a: ivy.array([[1., 3., 5.]]),
            b: ivy.array([[3., 5., 7.]])
        }

        >>> x = ivy.Container(a=ivy.array([[1, 3, 4]]),
        ...                   b=ivy.array([[3, 5, 8],
        ...                                [5, 6, 5]]),
        ...                   c=ivy.array([[2, 4, 1],
        ...                                [3, 6, 9],
        ...                                [0, 2, 3]]))
        >>> y = ivy.Container(a = ivy.zeros((1, 3)),
        ...                   b = ivy.zeros((2, 3)),
        ...                   c = ivy.zeros((3,3)))
        >>> ivy.cumsum(x,axis=1,exclusive=True, reverse=False, out=y)
        >>> print(y)
        {
            a: ivy.array([[0, 1, 4]]),
            b: ivy.array([[0, 3, 8],
                          [0, 5, 11]]),
            c: ivy.array([[0, 2, 6],
                          [0, 3, 9],
                          [0, 0, 2]])
        }

        >>> x = ivy.Container(a=ivy.array([[1, 3, 4], [5, 7, 8], [9, 10, 11]]),
        ...                   b=ivy.array([[3, 4, 5], [4, 5, 6], [5, 6, 7]]))
        >>> y = ivy.Container(a= ivy.zeros((3, 3)), b= ivy.zeros((3, 3)))
        >>> ivy.Container.static_cumsum(x, axis=1, exclusive=True, reverse=True, out=y)
        >>> print(y)
        {
            a: ivy.array([[7, 4, 0],
                          [15, 8, 0],
                          [21, 11, 0]]),
            b: ivy.array([[9, 5, 0],
                          [11, 6, 0],
                          [13, 7, 0]])
        }
        >>> x = ivy.Container(a=ivy.array([[1],
        ...                                [1]]),
        ...                   b=ivy.array([[6, 8, 7],
        ...                                [2, 0, 1]]),
        ...                   c=ivy.array([[1, 2],
        ...                                [3, 4],
        ...                                [6, 4]]))
        >>> ivy.Container.static_cumsum(x, axis=0, out=x)
        >>> print(x)
        {
            a: ivy.array([[1],
                          [2]]),
            b: ivy.array([[6, 8, 7],
                          [8, 8, 8]]),
            c: ivy.array([[1, 2],
                          [4, 6],
                          [10, 10]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "cumsum",
            x,
            axis=axis,
            exclusive=exclusive,
            reverse=reverse,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            out=out,
        )

    def cumsum(
        self: ivy.Container,
        axis: Union[int, ivy.Container] = 0,
        exclusive: Union[bool, ivy.Container] = False,
        reverse: Union[bool, ivy.Container] = False,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype, ivy.Container]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.cumsum. This method simply wraps
        the function, and so the docstring for ivy.cumsum also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            Input container to apply cumsum at leaves.
        axis
            Axis along which the cumulative sum is computed. Default is ``0``.
        exclusive
            Whether to perform cumsum exclusively. Default is ``False``.
        reverse
            Whether to perform the cumsum from last to first element in the selected
            axis. Default is ``False`` (from first to last element)
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
            Container whose leaves hold the result of applying cumsum
            at each original leaf arrays along the specified axis.

        Examples
        --------
        With :class:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([[1, 2, 3],
        ...                                [2, 4, 5]]),
        ...                   b=ivy.array([[4, 5, 6],
        ...                                [2, 3, 1 ]]))
        >>> y = x.cumsum(axis=0, dtype='float64')
        >>> print(y)
        {
            a: ivy.array([[1., 2., 3.],
                          [3., 6., 8.]]),
            b: ivy.array([[4., 5., 6.],
                          [6., 8., 7.]])
        }

        >>> x = ivy.Container(a=ivy.array([[1, 3, 4],
        ...                                [5, 7, 8],
        ...                                [9, 10, 11]]),
        ...                   b=ivy.array([[3, 4, 5],
        ...                                [4, 5, 6],
        ...                                [5, 6, 7]]))
        >>> y = ivy.Container(a= ivy.zeros((3, 3)), b= ivy.zeros((3, 3)))
        >>> x.cumsum(axis=1, exclusive=False, reverse=True, out=y)
        >>> print(y)
        {
            a: ivy.array([[8, 7, 4],
                          [20, 15, 8],
                          [30, 21, 11]]),
            b: ivy.array([[12, 9, 5],
                          [15, 11, 6],
                          [18, 13, 7]])
        }

        >>> x = ivy.Container(a=ivy.array([[1, 3, 4]]),
        ...                   b=ivy.array([[3, 5, 8],
        ...                                [5, 6, 5]]),
        ...                   c=ivy.array([[2, 4, 1],
        ...                                [3, 6, 9],
        ...                                [0, 2, 3]]))
        >>> y = ivy.Container(a = ivy.zeros((1, 3)),
        ...                   b = ivy.zeros((2, 3)),
        ...                   c = ivy.zeros((3,3)))
        >>> x.cumsum(axis=1,exclusive=True, reverse=False, out=y)
        >>> print(y)
        {
            a: ivy.array([[0, 1, 4]]),
            b: ivy.array([[0, 3, 8],
                          [0, 5, 11]]),
            c: ivy.array([[0, 2, 6],
                          [0, 3, 9],
                          [0, 0, 2]])
        }

        >>> x = ivy.Container(a=ivy.array([[0, 3, 2],
        ...                                [5, 10, 2],
        ...                                [1, 10, 1]]),
        ...                   b=ivy.array([[2, 4, 5],
        ...                                [4, 5, 5],
        ...                                [0, 1, 3]]))
        >>> y = x.cumsum(axis=1,exclusive=True, reverse=True, dtype='int64')
        >>> print(y)
        {
            a: ivy.array([[5, 2, 0],
                          [12, 2, 0],
                          [11, 1, 0]]),
            b: ivy.array([[9, 5, 0],
                          [10, 5, 0],
                          [4, 3, 0]])
        }

        >>> x = ivy.Container(a=ivy.array([[0],
        ...                                [5]]),
        ...                   b=ivy.array([[6, 8, 7],
        ...                                [4, 2, 3]]),
        ...                   c=ivy.array([[1, 2],
        ...                                [3, 4],
        ...                                [6, 4]]))
        >>> x.cumsum(axis=0, out=x)
        >>> print(x)
        {
            a: ivy.array([[0],
                         [5]]),
            b: ivy.array([[6, 8, 7],
                         [10, 10, 10]]),
            c: ivy.array([[1, 2],
                         [4, 6],
                         [10, 10]])
        }
        """
        return self._static_cumsum(
            self,
            axis=axis,
            exclusive=exclusive,
            reverse=reverse,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            out=out,
        )

    @staticmethod
    def _static_cumprod(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        axis: Union[int, ivy.Container] = 0,
        exclusive: Union[bool, ivy.Container] = False,
        reverse: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype, ivy.Container]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.cumprod. This method simply wraps the
        function, and so the docstring for ivy.cumprod also applies to this method with
        minimal changes.

        Parameters
        ----------
        x
            Input array or container to cumprod.
        axis
            Axis to cumprod along. Default is ``0``.
        exclusive
            Whether to exclude the first element of the input array.
            Default is ``False``.
        reverse
            Whether to perform the cumprod from last to first element in the selected
            axis. Default is ``False`` (from first to last element)
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
            Containers with arrays cumprod at leaves along specified axis.

        Examples
        --------
        With one :class:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([1, 2, 3]), b=ivy.array([4, 5, 6]))
        >>> y = ivy.Container.static_cumprod(x, axis=0)
        >>> print(y)
        {
            a: ivy.array([1, 2, 6]),
            b: ivy.array([4, 20, 120])
        }

        >>> x = ivy.Container(a=ivy.array([[2, 3], [5, 7], [11, 13]]),
                              b=ivy.array([[3, 4], [4, 5], [5, 6]]))
        >>> y = ivy.Container(a = ivy.zeros((3, 2)), b = ivy.zeros((3, 2)))
        >>> ivy.Container.static_cumprod(x, axis=1, exclusive=True, out=y)
        >>> print(y)
        {
            a: ivy.array([[1, 2],
                          [1, 5],
                          [1, 11]]),
            b: ivy.array([[1, 3],
                          [1, 4],
                          [1, 5]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "cumprod",
            x,
            axis=axis,
            exclusive=exclusive,
            reverse=reverse,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            out=out,
        )

    def cumprod(
        self: ivy.Container,
        /,
        *,
        axis: Union[int, ivy.Container] = 0,
        exclusive: Union[bool, ivy.Container] = False,
        reverse: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype, ivy.Container]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.cumprod. This method simply wraps
        the function, and so the docstring for ivy.cumprod also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            Input container to cumprod at leaves.
        axis
            Axis along which the cumulative product is computed. Default is ``0``.
        exclusive
            Whether to exclude the first element of the input array.
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
            Containers with arrays cumprod at leaves along specified axis.

        Examples
        --------
        With one :class:`ivy.Container` instances:

        >>> x = ivy.Container(a=ivy.array([1, 2, 3]), b=ivy.array([4, 5, 6]))
        >>> y = x.cumprod(axis=0)
        >>> print(y)
        {
            a: ivy.array([1, 2, 6]),
            b: ivy.array([4, 20, 120])
        }

        >>> x = ivy.Container(a=ivy.array([[2, 3], [5, 7], [11, 13]]),
                              b=ivy.array([[3, 4], [4, 5], [5, 6]]))
        >>> y = ivy.Container(a = ivy.zeros((3, 2)), b = ivy.zeros((3, 2)))
        >>> x.cumprod(axis=1, exclusive=True, out=y)
        {
            a: ivy.array([[1, 2],
                          [1, 5],
                          [1, 11]]),
            b: ivy.array([[1, 3],
                          [1, 4],
                          [1, 5]])
        }
        """
        return self._static_cumprod(
            self,
            axis=axis,
            exclusive=exclusive,
            reverse=reverse,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            dtype=dtype,
            out=out,
        )

    def einsum(
        self: ivy.Container,
        equation: Union[str, ivy.Container],
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        >>> x = ivy.Container(a=ivy.array([[0, 1, 0],[1, 1, 0],[1, 1, 1]]),
        ...                   b=ivy.array([[0, 1, 2],[4, 5, 6],[8, 9, 10]]))
        >>> y = x.einsum('ii')
        >>> print(y)
        {
            a: ivy.array(2),
            b: ivy.array(15)
        }

        """
        return self.cont_handle_inplace(
            self.cont_map(
                lambda x_, _: ivy.einsum(equation, x_) if ivy.is_array(x_) else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
            out=out,
        )
