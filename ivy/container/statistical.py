# global
from typing import Optional, Union, List, Dict, Tuple

# local
import ivy
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class ContainerWithStatistical(ContainerBase):
    def min(
        self: ivy.Container,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.handle_inplace(
            self.map(
                lambda x_, _: ivy.min(x_, axis, keepdims) if ivy.is_array(x_) else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
            out=out,
        )

    def max(
        self: ivy.Container,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.handle_inplace(
            self.map(
                lambda x_, _: ivy.max(x_, axis, keepdims) if ivy.is_array(x_) else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
            out=out,
        )

    def mean(
        self: ivy.Container,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.handle_inplace(
            self.map(
                lambda x_, _: ivy.mean(x_, axis, keepdims) if ivy.is_array(x_) else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
            out=out,
        )

    def var(
        self: ivy.Container,
        axis: Union[int, Tuple[int]] = None,
        correction: Union[int, float] = 0.0,
        keepdims: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.var.
        This method simply wraps the function, and so the
        docstring for ivy.var also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input container. Should have a floating-point data type.
        axis
            axis or axes along which variances must be computed. By default, the
            variance must be computed over the entire array for each array in the input
            container. If a tuple of integers, variances must be computed over
            multiple axes. Default: None.
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
            Default: 0.
        keepdims
            if True, the reduced axes (dimensions) must be included in the result as
            singleton dimensions, and, accordingly, the result must be compatible
            with the input array (see Broadcasting). Otherwise, if False, the
            reduced axes (dimensions) must not be included in the result.
            Default: False.
            input array. Should have a floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to.
            Default is None.
        to_apply
            If True, the method will be applied to key_chains,
            otherwise key_chains will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not
            applied. Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is False.
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
        >>> x = ivy.Container(a=ivy.array([0.0, 1.0, 2.0]), \
        b=ivy.array([3.0, 4.0, 5.0]))
        >>> y = x.var()
        >>> print(y)
        {
            a: ivy.array(0.6666667),
            b: ivy.array(0.6666667)
        }

        >>> x = ivy.Container(a=ivy.array([0.0, 1.0, 2.0]), \
        b=ivy.array([3.0, 4.0, 5.0]))
        >>> y = ivy.Container(a=ivy.array(0.), b=ivy.array(0.))
        >>> x.var(out=y)
        >>> print(y)
        {
            a: ivy.array(0.6666667),
            b: ivy.array(0.6666667)
        }

        >>> x = ivy.Container(a=ivy.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]), \
                              b=ivy.array([[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]))
        >>> y = ivy.Container(a=ivy.array([0., 0., 0.]), b=ivy.array([0., 0., 0.]))
        >>> x.var(axis=0, out=y)
        >>> print(y)
        {
            a: ivy.array([2.25, 2.25, 2.25]),
            b: ivy.array([2.25, 2.25, 2.25])
        }

        """
        return self.handle_inplace(
            self.map(
                lambda x_, _: ivy.var(x_, axis, correction, keepdims)
                if ivy.is_array(x_)
                else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
            out=out,
        )

    @staticmethod
    def static_var(
        x: ivy.Container,
        axis: Union[int, Tuple[int]] = None,
        correction: Union[int, float] = 0.0,
        keepdims: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.var.
        This method simply wraps the function, and so
        the docstring for ivy.var also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a floating-point data type.
        key_chains
            The key-chains to apply or not apply the method to.
            Default is None.
        to_apply
            If True, the method will be applied to key_chains,
            otherwise key_chains will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was
            not applied. Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is False.
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
        >>> x = ivy.Container(a=ivy.array([0.1, 0.2, 0.9]), \
                              b=ivy.array([0.7, 0.1, 0.9]))
        >>> y = ivy.Container.static_var(x)
        >>> print(y)
        {
            a:ivy.array(0.12666667),
            b:ivy.array(0.11555555)
        }

        """
        return ContainerBase.multi_map_in_static_method(
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

    def prod(
        self: ivy.Container,
        axis: Union[int, Tuple[int]] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        keepdims: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.handle_inplace(
            self.map(
                lambda x_, _: ivy.prod(x_, axis=axis, keepdims=keepdims, dtype=dtype)
                if ivy.is_array(x_)
                else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
            out=out,
        )

    def sum(
        self: ivy.Container,
        axis: Union[int, Tuple[int]] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        keepdims: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.handle_inplace(
            self.map(
                lambda x_, _: ivy.sum(x_, axis=axis, dtype=dtype, keepdims=keepdims)
                if ivy.is_array(x_)
                else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
            out=out,
        )

    def std(
        self: ivy.Container,
        axis: Union[int, Tuple[int]] = None,
        correction: Union[int, float] = 0.0,
        keepdims: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.handle_inplace(
            self.map(
                lambda x_, _: ivy.std(x_, axis, correction, keepdims)
                if ivy.is_array(x_)
                else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
            out=out,
        )

    def einsum(
        self: ivy.Container,
        equation: str,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.handle_inplace(
            self.map(
                lambda x_, _: ivy.einsum(equation, x_) if ivy.is_array(x_) else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
            out=out,
        )
