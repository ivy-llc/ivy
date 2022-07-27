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
        ivy.Container instance method variant of ivy.var. This method simply wraps the
        function, and so the docstring for ivy.var also applies to this method
        with the minimal changes.

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
        out
            optional output container, for writing the result to.

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
        >>> x = ivy.Container(a=ivy.array([0, 1, 2]), b=ivy.array([3, 4, 5]))
        >>> y = x.var()
        >>> print(y)
        {
            a: ivy.array(0.66666667),
            b: ivy.array(0.66666667)
        }

        >>> x = ivy.Container(a=ivy.array([0, 1, 2]), b=ivy.array([3, 4, 5]))
        >>> y = ivy.Container(c=ivy.array(0.), d=ivy.array(0.)
        >>> x.var(out=y)
        >>> print(y)
        {
            c: ivy.array(0.66666667),
            d: ivy.array(0.66666667)
        }

        >>> x = ivy.Container(a=ivy.array([[0, 1, 2], [3, 4, 5]]), 
                              b=ivy.array([[6, 7, 8], [9, 10, 11]]))
        >>> y = ivy.Container(c=ivy.array([0., 0., 0.]), d=ivy.array([0., 0., 0.]))
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
