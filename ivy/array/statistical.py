# global
from typing import Optional, Union, Tuple
import abc

# local
import ivy

# ToDo: implement all methods here as public instance methods


class ArrayWithStatistical(abc.ABC):
    def min(
        self: ivy.Array,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.min(self._data, axis, keepdims, out=out)

    def max(
        self: ivy.Array,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.max(self._data, axis, keepdims, out=out)

    def mean(
        self: ivy.Array,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.mean(self._data, axis, keepdims, out=out)

    def var(
        self: ivy.Array,
        axis: Union[int, Tuple[int]] = None,
        correction: Union[int, float] = 0.0,
        keepdims: bool = False,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.var. This method simply 
        wraps the function, and so the docstring for ivy.var also applies 
        to this method with minimal changes.

        **Special Cases**

        Let N equal the number of elements over which to compute the variance.

        If N - correction is less than or equal to 0, the variance is NaN.

        If x_i is NaN, the variance is NaN (i.e., NaN values propagate).

        Parameters
        ----------
        self
            input array. Should have a floating-point data type.
        axis
            axis or axes along which variances must be computed. By default, the
            variance must be computed over the entire array. If a tuple of integers,
            variances must be computed over multiple axes. Default: None.
        correction
            degrees of freedom adjustment. Setting this parameter to a value other 
            than 0 has the effect of adjusting the divisor during the calculation
            of the variance according to N-c where N corresponds to the total 
            number of elements over which the variance is computed and c corresponds
            to the provided degrees of freedom adjustment. When computing the variance
            of a population, setting this parameter to 0 is the standard choice
            (i.e., the provided array contains data constituting an entire population).
            When computing the unbiased sample variance, setting this parameter to 1 
            is the standard choice (i.e., the provided array contains data sampled
            from a larger population; this is commonly referred to as Bessel's
            correction). Default: 0.
        keepdims
            if True, the reduced axes (dimensions) must be included in the result as
            singleton dimensions, and, accordingly, the result must be compatible
            with the input array (see Broadcasting). Otherwise, if False, the
            reduced axes (dimensions) must not be included in the result.
            Default: False.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            if the variance was computed over the entire array, a zero-dimensional array
            containing the variance; otherwise, a non-zero-dimensional array containing
            the variances. The returned array must have the same data type as x.

        Examples
        --------
        >>> x = ivy.array([[0.0, 1.0, 2.0], \
        [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
        >>> y = x.var()
        >>> print(y)
        ivy.array(6.6666665)

        >>> x = ivy.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], \
        [6.0, 7.0, .08]])
        >>> y = x.var(axis=0)
        >>> print(y)
        ivy.array([6. , 6. , 4.1])

        >>> x = ivy.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], \
        [6.0, 7.0, .08]])
        >>> y = ivy.array([0., 0., 0.])
        >>> x.var(axis=1, out=y)
        >>> print(y)
        ivy.array([0.667, 0.667, 9.33 ])

        """
        return ivy.var(self._data, axis, correction, keepdims, out=out)

    def prod(
        self: ivy.Array,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.prod(self._data, axis=axis, keepdims=keepdims, dtype=dtype, out=out)

    def sum(
        self: ivy.Array,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.sum(self._data, axis=axis, dtype=dtype, keepdims=keepdims, out=out)

    def std(
        self: ivy.Array,
        axis: Union[int, Tuple[int]] = None,
        correction: Union[int, float] = 0.0,
        keepdims: bool = False,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.std(self._data, axis, correction, keepdims, out=out)

    def einsum(
        self: ivy.Array,
        equation: str,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.einsum(equation, self._data, out=out)
