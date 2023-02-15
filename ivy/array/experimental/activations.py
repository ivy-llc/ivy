# global
import abc
from typing import Optional, Union

# local
import ivy


class ArrayWithActivationsExperimental(abc.ABC):
    def logit(self, /, *, eps=None, out=None):
        """
        ivy.Array instance method variant of ivy.logit. This method
        simply wraps the function, and so the docstring for ivy.logit
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input array.
        eps
            When eps is None the function outpus NaN where x < 0 or x > 1.
            and inf or -inf where x = 1 or x = 0, respectively.
            Otherwise if eps is defined, x is clamped to [eps, 1 - eps]
        out
            Optional output array.

        Returns
        -------
        ret
            Array containing elementwise logits of x.

        Examples
        --------
        >>> x = ivy.array([1, 0, 0.9])
        >>> z = x.logit()
        >>> print(z)
        ivy.array([       inf,       -inf, 2.19722438])

        >>> x = ivy.array([1, 2, -0.9])
        >>> z = x.logit(eps=0.2)
        >>> print(z)
        ivy.array([ 1.38629448,  1.38629448, -1.38629436])

        """
        return ivy.logit(self, eps=eps, out=out)

    def thresholded_relu(
        self: ivy.Array,
        /,
        *,
        threshold: Optional[Union[int, float]] = 0,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.thresholded_relu.
        This method simply wraps the function, and so the docstring
        for ivy.thresholded_relu also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array.
        threshold
            threshold value above which the activation is linear. Default: ``0``.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            an array with the relu activation function applied element-wise
            with custom threshold.

        Examples
        --------
        >>> x = ivy.array([-1., .2, 1.])
        >>> y = x.thresholded_relu(threshold=0.5)
        >>> print(y)
        ivy.array([0., 0., 1.])
        """
        return ivy.thresholded_relu(self._data, threshold=threshold, out=out)

    def prelu(
        self,
        slope: Union[float, ivy.NativeArray, ivy.Array],
        /,
        *,
        out: Optional["ivy.Array"] = None,
    ) -> ivy.Array:
        """
        Prelu takes input data (Array) and slope array as input,
        and produces one output data (array) where the function
        f(x) = slope * x for x < 0, f(x) = x for x >= 0., is applied
        to the data array elementwise. This operator supports unidirectional
        broadcasting (array slope should be unidirectional broadcastable to
        input tensor X);

        Parameters
        ----------
        self
            input array.
        slope
            Slope Array. The shape of slope can be smaller then first input X;
            if so, its shape must be unidirectional broadcastable to X.
        out
            Optional output array.
        Returns
        -------
        """
        return ivy.prelu(self._data, slope, out=out)
