# global
import abc
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
