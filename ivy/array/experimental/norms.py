# global
import abc

import ivy


class ArrayWithNormsExperimental(abc.ABC):

    def l2_normalize(self, axis=None, out=None):
        """Normalizes the array to have unit L2 norm.

        Parameters
        ----------
        self
            Input array.
        axis
            Axis along which to normalize. If ``None``, the whole array
            is normalized.
        out
            optional output array, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
        ret
            The normalized array.

        Examples
        --------
        >>> x = ivy.array([[1., 2.], [3., 4.]])
        >>> x.l2_normalize(axis=1)
        ivy.array([[0.4472, 0.8944],
                   [0.6, 0.8]])
        """
        return ivy.l2_normalize(self, axis=axis, out=out)
