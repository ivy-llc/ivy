# global
import abc
from typing import Optional, Union

# local
import ivy


class ArrayWithRandomExtensions(abc.ABC):
    # dirichlet
    def dirichlet(
        self: ivy.Array,
        /,
        *,
        size: Optional[Union[ivy.Shape, ivy.NativeShape]] = None,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        seed: Optional[int] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.dirichlet. This method simply
        wraps the function, and so the docstring for ivy.shuffle also applies to
        this method with minimal changes.

        Parameters
        ----------
        alpha
            Sequence of floats of length k
        size
            optional int or tuple of ints, Output shape. If the given shape is,
            e.g., (m, n), then m * n * k samples are drawn. Default is None,
            in which case a vector of length k is returned.
        dtype
            output array data type. If ``dtype`` is ``None``, the output array data
            type will be the default floating-point data type. Default ``None``
        seed
            A python integer. Used to create a random seed distribution
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            The drawn samples, of shape (size, k).

        Examples
        --------
        >>> alpha = ivy.array([1.0, 2.0, 3.0])
        >>> alpha.dirichlet()
        ivy.array([0.10598304, 0.21537054, 0.67864642])

        >>> alpha = ivy.array([1.0, 2.0, 3.0])
        >>> alpha.dirichlet(size = (2,3))
        ivy.array([[[0.48006698, 0.07472073, 0.44521229],
            [0.55479872, 0.05426367, 0.39093761],
            [0.19531053, 0.51675832, 0.28793114]],

        [[0.12315625, 0.29823365, 0.5786101 ],
            [0.15564976, 0.50542368, 0.33892656],
            [0.1325352 , 0.44439589, 0.42306891]]])
        """
        return ivy.dirichlet(self, size=size, dtype=dtype, seed=seed, out=out)
