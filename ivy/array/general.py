# global
import abc

# ToDo: implement all methods here as public instance methods


class ArrayWithGeneral(abc.ABC):
    def all_equal( *xs: Iterable[Any], equality_matrix: bool = False
    ) -> Union[bool, Union[ivy.Array, ivy.NativeArray]]:
        """Determines whether the inputs are all equal.

        Parameters
        ----------
        xs
            inputs to compare.
        equality_matrix
            Whether to return a matrix of equalities comparing each input with every other.
            Default is False.

        Returns
        -------
        ret
            Boolean, whether or not the inputs are equal, or matrix array of booleans if
            equality_matrix=True is set.

        Examples
        --------

        With :code:`ivy.Array` input:

        >>> x1 = ivy.array([1, 2, 3])
        >>> x2 = ivy.array([1, 0, 1])
        >>> y = ivy.all_equal(x1, x2, equality_matrix=False)
        >>> print(y)
        False

        >>> x1 = ivy.array([1, 0, 1, 1])
        >>> x2 = ivy.array([1, 0, 1, 1])
        >>> y = ivy.all_equal(x1, x2, equality_matrix=False)
        >>> print(y)
        True

        With :code:`ivy.NativeArray` input:

        >>> x1 = ivy.native_array([1, 2, 3])
        >>> x2 = ivy.native_array([1, 0, 1])
        >>> y = ivy.all_equal(x1, x2, equality_matrix=False)
        >>> print(y)
        False

        >>> x1 = ivy.native_array([1, 0, 1, 1])
        >>> x2 = ivy.native_array([1, 0, 1, 1])
        >>> y = ivy.all_equal(x1, x2, equality_matrix=False)
        >>> print(y)
        True

        With a mix of :code:`ivy.Array` and :code:`ivy.NativeArray` input:

        >>> x1 = ivy.array([1, 1, 0, 1.2, 1])
        >>> x2 = ivy.native_array([1, 1, 0, 0.5, 1])
        >>> y = ivy.all_equal(x1, x2, equality_matrix=False)
        >>> print(y)
        False

        >>> x1 = ivy.array([1, 1, 0, 0.5, 1])
        >>> x2 = ivy.native_array([1, 1, 0, 0.5, 1])
        >>> y = ivy.all_equal(x1, x2, equality_matrix=False)
        >>> print(y)
        True

        """
        equality_fn = ivy.array_equal if ivy.is_native_array(xs[0]) else lambda a, b: a == b
        if equality_matrix:
            num_arrays = len(xs)
            mat = [[None for _ in range(num_arrays)] for _ in range(num_arrays)]
            for i, xa in enumerate(xs):
                for j_, xb in enumerate(xs[i:]):
                    j = j_ + i
                    res = equality_fn(xa, xb)
                    if ivy.is_native_array(res):
                        # noinspection PyTypeChecker
                        res = ivy.to_scalar(res)
                    # noinspection PyTypeChecker
                    mat[i][j] = res
                    # noinspection PyTypeChecker
                    mat[j][i] = res
            return ivy.array(mat)
        x0 = xs[0]
        for x in xs[1:]:
            if not equality_fn(x0, x):
                return False
        return True