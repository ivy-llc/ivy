# global
import abc

# local
import ivy


class _ArrayWithGeneralExperimental(abc.ABC):
    def isin(
        self: ivy.Array,
        test_elements: ivy.Array,
        /,
        *,
        assume_unique: bool = False,
        invert: bool = False,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.isin. This method simply
        wraps the function, and so the docstring for ivy.isin also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array
        test_elements
            values against which to test for each input element
        assume_unique
            If True, assumes both elements and test_elements contain unique elements,
            which can speed up the calculation. Default value is False.
        invert
            If True, inverts the boolean return array, resulting in True values for
            elements not in test_elements. Default value is False.

        Returns
        -------
        ret
            output a boolean array of the same shape as elements that is True for
            elements in test_elements and False otherwise.

        Examples
        --------
        >>> x = ivy.array([[10, 7, 4], [3, 2, 1]])
        >>> y = ivy.array([1, 2, 3])
        >>> x.isin(y)
        ivy.array([[False, False, False], [ True,  True,  True]])

        >>> x = ivy.array([3, 2, 1, 0])
        >>> y = ivy.array([1, 2, 3])
        >>> x.isin(y, invert=True)
        ivy.array([False, False, False,  True])
        """
        return ivy.isin(
            self._data, test_elements, assume_unique=assume_unique, invert=invert
        )
