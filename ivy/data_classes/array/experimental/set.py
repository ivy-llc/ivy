# global
import abc

# local
import ivy


class _ArrayWithSetExperimental(abc.ABC):
    def intersection(
        self: ivy.Array,
        x2: ivy.Array,
        /,
        *,
        assume_unique: bool = False,
        return_indices: bool = False,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.intersection. This method simply wraps
        the function, and so the docstring for ivy.intersection also applies to this
        method with minimal changes.

        Examples
        --------
        >>> a = ivy.array([1,2,3,5])
        >>> b = ivy.array([3,4,1,5])
        >>> a.intersection(b)
        ivy.array([1, 3, 5])
        """
        return ivy.intersection(
            self._data, x2, assume_unique=assume_unique, return_indices=return_indices
        )
