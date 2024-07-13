# global
from typing import Optional
import abc

# local
import ivy


class _ArrayWithUtilityExperimental(abc.ABC):
    def optional_get_element(
        self: Optional[ivy.Array] = None,
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """If the input is a tensor or sequence type, it returns the input. If
        the input is an optional type, it outputs the element in the input. It
        is an error if the input is an empty optional-type (i.e. does not have
        an element) and the behavior is undefined in this case.

        Parameters
        ----------
        self
            Input array
        out
            Optional output array, for writing the result to.

        Returns
        -------
        ret
            Input array if it is not None
        """
        return ivy.optional_get_element(self._data, out=out)
