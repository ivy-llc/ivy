from typing import Optional

import ivy
from ivy.container.base import ContainerBase


class ContainerWithGeneralExperimental(ContainerBase):
    @staticmethod
    def static_isin(
        element: ivy.Container,
        test_elements: ivy.Container,
        /,
        *,
        assume_unique: Optional[bool] = False,
        invert: Optional[bool] = False,
    ) -> ContainerBase:
        """Container instance method variant of ivy.isin. This method simply
        wraps the function, and so the docstring for ivy.isin also applies to
        this method with minimal changes.

        Parameters
        ----------
        element
            input container
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
            output a boolean container of the same shape as elements that is True for
            elements in test_elements and False otherwise.

        Examples
        --------
        >>> x = ivy.Container(a=[[10, 7, 4], [3, 2, 1]],\
                              b=[3, 2, 1, 0])
        >>> y = ivy.Container(a=[1, 2, 3],\
                              b=[1, 0, 3])
        >>> ivy.Container.static_isin(x, y)
        ivy.Container(a=[[False, False, False], [ True,  True,  True]],\
                      b=[ True, False,  True])

        >>> ivy.Container.static_isin(x, y, invert=True)
        ivy.Container(a=[[ True,  True,  True], [False, False, False]],\
                      b=[False,  True, False])
        """
        return ContainerBase.cont_multi_map_in_static_method(
            "isin", element, test_elements, assume_unique=assume_unique, invert=invert
        )

    def isin(
        self: ivy.Container,
        test_elements: ivy.Container,
        /,
        *,
        assume_unique: Optional[bool] = False,
        invert: Optional[bool] = False,
    ) -> ivy.Container:
        """Container instance method variant of ivy.isin. This method simply
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
        >>> x = ivy.Container(a=[[10, 7, 4], [3, 2, 1]],\
                                b=[3, 2, 1, 0])
        >>> y = ivy.Container(a=[1, 2, 3],\
                                b=[1, 0, 3])
        >>> x.isin(y)
        ivy.Container(a=[[False, False, False], [ True,  True,  True]],\
                        b=[ True, False,  True])
        """
        return self.static_isin(
            self, test_elements, assume_unique=assume_unique, invert=invert
        )
