from ._types import dtype

def __eq__(self: dtype, other: dtype, /) -> bool:
    """
    Computes the truth value of ``self == other`` in order to test for data type object equality.

    Parameters
    ----------
    self
        data type instance. May be any supported data type.
    other
        other data type instance. May be any supported data type.

    Returns
    -------
    ret
        a boolean indicating whether the data type objects are equal.

       This function conforms to the `Array API Standard
   <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
   `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.elementwise_functions.tan.html>`_
   in the standard.

   Both the description and the type hints above assumes an array input for simplicity,
   but this function is *nestable*, and therefore also accepts :code:`ivy.Container`
   instances in place of any of the arguments.


    """

__all__ = ['__eq__']
