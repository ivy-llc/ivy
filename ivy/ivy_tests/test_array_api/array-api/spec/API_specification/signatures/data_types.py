from ._types import dtype

def __eq__(self: dtype, other: dtype, /) -> bool:
    """
    Computes the truth value of ``self == other`` in order to test for data type object equality.

    Parameters
    ----------
    self: dtype
        data type instance. May be any supported data type.
    other: dtype
        other data type instance. May be any supported data type.

    Returns
    -------
    out: bool
        a boolean indicating whether the data type objects are equal.
    """

all = [__eq__]
