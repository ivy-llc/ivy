import ivy


def union_container(x: ivy.Container, y: ivy.Container, /) -> ivy.Container:
    """

    Args:
    ----
        x: The first container
        y: The second container
    ------

    Return:
    ------
        A new container that contains the union of the two input containers.
    """
    if isinstance(x, ivy.Array):
        return ivy.union_container(x, y)
    else:
        return ivy.Container(set(x) | set(y))


def union_array(x: ivy.Array, y: ivy.Array, /) -> ivy.Array:
    """

    Args:
    ----
        x: The first array.
        y: The second array.
    ------

    Return:
    ------
        A new array that contains the union of the two input arrays.
        The output array has the same dtype as the input arrays.
    """
    return ivy.unique_consecutive(ivy.concat((x, y), axis=0))
