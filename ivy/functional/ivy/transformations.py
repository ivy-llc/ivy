from typing import Callable, Union, Sequence, Optional
from ivy import current_backend


def vmap(func: Callable,
         in_axes: Union[int, Sequence[int], Sequence[None]] = 0,
         out_axes: Optional[int] = 0) -> Callable:
    """Vectorizing map. Creates a function which maps func over argument axes.

    Parameters
    ----------
    func
        Function to be mapped over additional axes.
    in_axes
       An integer, None, or (nested) standard Python container (tuple/list) thereof specifying which
       input array axes to map over.If each positional argument to fun is an array, then in_axes can be
       an integer, a None, or a tuple of integers and Nones with length equal to the number of positional
       arguments to fun. An integer or None indicates which array axis to map over for all arguments
       (with None indicating not to map any axis), and a tuple indicates which axis to map for each
       corresponding positional argument. Axis integers must be in the range [-ndim, ndim) for each array,
       where ndim is the number of dimensions (axes) of the corresponding input array.
    out_axes
        An integer indicating where the mapped axis should appear in the output.

    Returns
    -------
    ret
        Batched/vectorized version of func with arguments that correspond to those of func, but
        with extra array axes at positions indicated by in_axes, and a return value that corresponds
        to that of fun, but with extra array axes at positions indicated by out_axes.

    This docstring is a summarised version of the
    `docstring <https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html#jax-vmap>`_ for
    vmap from JAX documentation.


    Examples
    --------
    With :code:`ivy.matmul` func and :code:`ivy.Array` input:

    >>> x = ivy.array(ivy.arange(60).reshape((3, 5, 4)))
    >>> y = ivy.array(ivy.arange(40).reshape((5, 4, 2)))
    >>> z = ivy.vmap(ivy.matmul, (1, 0), 1)(x, y)
    >>> print(z.shape)
    (3, 5, 2)

    """
    # TODO: optimize in the numpy and tensorflow backends and extend functionality
    return current_backend().vmap(func, in_axes, out_axes)
