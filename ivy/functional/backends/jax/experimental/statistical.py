from typing import Optional, Union, Tuple, Sequence
from ivy.functional.backends.jax import JaxArray
import jax.numpy as jnp
import matplotlib.pyplot as ply

def median(
    input: JaxArray,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: Optional[bool] = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.median(
        input,
        axis=axis,
        keepdims=keepdims,
        out=out,
    )


def nanmean(
    a: JaxArray,
    /,
    *,
    axis: Optional[Union[int, Tuple[int]]] = None,
    keepdims: Optional[bool] = False,
    dtype: Optional[jnp.dtype] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.nanmean(a, axis=axis, keepdims=keepdims, dtype=dtype, out=out)


def unravel_index(
    indices: JaxArray,
    shape: Tuple[int],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.unravel_index(indices, shape)


def quantile(
    a: JaxArray,
    q: Union[float, JaxArray],
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    interpolation: str = "linear",
    keepdims: bool = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:

    if isinstance(axis, list):
        axis = tuple(axis)

    return jnp.quantile(
        a, q, axis=axis, method=interpolation, keepdims=keepdims, out=out
    )


def corrcoef(
    x: JaxArray,
    /,
    *,
    y: Optional[JaxArray] = None,
    rowvar: Optional[bool] = True,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.corrcoef(x, y=y, rowvar=rowvar)

#jax histogram
def histjax(self: ivy.Array, 
    /, 
    *,
    name: Optional[ivy.Array, ivy.NativeArray],
    data:Optional[Union[Tensorfloat64]],
    step:Optional[Union[ivyArray,ivy.NativeArray]] = None, 
    buckets:Optional[ivy.Array[int]], 
    description:Optional[ivy.Array[str]]
    ) -> ivy.Array:

    jax.numpy.histogram(a, bins=10, range=None, density=False, weights=None, cumulative=False)

    # Generate some random data
    data = jnp.random.normal(size=1000)

    # Compute the histogram with 10 bins
    hist, bin_edges = jnp.histogram(data, bins=10)

    # Plot the histogram
    plt.bar(bin_edges[:-1], hist, width=0.5)
    plt.show()