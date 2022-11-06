# global
from typing import Optional, Tuple

import jax.numpy as jnp
import jaxlib.xla_extension

# local
from ivy.functional.backends.jax import JaxArray
from ivy.functional.backends.jax.device import _to_device


# Array API Standard #
# ------------------ #


def triu_indices(
    n_rows: int,
    n_cols: Optional[int] = None,
    k: Optional[int] = 0,
    /,
    *,
    device: jaxlib.xla_extension.Device,
) -> Tuple[JaxArray]:
    return _to_device(
        jnp.triu_indices(n=n_rows, k=k, m=n_cols),
        device=device,
    )
