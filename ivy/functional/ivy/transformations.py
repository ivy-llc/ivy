from ivy import current_backend


def vmap(fun,
         in_axes=0,
         out_axes=0,
         ):
    # TODO: optimize in the numpy and tensorflow backends
    return current_backend().vmap(fun, in_axes, out_axes)
