from ivy import current_backend


def vmap(fun,
         in_axes=0,
         out_axes=0,
         ):
    return current_backend().vmap(fun,
                               in_axes=in_axes,
                               out_axes=out_axes)
