@@ -3,11 +3,11 @@


def argsort(
        x,
        /,
        *,
        axis=-1,
        kind=None,
        order=None,
    x,
    /,
    *,
    axis=-1,
    kind=None,
    order=None,
):
    return ivy.argsort(x, axis=axis)
