import ivy


def transpose(array, /, *, axes = None):
    if axes is None:
        axes = list(range(len(array.shape)))[::-1]
    assert len(axes) > 1, "`axes` should have the same size the input array.ndim"

    return ivy.permute_dims(array, axes, out=None)