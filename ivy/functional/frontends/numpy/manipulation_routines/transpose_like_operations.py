import ivy


def transpose(array, /, *, axes=None):
    if array.ndim <= 2:
        return ivy.matrix_transpose(array)
    origin_axes = list(range(len(array.shape)))
    if axes is None:
        axes = origin_axes[::-1]
    try:
        assert len(axes) > 1
    except AssertionError:
        raise ValueError("`axes` should have the same size the input array.ndim")
    # use matrix_transpose for inner-most transposal
    if (origin_axes[:-2] == axes[:-2]) & (origin_axes[-1] == axes[-2]):
        return ivy.matrix_transpose(array)
    mapping = dict(zip(origin_axes, [axes.index(ax) for ax in origin_axes]))
    ia, ai = dict(zip(origin_axes, origin_axes)), dict(zip(origin_axes, origin_axes))

    for i in range(len(axes)):
        target = mapping[i]
        if ai[i] == target:
            continue
        array = ivy.swapaxes(array, ai[i], target)
        ai[ia[target]] = ai[i]
        ai[i] = target
        ia[target] = i
        ia[i] = target

    return array

