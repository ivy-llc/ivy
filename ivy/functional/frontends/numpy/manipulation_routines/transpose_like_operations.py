import ivy


def transpose(array, /, *, axes=None):
    origin_axes = list(range(len(array.shape)))
    if axes is None:
        axes = origin_axes[::-1]
    try:
        assert len(axes) > 1
    except ValueError:
        raise
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

