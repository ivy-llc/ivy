import ivy
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def bartlett_window(
    window_length,
    periodic=True,
    *,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False
):
    """Generate a Bartlett window.

    Parameters:
    - window_length (int): Length of the window.
    - periodic (bool): Whether the window is periodic.
    - dtype (str, optional): Data type of the output window. Defaults to None.
    - layout (str, optional): Layout of the output window. Defaults to None.
    - device (str, optional): Device of the output window. Defaults to None.
    - requires_grad (bool, optional): Whether the output window requires gradients. Defaults to False.

    Returns:
    - Ivy array: Bartlett window.
    """
    if not isinstance(window_length, int) or window_length < 0:
        raise ValueError("Window length must be a non-negative integer")

    if window_length == 1:
        return ivy.ones(window_length)
    else:
        N = window_length + 1 if periodic else window_length

        res = ivy.arange(0, N, dtype=dtype)
        res = ivy.where(
            ivy.less_equal(res, (N - 1) / 2.0),
            2.0 * res / (N - 1),
            2.0 - 2.0 * res / (N - 1),
        )

        return res[:-1] if periodic else res
