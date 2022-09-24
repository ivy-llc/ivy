import ivy

# Hann window.


def hann_window(x, alpha=0.5):
    """Hann window.
    Args:
        x: The input tensor.
        alpha: The alpha parameter.
    Returns:
        The windowed tensor.
    """
    return 0.5 * (1 - alpha) - 0.5 * alpha * ivy.cos(2 * ivy.pi * x)
