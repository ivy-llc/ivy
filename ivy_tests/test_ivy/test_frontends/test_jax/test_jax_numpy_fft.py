import ivy.numpy as ivy_np
from ivy_jax.fft import fft, ifft

def test_fft_ifft():
    # define input data
    x = ivy_np.array([1, 2, 3, 4])
    # calculate FFT
    y = fft(x)
    # calculate inverse FFT
    z = ifft(y)
    # check if original input is recovered after inverse FFT
    assert ivy_np.allclose(z, x)

test_fft_ifft()
