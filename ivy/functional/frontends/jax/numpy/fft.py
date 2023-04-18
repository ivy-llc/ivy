import jax.numpy.fft as jax_fft
import ivy.numpy as ivy_np

def fft(x):
    # convert input array to Jax array
    x_jax = ivy_np.asarray(x)
    # calculate FFT using Jax
    y_jax = jax_fft.fft(x_jax)
    # convert output array to Ivy array
    y = ivy_np.asarray(y_jax)
    return y

def ifft(x):
    # convert input array to Jax array
    x_jax = ivy_np.asarray(x)
    # calculate inverse FFT using Jax
    y_jax = jax_fft.ifft(x_jax)
    # convert output array to Ivy array
    y = ivy_np.asarray(y_jax)
    return y
