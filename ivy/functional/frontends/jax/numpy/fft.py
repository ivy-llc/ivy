import ivy
import jax.numpy as jnp
from jax.numpy.fft import fft as jax_fft, ifft as jax_ifft

# Add JAX NumPy FFT functions to the JAX frontend for Ivy
ivy.jax_backend.jax_fft = jax_fft
ivy.jax_backend.jax_ifft = jax_ifft

# Wrapper functions for the FFT and IFFT functions using Ivy and JAX
def ivy_fft(x):
    if ivy.is_numpy(x):
        return jax_fft(x)
    return ivy.jax_backend.jax_fft(x)

def ivy_ifft(x):
    if ivy.is_numpy(x):
        return jax_ifft(x)
    return ivy.jax_backend.jax_ifft(x)
