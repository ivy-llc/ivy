import numpy as np

import ivy


def numpy_random_vonmises(mu, kappa, size=None):
    if size is None:
        size = 1

    # while len(li) < size:
    # Generate samples from the von Mises distribution using numpy
    #     u = ivy.random_uniform(low=-ivy.pi, high=ivy.pi, shape=size)
    #     v = ivy.random_uniform(low=0, high=1, shape=size)
    #
    #     condition = v < (1 + ivy.exp(kappa * i
    #     vy.cos(u - mu))) / (2 * ivy.pi * ivy.i0(kappa))
    #     selected_samples = u[condition]
    # li.extend(ivy.to_list(selected_samples))
    u = ivy.random_uniform(low=-ivy.pi, high=ivy.pi, shape=size)
    v = ivy.random_uniform(low=0, high=1, shape=size)

    condition = v < (1 + ivy.exp(kappa * ivy.cos(u - mu))) / (
        2 * ivy.pi * ivy.i0(kappa)
    )
    selected_samples = u[condition]
    return selected_samples


# Input parameters
mu = 0.5  # Mean direction parameter
kappa = 2  # Concentration parameter
sample_size = (9, 3)  # Number of samples to generate

# Generate von Mises samples using the implemented function
numpy_samples = numpy_random_vonmises(mu, kappa, sample_size)
print(len(np.random.vonmises(mu, kappa, sample_size)))
# Example usage
print(numpy_samples)
