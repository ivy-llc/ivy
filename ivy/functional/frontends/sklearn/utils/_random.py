DEFAULT_SEED = 1
# Max value for our rand_r replacement (same as RAND_R_MAX in Cython code)
RAND_R_MAX = 2147483647


def our_rand_r(seed):
    """Generate a pseudo-random np.uint32 from a np.uint32 seed."""
    # seed shouldn't ever be 0.
    if seed[0] == 0:
        seed[0] = DEFAULT_SEED

    seed[0] ^= seed[0] << 13
    seed[0] ^= seed[0] >> 17
    seed[0] ^= seed[0] << 5

    # Use the modulo to make sure that we don't return a value greater than the
    # maximum representable value for signed 32-bit integers (i.e., 2^31 - 1).
    # Note that the parentheses are needed to avoid overflow:
    # here RAND_R_MAX is cast to an unsigned 32-bit integer before 1 is added.
    return seed[0] % (RAND_R_MAX + 1)


def sample_without_replacement(n_population, n_samples, method=None, random_state=None):
    # Implement your function logic here
    pass
