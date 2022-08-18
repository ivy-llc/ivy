import ivy


def exponential(x, /):
    if isinstance(x.shape, tuple) and len(x.shape) > 0:
        return [exponential(value) for value in x]
    else:
        return ivy.exp(x)


exponential.unsupported_dtypes = {
    "numpy": ("bfloat16", "int8", "uint8", "uint16"),
    "jax": ("int8", "uint8"),
    "tensorflow": ("int8", "uint8"),
    "torch": ("int8", "uint8"),
}
