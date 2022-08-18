import ivy


def exponential(x):
    return ivy.exp(x)


# TODO: This doesn't seem right. Exponential should work with all Number types

exponential.supported_dtypes = {
    "numpy": ("bfloat16", "float8", "float16", "float32", "float64", "float128"),
    "jax": ("bfloat16", "float8", "float16", "float32", "float64", "float128"),
    "tensorflow": ("bfloat16", "float8", "float16", "float32", "float64", "float128"),
    "torch": ("float8", "float32", "float64", "float128"),
}
