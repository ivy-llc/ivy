import ivy


def exponential(x):
    return ivy.exp(x)


exponential.supported_dtypes = {
    "numpy": ("bfloat16", "float8", "float16", "float32", "float64", "float128"),
    "jax": ("bfloat16", "float8", "float16", "float32", "float64", "float128"),
    "tensorflow": ("bfloat16", "float8", "float16", "float32", "float64", "float128"),
    "torch": ("float8", "float32", "float64", "float128"),
}
