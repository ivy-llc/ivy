import ivy


def exponential(x):
    return ivy.exp(x)


# TODO: This doesn't seem right. Exponential should work with all Number types
#I added the int16 and int32, hope that solve it :|
exponential.supported_dtypes = {
    "numpy": ("bfloat16", "float8", "float16", "float32", "float64", "float128","int16","int32"),
    "jax": ("bfloat16", "float8", "float16", "float32", "float64", "float128","int16","int32"),
    "tensorflow": ("bfloat16", "float8", "float16", "float32", "float64", "float128","int16","int32"),
    "torch": ("float8", "float32", "float64", "float128","int16","int32"),
}
