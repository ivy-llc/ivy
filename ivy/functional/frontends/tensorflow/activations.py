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


def hard_sigmoid(x):
    point_two = ivy.full(x.shape, 0.2)
    point_five = ivy.full(x.shape, 0.5)
    x = ivy.multiply(x, point_two)
    x = ivy.add(x, point_five)
    x = ivy.clip(x, 0., 1.)
    return x


hard_sigmoid.unsupported_dtypes = {"torch": "float16"}
