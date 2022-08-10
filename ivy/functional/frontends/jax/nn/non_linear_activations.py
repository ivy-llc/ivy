import ivy


def relu(x):
    return ivy.relu(x)


relu.unsupported_dtypes = {"torch": ("float16", "bfloat16")}


def leaky_relu(x, alpha=0.01):
    return ivy.leaky_relu(x, alpha=alpha)


leaky_relu.unsupported_dtypes = {"torch": ("float16", "bfloat16")}
