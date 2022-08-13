# global
import ivy


def mean_squared_error(y_true, y_pred):
    return ivy.mean(ivy.square(ivy.subtract(y_true, y_pred)), axis=-1)


mean_squared_error.unsupported_dtypes = {"torch": ("float16", "bfloat16")}
