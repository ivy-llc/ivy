# global
import ivy
from ivy.func_wrapper import with_unsupported_dtypes, with_supported_dtypes
from ivy.functional.frontends.paddle.func_wrapper import to_ivy_arrays_and_back


# --- Helpers --- #
# --------------- #


def _ndtri(y):
    """Inverse normal distribution."""

    P0 = [
        -5.99633501014107895267e1,
        9.80010754185999661536e1,
        -5.66762857469070293439e1,
        1.39312609387279679503e1,
        -1.23916583867381258016e0,
    ]

    Q0 = [
        1.95448858338141759834e0,
        4.67627912898881538453e0,
        8.63602421390890590575e1,
        -2.25462687854119370527e2,
        2.00260212380060660359e2,
        -8.20372256168333339912e1,
        1.59056225126211695515e1,
        -1.18331621121330003142e0,
    ]

    P1 = [
        4.05544892305962419923e0,
        3.15251094599893866154e1,
        5.71628192246421288162e1,
        4.40805073893200834700e1,
        1.46849561928858024014e1,
        2.18663306850790267539e0,
        -1.40256079171354495875e-1,
        -3.50424626827848203418e-2,
        -8.57456785154685413611e-4,
    ]

    Q1 = [
        1.57799883256466749731e1,
        4.53907635128879210584e1,
        4.13172038254672030440e1,
        1.50425385692907503408e1,
        2.50464946208309415979e0,
        -1.42182922854787788574e-1,
        -3.80806407691578277194e-2,
        -9.33259480895457427372e-4,
    ]

    P2 = [
        3.23774891776946035970e0,
        6.91522889068984211695e0,
        3.93881025292474443415e0,
        1.33303460815807542389e0,
        2.01485389549179081538e-1,
        1.23716634817820021358e-2,
        3.01581553508235416007e-4,
        2.65806974686737550832e-6,
        6.23974539184983293730e-9,
    ]

    Q2 = [
        6.02427039364742014255e0,
        3.67983563856160859403e0,
        1.37702099489081330271e0,
        2.16236993594496635890e-1,
        1.34204006088543189037e-2,
        3.28014464682127739104e-4,
        2.89247864745380683936e-6,
        6.79019408009981274425e-9,
    ]
    sign_change = False
    if y.size == 1 and y < 1 - ivy.exp(-2.0):
        sign_change = True
    sign_indices = ivy.argwhere(y <= 1.0 - ivy.exp(-2.0))
    y = ivy.where(y <= 1.0 - ivy.exp(-2.0), y, 1.0 - y)

    x = ivy.sqrt(-2.0 * ivy.log(y))
    x0 = x - ivy.log(x) / x

    z = 1.0 / x
    x1 = 0 * x
    if x.size > 1:
        indices_less = ivy.argwhere(x < 8.0)
        if indices_less.size != 0:
            for ind in indices_less:
                x1[ind] = ivy.divide(
                    ivy.multiply(z[ind], _polevl(z[ind], P1)),
                    _polevl(z[ind], [1.0] + Q1),
                )

        indices_greater = ivy.argwhere(x >= 8.0)
        if indices_greater.size != 0:
            for ind in indices_greater:
                x1[ind] = ivy.divide(
                    ivy.multiply(z[ind], _polevl(z[ind], P2)),
                    _polevl(z[ind], [1.0] + Q2),
                )
    else:
        if x < 8.0:
            x1 = z * _polevl(z, P1) / _polevl(z, [1.0] + Q1)
        else:
            x1 = z * _polevl(z, P2) / _polevl(z, [1.0] + Q2)

    x = x0 - x1
    if sign_indices.size != 0:
        for ind in sign_indices:
            x[ind] = -1.0 * x[ind]
    elif sign_change:
        x = -x

    return x


def _polevl(x, coefs):
    """Polynomial Evaluation."""
    answer = 0
    power = len(coefs) - 1
    for coef in coefs:
        answer += coef * x**power
        power -= 1
    return answer


# --- Main --- #
# ------------ #


@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def abs(x, name=None):
    return ivy.abs(x)


@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def acos(x, name=None):
    return ivy.acos(x)


@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def acosh(x, name=None):
    return ivy.acosh(x)


@with_unsupported_dtypes(
    {"2.5.1 and below": ("bool", "unsigned", "int8", "float16", "bfloat16")}, "paddle"
)
@to_ivy_arrays_and_back
def add(x, y, name=None):
    return ivy.add(x, y)


@with_supported_dtypes(
    {"2.5.1 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def addmm(input, x, y, beta=1.0, alpha=1.0, name=None):
    value = alpha * ivy.matmul(x, y) + (beta * input)
    return value


@with_supported_dtypes(
    {"2.5.1 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def amax(x, axis=None, keepdims=False):
    if axis is None:
        return ivy.max(x)
    if isinstance(axis, int):
        axis = [axis]
    for i in range(len(axis)):
        if axis[i] < 0:
            axis[i] += x.ndim
    for i in axis:
        if i < 0 or i >= x.ndim:
            raise ValueError("axis {} is out of range [-{}:{}]".format(i, 0, x.ndim))
    return ivy.max(x, axis=axis, keepdims=keepdims)


@with_supported_dtypes(
    {"2.5.1 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def amin(x, axis=None, keepdim=False, name=None):
    return ivy.min(x, axis=axis, keepdims=keepdim)


@with_supported_dtypes(
    {"2.5.1 and below": ("complex64", "complex128", "float32", "float64")},
    "paddle",
)
@to_ivy_arrays_and_back
def angle(x, name=None):
    return ivy.angle(x)


@with_supported_dtypes({"2.5.0 and below": "bool"}, "paddle")
@to_ivy_arrays_and_back
def any(x, axis=None, keepdim=False, name=None):
    return ivy.any(x, axis=axis, keepdims=keepdim)


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def asin(x, name=None):
    return ivy.asin(x)


@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def asinh(x, name=None):
    return ivy.asinh(x)


@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def atan(x, name=None):
    return ivy.atan(x)


@with_unsupported_dtypes({"2.4.2 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def atan2(x, y, name=None):
    return ivy.atan2(x, y)


@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def atanh(x, name=None):
    return ivy.atanh(x)


@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def ceil(x, name=None):
    return ivy.ceil(x)


@with_unsupported_dtypes({"2.4.2 and below": ("int16", "float16")}, "paddle")
@to_ivy_arrays_and_back
def conj(x, name=None):
    return ivy.conj(x)


@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def cos(x, name=None):
    return ivy.cos(x)


@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def cosh(x, name=None):
    return ivy.cosh(x)


@with_supported_dtypes(
    {"2.5.1 and below": ("int32", "int64", "float16", "float32", "float64", "bool")},
    "paddle",
)
@to_ivy_arrays_and_back
def count_nonzero(x, axis=None, keepdim=False, name=None):
    return ivy.astype(ivy.count_nonzero(x, axis=axis, keepdims=keepdim), ivy.int64)


@with_supported_dtypes(
    {
        "2.5.1 and below": (
            "int32",
            "int64",
            "float32",
            "float64",
            "complex64",
            "complex128",
        )
    },
    "paddle",
)
@to_ivy_arrays_and_back
def cumprod(x, dim=None, dtype=None, name=None):
    return ivy.cumprod(x, axis=dim, dtype=dtype)


@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def deg2rad(x, name=None):
    return ivy.deg2rad(x)


@with_supported_dtypes(
    {"2.5.1 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def diff(x, n=1, axis=-1, prepend=None, append=None, name=None):
    return ivy.diff(x, n=n, axis=axis, prepend=prepend, append=append)


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def digamma(x, name=None):
    digamma_fun = ivy.digamma
    return ivy.array(digamma_fun(x), dtype=x.dtype)


@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def divide(x, y, name=None):
    return ivy.divide(x, y)


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def erf(x, name=None):
    return ivy.erf(x)


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def erfinv(x, name=None):
    """
    Calculate the inverse error function.

    Parameters
    ----------
    x : number between -1 and 1

    Returns
    -------
    float


    Examples
    --------
    >>> erfinv(0.1)
    0.08885599
    """
    if ivy.max(ivy.abs(x)) >= 1:
        raise ValueError(" 'x' must be between -1 and 1 inclusive")

    return _ndtri((x + 1) / 2.0) / ivy.sqrt(2.0)


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def exp(x, name=None):
    return ivy.exp(x)


@with_supported_dtypes({"2.5.1 and below": ("float16", "float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def expm1(x, name=None):
    return ivy.expm1(x)


@with_supported_dtypes(
    {"2.5.1 and below": ("bfloat16", "float32", "float64")}, "paddle"
)
@to_ivy_arrays_and_back
def floor(x, name=None):
    return ivy.floor(x)


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def floor_divide(x, y, name=None):
    return ivy.floor_divide(x, y)


@with_unsupported_dtypes({"2.5.1 and below": "bfloat16"}, "paddle")
@to_ivy_arrays_and_back
def fmax(x, y, name=None):
    return ivy.fmax(x, y)


@with_unsupported_dtypes({"2.5.1 and below": "bfloat16"}, "paddle")
@to_ivy_arrays_and_back
def fmin(x, y, name=None):
    return ivy.fmin(x, y)


@with_supported_dtypes(
    {"2.5.1 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def frac(x, name=None):
    y = ivy.trunc(x)
    return ivy.subtract(x, y)


@with_supported_dtypes({"2.5.1 and below": ("int32", "int64")}, "paddle")
@to_ivy_arrays_and_back
def gcd(x, y, name=None):
    return ivy.gcd(x, y)


@with_supported_dtypes(
    {"2.5.1 and below": ("float16", "float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def heaviside(x, y, name=None):
    return ivy.heaviside(x, y)


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def inner(x, y, name=None):
    result = ivy.inner(x, y)
    if (x.shape == () and y.shape == (1,)) or (x.shape == (1,) and y.shape == ()):
        result = result.reshape((1,))
    elif x.shape == (1,) and y.shape == (1,):
        result = result.reshape((1,))
    return result


@with_supported_dtypes(
    {"2.5.1 and below": ("float16", "float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def isfinite(x, name=None):
    return ivy.isfinite(x)


@with_supported_dtypes(
    {"2.5.1 and below": ("float16", "float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def isinf(x, name=None):
    return ivy.isinf(x)


@with_supported_dtypes(
    {"2.5.1 and below": ("float16", "float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def isnan(x, name=None):
    return ivy.isnan(x)


@with_supported_dtypes(
    {"2.5.1 and below": ("float16", "float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def kron(x, y, name=None):
    return ivy.kron(x, y)


@with_supported_dtypes({"2.5.1 and below": ("int32", "int64")}, "paddle")
@to_ivy_arrays_and_back
def lcm(x, y, name=None):
    return ivy.lcm(x, y)


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def lerp(x, y, weight, name=None):
    return ivy.lerp(x, y, weight)


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def lgamma(x, name=None):
    return ivy.lgamma(x)


@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def log(x, name=None):
    return ivy.log(x)


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def log1p(x, name=None):
    return ivy.log1p(x)


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def log2(x, name=None):
    return ivy.log2(x)


@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def logit(x, eps=None, name=None):
    return ivy.logit(x, eps=eps)


@with_supported_dtypes(
    {"2.5.1 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def max(x, axis=None, keepdim=False, name=None):
    return ivy.max(x, axis=axis, keepdims=keepdim)


# maximum
@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def maximum(x, y, name=None):
    return ivy.maximum(x, y)


@with_supported_dtypes(
    {"2.5.1 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def min(x, axis=None, keepdim=False, name=None):
    return ivy.min(x, axis=axis, keepdims=keepdim)


@with_supported_dtypes(
    {"2.5.1 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def minimum(x, y, name=None):
    return ivy.minimum(x, y)


@with_supported_dtypes(
    {"2.5.1 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def mm(input, mat2, name=None):
    return ivy.matmul(input, mat2)


@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def multiply(x, y, name=None):
    return ivy.multiply(x, y)


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def nanmean(x, axis=None, keepdims=False):
    return ivy.nanmean(x, axis=axis, keepdims=keepdims)


@with_supported_dtypes(
    {"2.5.1 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def nansum(x, axis=None, dtype=None, name=None):
    return ivy.nansum(x, axis=axis, dtype=dtype)


@with_supported_dtypes(
    {"2.5.1 and below": ("float32", "float64", "int8", "int16", "int32", "int64")},
    "paddle",
)
@to_ivy_arrays_and_back
def neg(x, name=None):
    return ivy.negative(x)


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def outer(x, y, name=None):
    return ivy.outer(x, y)


@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def pow(x, y, name=None):
    return ivy.pow(x, y)


@with_supported_dtypes(
    {"2.5.1 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def prod(x, axis=None, keepdim=False, dtype=None, name=None):
    return ivy.prod(x, axis=axis, keepdims=keepdim, dtype=dtype)


@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def rad2deg(x, name=None):
    return ivy.rad2deg(x)


@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def reciprocal(x, name=None):
    return ivy.reciprocal(x)


@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def remainder(x, y, name=None):
    return ivy.remainder(x, y)


@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def round(x, name=None):
    return ivy.round(x)


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def rsqrt(x, name=None):
    return 1 / ivy.sqrt(x)


@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def sgn(x, name=None):
    return ivy.sign(x, np_variant=True)


@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def sign(x, name=None):
    return ivy.sign(x, np_variant=False)


@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def sin(x, name=None):
    return ivy.sin(x)


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def sinh(x, name=None):
    return ivy.sinh(x)


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def sqrt(x, name=None):
    return ivy.sqrt(x)


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def square(x, name=None):
    return ivy.square(x)


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def stanh(x, scale_a=0.67, scale_b=1.7159, name=None):
    # TODO this function will be simplified as soon as the ivy.stanh(x,a,b) is added
    exp_ax = ivy.exp(ivy.multiply(scale_a, x))
    exp_minus_ax = ivy.exp(ivy.multiply(-scale_a, x))
    numerator = ivy.subtract(exp_ax, exp_minus_ax)
    denominator = ivy.add(exp_ax, exp_minus_ax)
    ret = ivy.multiply(scale_b, ivy.divide(numerator, denominator))
    return ret


@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def subtract(x, y, name=None):
    return ivy.subtract(x, y)


@with_supported_dtypes(
    {"2.5.1 and below": ("float32", "float64", "int32", "int6")}, "paddle"
)
@to_ivy_arrays_and_back
def take(
    x,
    index,
    mode="raise",
    name=None,
):
    if mode not in ["raise", "wrap", "clip"]:
        raise ValueError(
            "'mode' in 'take' should be 'raise', 'wrap', 'clip', but received {}."
            .format(mode)
        )
    x = ivy.reshape(x, (-1,))
    if mode == "clip":
        index = ivy.clip(index, 0, x.shape[-1] - 1)
    elif mode == "wrap":
        index = ivy.where(index < 0, index % x.shape[-1], index)
        index = ivy.where(index >= x.shape[-1], index % x.shape[-1], index)
    return ivy.gather(x, index, axis=0)


@with_unsupported_dtypes({"2.5.1 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def tan(x, name=None):
    return ivy.tan(x)


@with_supported_dtypes({"2.5.1 and below": ("float32", "float64")}, "paddle")
@to_ivy_arrays_and_back
def tanh(x, name=None):
    return ivy.tanh(x)


@with_supported_dtypes(
    {"2.4.2 and below": ("float32", "float64", "int32", "int64")}, "paddle"
)
@to_ivy_arrays_and_back
def trunc(x, name=None):
    return ivy.trunc(x)
