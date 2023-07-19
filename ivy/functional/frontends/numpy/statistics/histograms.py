import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import with_supported_dtypes
import operator
import numpy as np

_range = range


# helper functions
def _ravel_and_check_weights(a, weights):
    """Check a and weights have matching shapes, and ravel both."""
    a = ivy.asarray(a)

    # Ensure that the array is a "subtractable" dtype
    if ivy.is_bool_dtype(a):
        ivy.logging.warning(
            "Converting input from {} to {} for compatibility.".format(
                a.dtype, ivy.uint8
            ),
            stacklevel=3,
        )
        a = a.astype(ivy.uint8)

    if weights is not None:
        weights = ivy.asarray(weights)
        if weights.shape != a.shape:
            raise ivy.utils.exceptions.IvyValueError(
                "weights should have the same shape as a."
            )
        weights = weights.reshape((-1,))
    a = a.reshape((-1,))
    return a, weights


def _get_outer_edges(a, range):
    if range is not None:
        first_edge, last_edge = range
        if first_edge > last_edge:
            raise ivy.utils.exceptions.IvyValueError(
                "max must be larger than min in range parameter."
            )
    elif a.size == 0:
        # handle empty arrays. Can't determine range, so use 0-1.
        first_edge, last_edge = 0, 1
    else:
        first_edge, last_edge = a.min(), a.max()

    # expand empty range to avoid divide by zero
    if first_edge == last_edge:
        first_edge = first_edge - 0.5
        last_edge = last_edge + 0.5

    return first_edge, last_edge


def _get_bin_edges(a, bins, range, weights):
    n_equal_bins = None
    bin_edges = None

    if isinstance(bins, str):
        bin_name = bins
        a = ivy.array(a)
        # if `bins` is a string for an automatic method,
        # this will replace it with the number of bins calculated
        if bin_name not in _hist_bin_selectors:
            raise ivy.utils.exceptions.IvyValueError(
                "{!r} is not a valid estimator for `bins`".format(bin_name)
            )
        if weights is not None:
            raise TypeError(
                "Automated estimation of the number of "
                "bins is not supported for weighted data"
            )

        first_edge, last_edge = _get_outer_edges(a, range)

        # truncate the range if needed
        if range is not None:
            keep = a >= first_edge
            keep &= a <= last_edge
            if not ivy.reduce(keep, True, ivy.logical_and):
                a = a[keep]

        if a.size == 0:
            n_equal_bins = 1
        else:
            # Do not call selectors on empty arrays
            width = _hist_bin_selectors[bin_name](a, (first_edge, last_edge))
            if width:
                n_equal_bins = int(ivy.ceil((last_edge - first_edge) / width))
            else:
                # Width can be zero for some estimators, e.g. FD when
                # the IQR of the data is zero.
                n_equal_bins = 1

    elif ivy.asarray(bins).ndim == 0:
        try:
            n_equal_bins = operator.index(bins)
        except TypeError as e:
            raise TypeError("`bins` must be an integer, a string, or an array") from e
        if n_equal_bins < 1:
            raise ivy.utils.exceptions.IvyValueError(
                "`bins` must be positive, when an integer"
            )

        first_edge, last_edge = _get_outer_edges(a, range)

    elif ivy.asarray(bins).ndim == 1:
        bin_edges = ivy.asarray(bins)
        if ivy.any(bin_edges[:-1] > bin_edges[1:]):
            raise ivy.utils.exceptions.IvyValueError(
                "`bins` must increase monotonically, when an array"
            )

    else:
        raise ivy.utils.exceptions.IvyValueError("`bins` must be 1d, when an array")

    if n_equal_bins is not None:
        bin_type = ivy.result_type(ivy.array([first_edge, last_edge]), a)
        if ivy.is_int_dtype(bin_type):
            bin_type = ivy.result_type(bin_type, float)

        # bin edges must be computed
        bin_edges = ivy.linspace(
            first_edge, last_edge, n_equal_bins + 1, endpoint=True, dtype=bin_type
        )
        return bin_edges, (first_edge, last_edge, n_equal_bins)
    else:
        return bin_edges, None


def _ptp(x):
    return x.max() - x.min()


def _hist_bin_sqrt(x, range):
    del range  # unused
    return _ptp(x) / ivy.sqrt(x.size)


def _hist_bin_sturges(x, range):
    del range  # unused
    return _ptp(x) / (ivy.log2(float(x.size)) + 1.0)


def _hist_bin_rice(x, range):
    del range  # unused
    return _ptp(x) / (2.0 * x.size ** (1.0 / 3))


def _hist_bin_scott(x, range):
    del range  # unused
    return (24.0 * ivy.pi**0.5 / x.size) ** (1.0 / 3.0) * ivy.std(x)


def _hist_bin_stone(x, range):
    n = x.size
    ptp_x = _ptp(x)
    if n <= 1 or ptp_x == 0:
        return 0

    def jhat(nbins):
        hh = ptp_x / nbins
        p_k = ivy.array(np.histogram(np.array(x), bins=nbins, range=range)[0] / n)
        return (2 - (n + 1) * p_k.dot(p_k)) / hh

    nbins_upper_bound = max(100, int(ivy.sqrt(n)))
    nbins = min(_range(1, nbins_upper_bound + 1), key=jhat)
    if nbins == nbins_upper_bound:
        ivy.logging.warning(
            "The number of bins estimated may be suboptimal.", stacklevel=3
        )
    return ptp_x / nbins


def _hist_bin_doane(x, range):
    del range  # unused
    if x.size > 2:
        sg1 = (6.0 * (x.size - 2) / ((x.size + 1.0) * (x.size + 3))) ** 0.5
        sigma = ivy.std(ivy.array(x, dtype=ivy.float64))
        if sigma > 0.0:
            temp = x - ivy.mean(ivy.array(x, dtype=ivy.float64))
            ivy.divide(ivy.array(temp, dtype=ivy.float64), sigma, out=temp)
            ivy.pow(ivy.array(temp, dtype=ivy.float64), 3, out=temp)
            g1 = ivy.mean(ivy.array(temp, dtype=ivy.float64))
            return _ptp(x) / (
                1.0 + ivy.log2(float(x.size)) + ivy.log2(1.0 + ivy.abs(g1) / sg1)
            )
    return 0.0


def _hist_bin_fd(x, range):
    del range  # unused
    iqr = ivy.subtract(*np.percentile(x, [75, 25]))
    return 2.0 * iqr * x.size ** (-1.0 / 3.0)


def _hist_bin_auto(x, range):
    fd_bw = _hist_bin_fd(x, range)
    sturges_bw = _hist_bin_sturges(x, range)
    del range  # unused
    if fd_bw:
        return min(fd_bw, sturges_bw)
    else:
        # limited variance, so we return a len dependent bw estimator
        return sturges_bw


_hist_bin_selectors = {
    "stone": _hist_bin_stone,
    "auto": _hist_bin_auto,
    "doane": _hist_bin_doane,
    "fd": _hist_bin_fd,
    "rice": _hist_bin_rice,
    "scott": _hist_bin_scott,
    "sqrt": _hist_bin_sqrt,
    "sturges": _hist_bin_sturges,
}


@with_supported_dtypes({"1.25.1 and below": ("int64",)}, "numpy")
@to_ivy_arrays_and_back
def bincount(x, /, weights=None, minlength=0):
    return ivy.bincount(x, weights=weights, minlength=minlength)


@to_ivy_arrays_and_back
def histogram_bin_edges(a, bins=10, range=None, weights=None):
    a, weights = _ravel_and_check_weights(a, weights)
    bin_edges, _ = _get_bin_edges(a, bins, range, weights)
    if ivy.is_int_dtype(bin_edges) and ivy.is_int_dtype(a):
        return ivy.array(bin_edges, dtype=a.dtype)
    if ivy.is_float_dtype(bin_edges) and ivy.is_float_dtype(a):
        return ivy.array(bin_edges, dtype=a.dtype)
    if ivy.is_complex_dtype(bin_edges) and ivy.is_complex_dtype(a):
        return ivy.array(bin_edges, dtype=a.dtype)
    return ivy.array(bin_edges)
