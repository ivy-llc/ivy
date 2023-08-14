import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back
import ivy.functional.frontends.numpy as np_frontend
from ivy.func_wrapper import with_supported_dtypes
import operator

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
            width = float(_hist_bin_selectors[bin_name](a, (first_edge, last_edge)))
            if width:
                n_equal_bins = int(
                    ivy.ceil(
                        ivy.array(
                            float(last_edge - first_edge) / width, dtype=ivy.float64
                        )
                    )
                )
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
        bin_type = ivy.result_type(a, first_edge, last_edge)
        if ivy.is_int_dtype(bin_type):
            bin_type = ivy.result_type(bin_type, float)

        # bin edges must be computed
        bin_edges = np_frontend.linspace(
            first_edge, last_edge, n_equal_bins + 1, endpoint=True, dtype=bin_type
        ).ivy_array
        return bin_edges, (first_edge, last_edge, n_equal_bins)
    else:
        return bin_edges, None


def _ptp(x):
    return x.max() - x.min()


def _hist_bin_sqrt(x, range):
    del range  # unused
    return _ptp(x) / ivy.sqrt(ivy.array(x.size, dtype=ivy.float64))


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
    ptp_x = float(_ptp(x))
    if n <= 1 or ptp_x == 0:
        return 0

    def jhat(nbins):
        hh = ptp_x / nbins
        p_k = (histogram(x, bins=nbins, range=range)[0] / n).ivy_array
        return (2 - (n + 1) * ivy.vecdot(p_k, p_k)) / hh

    nbins_upper_bound = max(100, int(ivy.sqrt(ivy.array(n, dtype=ivy.float64))))
    nbins = min(_range(1, nbins_upper_bound + 1), key=jhat)
    if nbins == nbins_upper_bound:
        ivy.logging.warning(
            "The number of bins estimated may be suboptimal.", stacklevel=3
        )
    return float(ptp_x / nbins)


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
    iqr = ivy.subtract(*np_frontend.nanpercentile(x, q=[75, 25]))
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


def _search_sorted_inclusive(a, v):
    """
    Like `searchsorted`, but where the last item in `v` is placed on the right.

    In the context of a histogram, this makes the last bin edge
    inclusive
    """
    return ivy.concat(
        (
            ivy.searchsorted(a, v[:-1], side="left"),
            ivy.searchsorted(a, v[-1:], side="right"),
        )
    )


@with_supported_dtypes({"1.25.2 and below": ("int64",)}, "numpy")
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


@to_ivy_arrays_and_back
def histogram(a, bins=10, range=None, density=None, weights=None):
    a, weights = _ravel_and_check_weights(a, weights)

    bin_edges, uniform_bins = _get_bin_edges(a, bins, range, weights)

    # Histogram is an integer or a float array depending on the weights.
    if weights is None:
        ntype = ivy.int64
    else:
        ntype = weights.dtype

    # We set a block size, as this allows us to iterate over chunks when
    # computing histograms, to minimize memory usage.
    BLOCK = 65536

    # The fast path uses bincount, but that only works for certain types
    # of weight
    simple_weights = (
        weights is None
        or ivy.can_cast(weights.dtype, ivy.double)
        or ivy.can_cast(weights.dtype, ivy.complex128)
    )

    if uniform_bins is not None and simple_weights:
        # Fast algorithm for equal bins
        # We now convert values of a to bin indices, under the assumption of
        # equal bin widths (which is valid here).
        first_edge, last_edge, n_equal_bins = uniform_bins

        # Initialize empty histogram
        n = ivy.zeros(n_equal_bins, dtype=ntype)

        # Pre-compute histogram scaling factor
        norm = n_equal_bins / float(ivy.subtract(last_edge, first_edge))

        # We iterate over blocks here for two reasons: the first is that for
        # large arrays, it is actually faster (for example for a 10^8 array it
        # is 2x as fast) and it results in a memory footprint 3x lower in the
        # limit of large arrays.
        for i in _range(0, len(a), BLOCK):
            tmp_a = a[i : i + BLOCK]
            if weights is None:
                tmp_w = None
            else:
                tmp_w = weights[i : i + BLOCK]

            # Only include values in the right range
            keep = tmp_a >= first_edge
            keep &= tmp_a <= last_edge
            if not ivy.reduce(keep, True, ivy.logical_and):
                tmp_a = tmp_a[keep]
                if tmp_w is not None:
                    tmp_w = tmp_w[keep]

            # This cast ensures no type promotions occur below, which gh-10322
            # make unpredictable. Getting it wrong leads to precision errors
            # like gh-8123.
            tmp_a = ivy.astype(tmp_a, bin_edges.dtype)

            # Compute the bin indices, and for values that lie exactly on
            # last_edge we need to subtract one
            f_indices = ivy.subtract(tmp_a, first_edge) * norm
            indices = ivy.astype(f_indices, ivy.int64)
            indices[indices == n_equal_bins] -= 1

            # The index computation is not guaranteed to give exactly
            # consistent results within ~1 ULP of the bin edges.
            decrement = tmp_a < bin_edges[indices]
            indices[decrement] -= 1
            # The last bin includes the right edge. The other bins do not.
            increment = (tmp_a >= bin_edges[indices + 1]) & (
                indices != n_equal_bins - 1
            )
            indices[increment] += 1

            # We now compute the histogram using bincount
            if ivy.is_complex_dtype(ntype):
                real = bincount(
                    indices, weights=tmp_w.real(), minlength=n_equal_bins
                ).ivy_array
                imag = bincount(
                    indices, weights=tmp_w.imag(), minlength=n_equal_bins
                ).ivy_array
                n += real + ivy.array(imag, dtype=ivy.complex128) * 1j

            else:
                n += ivy.astype(
                    bincount(indices, weights=tmp_w, minlength=n_equal_bins).ivy_array,
                    ntype,
                )
    else:
        # Compute via cumulative histogram
        cum_n = ivy.zeros(bin_edges.shape, dtype=ntype)
        if weights is None:
            for i in _range(0, len(a), BLOCK):
                sa = ivy.sort(a[i : i + BLOCK])
                cum_n += _search_sorted_inclusive(sa, bin_edges)
        else:
            zero = ivy.zeros(1, dtype=ntype)
            for i in _range(0, len(a), BLOCK):
                tmp_a = a[i : i + BLOCK]
                tmp_w = weights[i : i + BLOCK]
                sorting_index = ivy.argsort(tmp_a)
                sa = tmp_a[sorting_index]
                sw = tmp_w[sorting_index]
                cw = ivy.concat((zero, ivy.cumsum(sw)))
                bin_index = _search_sorted_inclusive(sa, bin_edges)
                cum_n += cw[bin_index]

        n = ivy.diff(cum_n)

    if density:
        db = ivy.array(ivy.diff(bin_edges), dtype=ivy.float64)
        return n / db / n.sum(), bin_edges

    return n, bin_edges
