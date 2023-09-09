import numpy as np
import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import with_unsupported_dtypes
import numbers
from _array_api import get_namespace, _asarray_with_order
from contextlib import suppress
import warnings
import scipy.sparse as sp
from _isfinite import cy_isfinite, FiniteStatus
from .._config import get_config as _get_config
from fixes import _object_dtype_isnan
from numpy.core.numeric import ComplexWarning


# --- Helpers --- #
# --------------- #


@to_ivy_arrays_and_back
def _assert_all_finite(
    X, allow_nan=False, msg_dtype=None, estimator_name=None, input_name=""
):
    """Like assert_all_finite, but only for ndarray."""

    xp, _ = get_namespace(X)

    if _get_config()["assume_finite"]:
        return

    X = xp.asarray(X)

    # for object dtype data, we only check for NaNs (GH-13254)
    if X.dtype == ivy.dtype("object") and not allow_nan:
        if _object_dtype_isnan(X).any():
            raise ValueError("Input contains NaN")

    # We need only consider float arrays, hence can early return for all else.
    if X.dtype.kind not in "fc":
        return

    # First try an O(n) time, O(1) space solution for the common case that
    # everything is finite; fall back to O(n) space `ivy.isinf/isnan` or custom
    # Cython implementation to prevent false positives and provide a detailed
    # error message.
    with np.errstate(over="ignore"):
        first_pass_isfinite = xp.isfinite(xp.sum(X))
    if first_pass_isfinite:
        return
    # Cython implementation doesn't support FP16 or complex numbers
    use_cython = (
        xp is np and X.data.contiguous and X.dtype.type in {ivy.float32, ivy.float64}
    )
    if use_cython:
        out = cy_isfinite(X.reshape(-1), allow_nan=allow_nan)
        has_nan_error = False if allow_nan else out == FiniteStatus.has_nan
        has_inf = out == FiniteStatus.has_infinite
    else:
        has_inf = ivy.isinf(X).any()
        has_nan_error = False if allow_nan else xp.isnan(X).any()
    if has_inf or has_nan_error:
        if has_nan_error:
            type_err = "NaN"
        else:
            msg_dtype = msg_dtype if msg_dtype is not None else X.dtype
            type_err = f"infinity or a value too large for {msg_dtype!r}"
        padded_input_name = input_name + " " if input_name else ""
        msg_err = f"Input {padded_input_name}contains {type_err}."
        if estimator_name and input_name == "X" and has_nan_error:
            # Improve the error message on how to handle missing values in
            # scikit-learn.
            msg_err += (
                f"\n{estimator_name} does not accept missing values"
                " encoded as NaN natively. For supervised learning, you might want"
                " to consider sklearn.ensemble.HistGradientBoostingClassifier and"
                " Regressor which accept missing values encoded as NaNs natively."
                " Alternatively, it is possible to preprocess the data, for"
                " instance by using an imputer transformer in a pipeline or drop"
                " samples with missing values. See"
                " https://scikit-learn.org/stable/modules/impute.html"
                " You can find a list of all estimators that handle NaN values"
                " at the following page:"
                " https://scikit-learn.org/stable/modules/impute.html"
                "#estimators-that-handle-nan-values"
            )
        raise ValueError(msg_err)


@to_ivy_arrays_and_back
def _check_estimator_name(estimator):
    if estimator is not None:
        if isinstance(estimator, str):
            return estimator
        else:
            return estimator.__class__.__name__
    return None


@to_ivy_arrays_and_back
def _check_large_sparse(X, accept_large_sparse=False):
    """Raise a ValueError if X has 64bit indices and accept_large_sparse=False."""
    if not accept_large_sparse:
        supported_indices = ["int32"]
        if X.getformat() == "coo":
            index_keys = ["col", "row"]
        elif X.getformat() in ["csr", "csc", "bsr"]:
            index_keys = ["indices", "indptr"]
        else:
            return
        for key in index_keys:
            indices_datatype = getattr(X, key).dtype
            if indices_datatype not in supported_indices:
                raise ValueError(
                    "Only sparse matrices with 32-bit integer"
                    " indices are accepted. Got %s indices." % indices_datatype
                )


@to_ivy_arrays_and_back
def _check_sample_weight(
    sample_weight, X, dtype=None, copy=False, only_non_negative=False
):
    """
    Validate sample weights.

    Note that passing sample_weight=None will output an array of ones.
    Therefore, in some cases, you may want to protect the call with:
    if sample_weight is not None:
        sample_weight = _check_sample_weight(...)

    Parameters
    ----------
    sample_weight : {ndarray, Number or None}, shape (n_samples,)
        Input sample weights.

    X : {ndarray, list, sparse matrix}
        Input data.

    only_non_negative : bool, default=False,
        Whether or not the weights are expected to be non-negative.

        .. versionadded:: 1.0

    dtype : dtype, default=None
        dtype of the validated `sample_weight`.
        If None, and the input `sample_weight` is an array, the dtype of the
        input is preserved; otherwise an array with the default numpy dtype
        is be allocated.  If `dtype` is not one of `float32`, `float64`,
        `None`, the output will be of dtype `float64`.

    copy : bool, default=False
        If True, a copy of sample_weight will be created.

    Returns
    -------
    sample_weight : ndarray of shape (n_samples,)
        Validated sample weight. It is guaranteed to be "C" contiguous.
    """
    n_samples = _num_samples(X)

    if dtype is not None and dtype not in [ivy.float32, ivy.float64]:
        dtype = ivy.float64

    if sample_weight is None:
        sample_weight = ivy.ones(n_samples, dtype=dtype)
    elif isinstance(sample_weight, numbers.Number):
        sample_weight = ivy.full(n_samples, sample_weight, dtype=dtype)
    else:
        if dtype is None:
            dtype = [ivy.float64, ivy.float32]
        sample_weight = check_array(
            sample_weight,
            accept_sparse=False,
            ensure_2d=False,
            dtype=dtype,
            order="C",
            copy=copy,
            input_name="sample_weight",
        )
        if sample_weight.ndim != 1:
            raise ValueError("Sample weights must be 1D array or scalar")

        if sample_weight.shape != (n_samples,):
            raise ValueError(
                "sample_weight.shape == {}, expected {}!".format(
                    sample_weight.shape, (n_samples,)
                )
            )

    if only_non_negative:
        check_non_negative(sample_weight, "`sample_weight`")

    return sample_weight


@to_ivy_arrays_and_back
def _ensure_no_complex_data(array):
    if (
        hasattr(array, "dtype")
        and array.dtype is not None
        and hasattr(array.dtype, "kind")
        and array.dtype.kind == "c"
    ):
        raise ValueError("Complex data not supported\n{}\n".format(array))


@to_ivy_arrays_and_back
def _ensure_sparse_format(
    spmatrix,
    accept_sparse,
    dtype,
    copy,
    force_all_finite,
    accept_large_sparse,
    estimator_name=None,
    input_name="",
):
    """
    Convert a sparse matrix to a given format.

    Checks the sparse format of spmatrix and converts if necessary.

    Parameters
    ----------
    spmatrix : sparse matrix
        Input to validate and convert.

    accept_sparse : str, bool or list/tuple of str
        String[s] representing allowed sparse matrix formats ('csc',
        'csr', 'coo', 'dok', 'bsr', 'lil', 'dia'). If the input is sparse but
        not in the allowed format, it will be converted to the first listed
        format. True allows the input to be any format. False means
        that a sparse matrix input will raise an error.

    dtype : str, type or None
        Data type of result. If None, the dtype of the input is preserved.

    copy : bool
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_all_finite : bool or 'allow-nan'
        Whether to raise an error on ivy.inf, ivy.nan, pd.NA in X. The
        possibilities are:

        - True: Force all values of X to be finite.
        - False: accepts ivy.inf, ivy.nan, pd.NA in X.
        - 'allow-nan': accepts only ivy.nan and pd.NA values in X. Values cannot
          be infinite.

        .. versionadded:: 0.20
           ``force_all_finite`` accepts the string ``'allow-nan'``.

        .. versionchanged:: 0.23
           Accepts `pd.NA` and converts it into `ivy.nan`


    estimator_name : str, default=None
        The estimator name, used to construct the error message.

    input_name : str, default=""
        The data name used to construct the error message. In particular
        if `input_name` is "X" and the data has NaN values and
        allow_nan is False, the error message will link to the imputer
        documentation.

    Returns
    -------
    spmatrix_converted : sparse matrix.
        Matrix that is ensured to have an allowed type.
    """
    if dtype is None:
        dtype = spmatrix.dtype

    changed_format = False

    if isinstance(accept_sparse, str):
        accept_sparse = [accept_sparse]

    # Indices dtype validation
    _check_large_sparse(spmatrix, accept_large_sparse)

    if accept_sparse is False:
        raise TypeError(
            "A sparse matrix was passed, but dense "
            "data is required. Use X.toarray() to "
            "convert to a dense numpy array."
        )
    elif isinstance(accept_sparse, (list, tuple)):
        if len(accept_sparse) == 0:
            raise ValueError(
                "When providing 'accept_sparse' "
                "as a tuple or list, it must contain at "
                "least one string value."
            )
        # ensure correct sparse format
        if spmatrix.format not in accept_sparse:
            # create new with correct sparse
            spmatrix = spmatrix.asformat(accept_sparse[0])
            changed_format = True
    elif accept_sparse is not True:
        # any other type
        raise ValueError(
            "Parameter 'accept_sparse' should be a string, "
            "boolean or list of strings. You provided "
            "'accept_sparse={}'.".format(accept_sparse)
        )

    if dtype != spmatrix.dtype:
        # convert dtype
        spmatrix = spmatrix.astype(dtype)
    elif copy and not changed_format:
        # force copy
        spmatrix = spmatrix.copy()

    if force_all_finite:
        if not hasattr(spmatrix, "data"):
            warnings.warn(
                "Can't check %s sparse matrix for nan or inf." % spmatrix.format,
                stacklevel=2,
            )
        else:
            _assert_all_finite(
                spmatrix.data,
                allow_nan=force_all_finite == "allow-nan",
                estimator_name=estimator_name,
                input_name=input_name,
            )

    return spmatrix


@to_ivy_arrays_and_back
def _num_samples(x):
    """Return number of samples in array-like x."""
    message = "Expected sequence or array-like, got %s" % type(x)
    if hasattr(x, "fit") and callable(x.fit):
        # Don't get num_samples from an ensembles length!
        raise TypeError(message)

    if not hasattr(x, "__len__") and not hasattr(x, "shape"):
        if hasattr(x, "__array__"):
            x = ivy.asarray(x)
        else:
            raise TypeError(message)

    if hasattr(x, "shape") and x.shape is not None:
        if len(x.shape) == 0:
            raise TypeError(
                "Singleton array %r cannot be considered a valid collection." % x
            )
        # Check that shape is returning an integer or default to len
        # Dask dataframes may not return numeric shape[0] value
        if isinstance(x.shape[0], numbers.Integral):
            return x.shape[0]

    try:
        return len(x)
    except TypeError as type_error:
        raise TypeError(message) from type_error


@to_ivy_arrays_and_back
def _pandas_dtype_needs_early_conversion(pd_dtype):
    """Return True if pandas extension pd_dtype need to be converted early."""
    # Check these early for pandas versions without extension dtypes
    from pandas.api.types import (
        is_bool_dtype,
        is_sparse,
        is_float_dtype,
        is_integer_dtype,
    )

    if is_bool_dtype(pd_dtype):
        # bool and extension booleans need early converstion because __array__
        # converts mixed dtype dataframes into object dtypes
        return True

    if is_sparse(pd_dtype):
        # Sparse arrays will be converted later in `check_array`
        return False

    try:
        from pandas.api.types import is_extension_array_dtype
    except ImportError:
        return False

    if is_sparse(pd_dtype) or not is_extension_array_dtype(pd_dtype):
        # Sparse arrays will be converted later in `check_array`
        # Only handle extension arrays for integer and floats
        return False
    elif is_float_dtype(pd_dtype):
        # Float ndarrays can normally support nans. They need to be converted
        # first to map pd.NA to ivy.nan
        return True
    elif is_integer_dtype(pd_dtype):
        # XXX: Warn when converting from a high integer to a float
        return True

    return False


# --- Main --- #
# ------------ #


@to_ivy_arrays_and_back
def as_float_array(X, *, copy=True, force_all_finite=True):
    if X.dtype in [ivy.float32, ivy.float64]:
        return X.copy_array() if copy else X
    if ("bool" in X.dtype or "int" in X.dtype or "uint" in X.dtype) and ivy.itemsize(
        X
    ) <= 4:
        return_dtype = ivy.float32
    else:
        return_dtype = ivy.float64
    return ivy.asarray(X, dtype=return_dtype)


@to_ivy_arrays_and_back
def assert_all_finite(
    X,
    *,
    allow_nan=False,
    estimator_name=None,
    input_name="",
):
    """
    Throw a ValueError if X contains NaN or infinity.

    Parameters
    ----------
    X : {ndarray, sparse matrix}
        The input data.

    allow_nan : bool, default=False
        If True, do not throw error when `X` contains NaN.

    estimator_name : str, default=None
        The estimator name, used to construct the error message.

    input_name : str, default=""
        The data name used to construct the error message. In particular
        if `input_name` is "X" and the data has NaN values and
        allow_nan is False, the error message will link to the imputer
        documentation.
    """
    _assert_all_finite(
        X.data if sp.issparse(X) else X,
        allow_nan=allow_nan,
        estimator_name=estimator_name,
        input_name=input_name,
    )


@to_ivy_arrays_and_back
def check_array(
    array,
    accept_sparse=False,
    *,
    accept_large_sparse=True,
    dtype="numeric",
    order=None,
    copy=False,
    force_all_finite=True,
    ensure_2d=True,
    allow_nd=False,
    ensure_min_samples=1,
    ensure_min_features=1,
    estimator=None,
    input_name="",
):
    """
    Input validation on an array, list, sparse matrix or similar.

    By default, the input is checked to be a non-empty 2D array containing
    only finite values. If the dtype of the array is object, attempt
    converting to float, raising on failure.

    Parameters
    ----------
    array : object
        Input object to check / convert.

    accept_sparse : str, bool or list/tuple of str, default=False
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc. If the input is sparse but not in the allowed format,
        it will be converted to the first listed format. True allows the input
        to be any format. False means that a sparse matrix input will
        raise an error.

    accept_large_sparse : bool, default=True
        If a CSR, CSC, COO or BSR sparse matrix is supplied and accepted by
        accept_sparse, accept_large_sparse=False will cause it to be accepted
        only if its indices are stored with a 32-bit dtype.

        .. versionadded:: 0.20

    dtype : 'numeric', type, list of type or None, default='numeric'
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.

    order : {'F', 'C'} or None, default=None
        Whether an array will be forced to be fortran or c-style.
        When order is None (default), then if copy=False, nothing is ensured
        about the memory layout of the output array; otherwise (copy=True)
        the memory layout of the returned array is kept as close as possible
        to the original array.

    copy : bool, default=False
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_all_finite : bool or 'allow-nan', default=True
        Whether to raise an error on ivy.inf, ivy.nan, pd.NA in array. The
        possibilities are:

        - True: Force all values of array to be finite.
        - False: accepts ivy.inf, ivy.nan, pd.NA in array.
        - 'allow-nan': accepts only ivy.nan and pd.NA values in array. Values
          cannot be infinite.

        .. versionadded:: 0.20
           ``force_all_finite`` accepts the string ``'allow-nan'``.

        .. versionchanged:: 0.23
           Accepts `pd.NA` and converts it into `ivy.nan`

    ensure_2d : bool, default=True
        Whether to raise a value error if array is not 2D.

    allow_nd : bool, default=False
        Whether to allow array.ndim > 2.

    ensure_min_samples : int, default=1
        Make sure that the array has a minimum number of samples in its first
        axis (rows for a 2D array). Setting to 0 disables this check.

    ensure_min_features : int, default=1
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when the input data has effectively 2
        dimensions or is originally 1D and ``ensure_2d`` is True. Setting to 0
        disables this check.

    estimator : str or estimator instance, default=None
        If passed, include the name of the estimator in warning messages.

    input_name : str, default=""
        The data name used to construct the error message. In particular
        if `input_name` is "X" and the data has NaN values and
        allow_nan is False, the error message will link to the imputer
        documentation.

        .. versionadded:: 1.1.0

    Returns
    -------
    array_converted : object
        The converted and validated array.
    """
    xp, is_array_api = get_namespace(array)

    # store reference to original array to check if copy is needed when
    # function returns
    array_orig = array

    # store whether originally we wanted numeric dtype
    dtype_numeric = isinstance(dtype, str) and dtype == "numeric"

    dtype_orig = getattr(array, "dtype", None)
    if not hasattr(dtype_orig, "kind"):
        # not a data type (e.g. a column named dtype in a pandas DataFrame)
        dtype_orig = None

    # check if the object contains several dtypes (typically a pandas
    # DataFrame), and store them. If not, store None.
    dtypes_orig = None
    pandas_requires_conversion = False
    if hasattr(array, "dtypes") and hasattr(array.dtypes, "__array__"):
        # throw warning if columns are sparse. If all columns are sparse, then
        # array.sparse exists and sparsity will be preserved (later).
        with suppress(ImportError):
            from pandas.api.types import is_sparse

            if not hasattr(array, "sparse") and array.dtypes.apply(is_sparse).any():
                warnings.warn(
                    "pandas.DataFrame with sparse columns found."
                    "It will be converted to a dense numpy array."
                )

        dtypes_orig = list(array.dtypes)
        pandas_requires_conversion = any(
            _pandas_dtype_needs_early_conversion(i) for i in dtypes_orig
        )
        if all(isinstance(dtype_iter, ivy.dtype) for dtype_iter in dtypes_orig):
            dtype_orig = ivy.result_type(*dtypes_orig)

    elif hasattr(array, "iloc") and hasattr(array, "dtype"):
        # array is a pandas series
        pandas_requires_conversion = _pandas_dtype_needs_early_conversion(array.dtype)
        if isinstance(array.dtype, ivy.dtype):
            dtype_orig = array.dtype
        else:
            # Set to None to let array.astype work out the best dtype
            dtype_orig = None

    if dtype_numeric:
        if dtype_orig is not None and dtype_orig.kind == "O":
            # if input is object, convert to float.
            dtype = xp.float64
        else:
            dtype = None

    if isinstance(dtype, (list, tuple)):
        if dtype_orig is not None and dtype_orig in dtype:
            # no dtype conversion required
            dtype = None
        else:
            # dtype conversion required. Let's select the first element of the
            # list of accepted types.
            dtype = dtype[0]

    if pandas_requires_conversion:
        # pandas dataframe requires conversion earlier to handle extension dtypes with
        # nans
        # Use the original dtype for conversion if dtype is None
        new_dtype = dtype_orig if dtype is None else dtype
        array = array.astype(new_dtype)
        # Since we converted here, we do not need to convert again later
        dtype = None

    if force_all_finite not in (True, False, "allow-nan"):
        raise ValueError(
            'force_all_finite should be a bool or "allow-nan". Got {!r} instead'.format(
                force_all_finite
            )
        )

    estimator_name = _check_estimator_name(estimator)
    context = " by %s" % estimator_name if estimator is not None else ""

    # When all dataframe columns are sparse, convert to a sparse array
    if hasattr(array, "sparse") and array.ndim > 1:
        with suppress(ImportError):
            from pandas.api.types import is_sparse

            if array.dtypes.apply(is_sparse).all():
                # DataFrame.sparse only supports `to_coo`
                array = array.sparse.to_coo()
                if array.dtype == ivy.dtype("object"):
                    unique_dtypes = set([dt.subtype.name for dt in array_orig.dtypes])
                    if len(unique_dtypes) > 1:
                        raise ValueError(
                            "Pandas DataFrame with mixed sparse extension arrays "
                            "generated a sparse matrix with object dtype which "
                            "can not be converted to a scipy sparse matrix."
                            "Sparse extension arrays should all have the same "
                            "numeric type."
                        )

    if sp.issparse(array):
        _ensure_no_complex_data(array)
        array = _ensure_sparse_format(
            array,
            accept_sparse=accept_sparse,
            dtype=dtype,
            copy=copy,
            force_all_finite=force_all_finite,
            accept_large_sparse=accept_large_sparse,
            estimator_name=estimator_name,
            input_name=input_name,
        )
    else:
        # If ivy.array(..) gives ComplexWarning, then we convert the warning
        # to an error. This is needed because specifying a non complex
        # dtype to the function converts complex to real dtype,
        # thereby passing the test made in the lines following the scope
        # of warnings context manager.
        with warnings.catch_warnings():
            try:
                warnings.simplefilter("error", ComplexWarning)
                if dtype is not None and ivy.dtype(dtype).kind in "iu":
                    # Conversion float -> int should not contain NaN or
                    # inf (numpy#14412). We cannot use casting='safe' because
                    # then conversion float -> int would be disallowed.
                    array = _asarray_with_order(array, order=order, xp=xp)
                    if array.dtype.kind == "f":
                        _assert_all_finite(
                            array,
                            allow_nan=False,
                            msg_dtype=dtype,
                            estimator_name=estimator_name,
                            input_name=input_name,
                        )
                    array = xp.astype(array, dtype, copy=False)
                else:
                    array = _asarray_with_order(array, order=order, dtype=dtype, xp=xp)
            except ComplexWarning as complex_warning:
                raise ValueError(
                    "Complex data not supported\n{}\n".format(array)
                ) from complex_warning

        # It is possible that the ivy.array(..) gave no warning. This happens
        # when no dtype conversion happened, for example dtype = None. The
        # result is that ivy.array(..) produces an array of complex dtype
        # and we need to catch and raise exception for such cases.
        _ensure_no_complex_data(array)

        if ensure_2d:
            # If input is scalar raise error
            if array.ndim == 0:
                raise ValueError(
                    "Expected 2D array, got scalar array instead:\narray={}.\n"
                    "Reshape your data either using array.reshape(-1, 1) if "
                    "your data has a single feature or array.reshape(1, -1) "
                    "if it contains a single sample.".format(array)
                )
            # If input is 1D raise error
            if array.ndim == 1:
                raise ValueError(
                    "Expected 2D array, got 1D array instead:\narray={}.\n"
                    "Reshape your data either using array.reshape(-1, 1) if "
                    "your data has a single feature or array.reshape(1, -1) "
                    "if it contains a single sample.".format(array)
                )

        if dtype_numeric and array.dtype.kind in "USV":
            raise ValueError(
                "dtype='numeric' is not compatible with arrays of bytes/strings."
                "Convert your data to numeric values explicitly instead."
            )
        if not allow_nd and array.ndim >= 3:
            raise ValueError(
                "Found array with dim %d. %s expected <= 2."
                % (array.ndim, estimator_name)
            )

        if force_all_finite:
            _assert_all_finite(
                array,
                input_name=input_name,
                estimator_name=estimator_name,
                allow_nan=force_all_finite == "allow-nan",
            )

    if ensure_min_samples > 0:
        n_samples = _num_samples(array)
        if n_samples < ensure_min_samples:
            raise ValueError(
                "Found array with %d sample(s) (shape=%s) while a"
                " minimum of %d is required%s."
                % (n_samples, array.shape, ensure_min_samples, context)
            )

    if ensure_min_features > 0 and array.ndim == 2:
        n_features = array.shape[1]
        if n_features < ensure_min_features:
            raise ValueError(
                "Found array with %d feature(s) (shape=%s) while"
                " a minimum of %d is required%s."
                % (n_features, array.shape, ensure_min_features, context)
            )

    if copy:
        if xp.__name__ in {"numpy", "numpy.array_api"}:
            # only make a copy if `array` and `array_orig` may share memory`
            if np.may_share_memory(array, array_orig):
                array = _asarray_with_order(
                    array, dtype=dtype, order=order, copy=True, xp=xp
                )
        else:
            # always make a copy for non-numpy arrays
            array = _asarray_with_order(
                array, dtype=dtype, order=order, copy=True, xp=xp
            )

    return array


@to_ivy_arrays_and_back
def check_non_negative(X, whom):
    """
    Check if there is any negative value in an array.

    Parameters
    ----------
    X : {array-like, sparse matrix}
        Input data.

    whom : str
        Who passed X to this function.
    """
    xp, _ = get_namespace(X)
    # avoid X.min() on sparse matrix since it also sorts the indices
    if sp.issparse(X):
        if X.format in ["lil", "dok"]:
            X = X.tocsr()
        if X.data.size == 0:
            X_min = 0
        else:
            X_min = X.data.min()
    else:
        X_min = xp.min(X)

    if X_min < 0:
        raise ValueError("Negative values in data passed to %s" % whom)


@with_unsupported_dtypes({"1.3.0 and below": ("complex",)}, "sklearn")
@to_ivy_arrays_and_back
def column_or_1d(y, *, warn=False):
    shape = y.shape
    if len(shape) == 2 and shape[1] == 1:
        y = ivy.reshape(y, (-1,))
    elif len(shape) > 2:
        raise ValueError("y should be a 1d array or a column vector")
    return y
