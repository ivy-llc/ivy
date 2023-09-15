from .utils.validation import (
    check_array,
    _check_y,
    check_X_y,
    _get_feature_names,
    _num_features,
)
import warnings
import ivy
import inspect
from .utils._tags import (
    _DEFAULT_TAGS,
)


class BaseEstimator:
    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self

    def _check_feature_names(self, X, *, reset):
        """
        Set or check the `feature_names_in_` attribute.

        .. versionadded:: 1.0

        Parameters
        ----------
        X : {ndarray, dataframe} of shape (n_samples, n_features)
            The input samples.

        reset : bool
            Whether to reset the `feature_names_in_` attribute.
            If False, the input will be checked for consistency with
            feature names of data provided when reset was last True.
            .. note::
               It is recommended to call `reset=True` in `fit` and in the first
               call to `partial_fit`. All other methods that validate `X`
               should set `reset=False`.
        """

        if reset:
            feature_names_in = _get_feature_names(X)
            if feature_names_in is not None:
                self.feature_names_in_ = feature_names_in
            elif hasattr(self, "feature_names_in_"):
                # Delete the attribute when the estimator is fitted on a new dataset
                # that has no feature names.
                delattr(self, "feature_names_in_")
            return

        fitted_feature_names = getattr(self, "feature_names_in_", None)
        X_feature_names = _get_feature_names(X)

        if fitted_feature_names is None and X_feature_names is None:
            # no feature names seen in fit and in X
            return

        if X_feature_names is not None and fitted_feature_names is None:
            warnings.warn(
                f"X has feature names, but {self.__class__.__name__} was fitted without"
                " feature names"
            )
            return

        if X_feature_names is None and fitted_feature_names is not None:
            warnings.warn(
                "X does not have valid feature names, but"
                f" {self.__class__.__name__} was fitted with feature names"
            )
            return

        # validate the feature names against the `feature_names_in_` attribute
        if len(fitted_feature_names) != len(X_feature_names) or ivy.any(
            fitted_feature_names != X_feature_names
        ):
            message = (
                "The feature names should match those that were passed during fit.\n"
            )
            fitted_feature_names_set = set(fitted_feature_names)
            X_feature_names_set = set(X_feature_names)

            unexpected_names = sorted(X_feature_names_set - fitted_feature_names_set)
            missing_names = sorted(fitted_feature_names_set - X_feature_names_set)

            def add_names(names):
                output = ""
                max_n_names = 5
                for i, name in enumerate(names):
                    if i >= max_n_names:
                        output += "- ...\n"
                        break
                    output += f"- {name}\n"
                return output

            if unexpected_names:
                message += "Feature names unseen at fit time:\n"
                message += add_names(unexpected_names)

            if missing_names:
                message += "Feature names seen at fit time, yet now missing:\n"
                message += add_names(missing_names)

            if not missing_names and not unexpected_names:
                message += (
                    "Feature names must be in the same order as they were in fit.\n"
                )

            raise ValueError(message)

    def _validate_data(
        self,
        X="no_validation",
        y="no_validation",
        reset=True,
        validate_separately=False,
        **check_params,
    ):
        """
        Validate input data and set or check the `n_features_in_` attribute.

        Parameters
        ----------
        X : {array-like, sparse matrix, dataframe} of shape \
                (n_samples, n_features), default='no validation'
            The input samples.
            If `'no_validation'`, no validation is performed on `X`. This is
            useful for meta-estimator which can delegate input validation to
            their underlying estimator(s). In that case `y` must be passed and
            the only accepted `check_params` are `multi_output` and
            `y_numeric`.

        y : array-like of shape (n_samples,), default='no_validation'
            The targets.

            - If `None`, `check_array` is called on `X`. If the estimator's
              requires_y tag is True, then an error will be raised.
            - If `'no_validation'`, `check_array` is called on `X` and the
              estimator's requires_y tag is ignored. This is a default
              placeholder and is never meant to be explicitly set. In that case
              `X` must be passed.
            - Otherwise, only `y` with `_check_y` or both `X` and `y` are
              checked with either `check_array` or `check_X_y` depending on
              `validate_separately`.

        reset : bool, default=True
            Whether to reset the `n_features_in_` attribute.
            If False, the input will be checked for consistency with data
            provided when reset was last True.
            .. note::
               It is recommended to call reset=True in `fit` and in the first
               call to `partial_fit`. All other methods that validate `X`
               should set `reset=False`.

        validate_separately : False or tuple of dicts, default=False
            Only used if y is not None.
            If False, call validate_X_y(). Else, it must be a tuple of kwargs
            to be used for calling check_array() on X and y respectively.

            `estimator=self` is automatically added to these dicts to generate
            more informative error message in case of invalid input data.

        **check_params : kwargs
            Parameters passed to :func:`sklearn.utils.check_array` or
            :func:`sklearn.utils.check_X_y`. Ignored if validate_separately
            is not False.

            `estimator=self` is automatically added to these params to generate
            more informative error message in case of invalid input data.

        Returns
        -------
        out : {ndarray, sparse matrix} or tuple of these
            The validated input. A tuple is returned if both `X` and `y` are
            validated.
        """
        self._check_feature_names(X, reset=reset)

        if y is None and self._get_tags()["requires_y"]:
            raise ValueError(
                f"This {self.__class__.__name__} estimator "
                "requires y to be passed, but the target y is None."
            )

        no_val_X = isinstance(X, str) and X == "no_validation"
        no_val_y = y is None or isinstance(y, str) and y == "no_validation"

        default_check_params = {"estimator": self}
        check_params = {**default_check_params, **check_params}

        if no_val_X and no_val_y:
            raise ValueError("Validation should be done on X, y or both.")
        elif not no_val_X and no_val_y:
            X = check_array(X, input_name="X", **check_params)
            out = X
        elif no_val_X and not no_val_y:
            y = _check_y(y, **check_params)
            out = y
        else:
            if validate_separately:
                # We need this because some estimators validate X and y
                # separately, and in general, separately calling check_array()
                # on X and y isn't equivalent to just calling check_X_y()
                # :(
                check_X_params, check_y_params = validate_separately
                if "estimator" not in check_X_params:
                    check_X_params = {**default_check_params, **check_X_params}
                X = check_array(X, input_name="X", **check_X_params)
                if "estimator" not in check_y_params:
                    check_y_params = {**default_check_params, **check_y_params}
                y = check_array(y, input_name="y", **check_y_params)
            else:
                X, y = check_X_y(X, y, **check_params)
            out = X, y

        if not no_val_X and check_params.get("ensure_2d", True):
            self._check_n_features(X, reset=reset)

        return out

    def _more_tags(self):
        return _DEFAULT_TAGS

    def _get_tags(self):
        collected_tags = {}
        for base_class in reversed(inspect.getmro(self.__class__)):
            if hasattr(base_class, "_more_tags"):
                # need the if because mixins might not have _more_tags
                # but might do redundant work in estimators
                # (i.e. calling more tags on BaseEstimator multiple times)
                more_tags = base_class._more_tags(self)
                collected_tags.update(more_tags)
        return collected_tags

    def _check_n_features(self, X, reset):
        """
        Set the `n_features_in_` attribute, or check against it.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        reset : bool
            If True, the `n_features_in_` attribute is set to `X.shape[1]`.
            If False and the attribute exists, then check that it is equal to
            `X.shape[1]`. If False and the attribute does *not* exist, then
            the check is skipped.
            .. note::
               It is recommended to call reset=True in `fit` and in the first
               call to `partial_fit`. All other methods that validate `X`
               should set `reset=False`.
        """
        try:
            n_features = _num_features(X)
        except TypeError as e:
            if not reset and hasattr(self, "n_features_in_"):
                raise ValueError(
                    "X does not contain any features, but "
                    f"{self.__class__.__name__} is expecting "
                    f"{self.n_features_in_} features"
                ) from e
            # If the number of features is not defined and reset=True,
            # then we skip this check
            return

        if reset:
            self.n_features_in_ = n_features
            return

        if not hasattr(self, "n_features_in_"):
            # Skip this check if the expected number of expected input features
            # was not recorded by calling fit first. This is typically the case
            # for stateless transformers.
            return

        if n_features != self.n_features_in_:
            raise ValueError(
                f"X has {n_features} features, but {self.__class__.__name__} "
                f"is expecting {self.n_features_in_} features as input."
            )


class ClassifierMixin:
    def score(self, X, y, sample_weight=None):
        raise NotImplementedError

    def fit(self, X, y, **kwargs):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


class TransformerMixin:
    def fit_transform(self, X, y=None, **fit_params):
        raise NotImplementedError


class RegressorMixin:
    def score(self, X, y, sample_weight=None):
        raise NotImplementedError

    def fit(self, X, y, **kwargs):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


class MultiOutputMixin:
    def _more_tags(self):
        return {"multioutput": True}


# --- Helpers --- #
# --------------- #


def _clone_parametrized(estimator, *, safe=True):
    """
    Default implementation of clone.

    See :func:`sklearn.base.clone` for details.
    """

    estimator_type = type(estimator)
    # XXX: not handling dictionaries
    if estimator_type in (list, tuple, set, frozenset):
        return estimator_type([clone(e, safe=safe) for e in estimator])
    elif not hasattr(estimator, "get_params") or isinstance(estimator, type):
        if not safe:
            return ivy.copy.deepcopy(estimator)
        else:
            if isinstance(estimator, type):
                raise TypeError(
                    "Cannot clone object. "
                    + "You should provide an instance of "
                    + "scikit-learn estimator instead of a class."
                )
            else:
                raise TypeError(
                    "Cannot clone object '%s' (type %s): "
                    "it does not seem to be a scikit-learn "
                    "estimator as it does not implement a "
                    "'get_params' method." % (repr(estimator), type(estimator))
                )

    klass = estimator.__class__
    new_object_params = estimator.get_params(deep=False)
    for name, param in new_object_params.items():
        new_object_params[name] = clone(param, safe=False)
    new_object = klass(**new_object_params)
    params_set = new_object.get_params(deep=False)

    # quick sanity check of the parameters of the clone
    for name in new_object_params:
        param1 = new_object_params[name]
        param2 = params_set[name]
        if param1 is not param2:
            raise RuntimeError(
                "Cannot clone object %s, as the constructor "
                "either does not set or modifies parameter %s" % (estimator, name)
            )

    # _sklearn_output_config is used by `set_output` to configure the output
    # container of an estimator.
    if hasattr(estimator, "_sklearn_output_config"):
        new_object._sklearn_output_config = ivy.copy.deepcopy(
            estimator._sklearn_output_config
        )
    return new_object


# --- Main --- #
# ------------ #


def clone(estimator, *, safe=True):
    """
    Construct a new unfitted estimator with the same parameters.

    Clone does a deep copy of the model in an estimator
    without actually copying attached data. It returns a new estimator
    with the same parameters that has not been fitted on any data.

    .. versionchanged:: 1.3
        Delegates to `estimator.__sklearn_clone__` if the method exists.

    Parameters
    ----------
    estimator : {list, tuple, set} of estimator instance or a single \
            estimator instance
        The estimator or group of estimators to be cloned.
    safe : bool, default=True
        If safe is False, clone will fall back to a deep copy on objects
        that are not estimators. Ignored if `estimator.__sklearn_clone__`
        exists.

    Returns
    -------
    estimator : object
        The deep copy of the input, an estimator if input is an estimator.

    Notes
    -----
    If the estimator's `random_state` parameter is an integer (or if the
    estimator doesn't have a `random_state` parameter), an *exact clone* is
    returned: the clone and the original estimator will give the exact same
    results. Otherwise, *statistical clone* is returned: the clone might
    return different results from the original estimator. More details can be
    found in :ref:`randomness`.
    """
    if hasattr(estimator, "__sklearn_clone__") and not inspect.isclass(estimator):
        return estimator.__sklearn_clone__()
    return _clone_parametrized(estimator, safe=safe)


def is_classifier(estimator):
    """
    Return True if the given estimator is (probably) a classifier.

    Parameters
    ----------
    estimator : object
        Estimator object to test.

    Returns
    -------
    out : bool
        True if estimator is a classifier and False otherwise.
    """
    return getattr(estimator, "_estimator_type", None) == "classifier"
