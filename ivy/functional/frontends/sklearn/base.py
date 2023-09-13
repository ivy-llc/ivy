class BaseEstimator:
    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


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
