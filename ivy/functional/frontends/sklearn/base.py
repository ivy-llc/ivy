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


class RegressorMixin:

    def score(self, X, y, sample_weight=None):
        raise NotImplementedError

    def fit(self, X, y, **kwargs):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
