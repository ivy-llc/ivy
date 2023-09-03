from ivy.functional.frontends.sklearn.base import BaseEstimator, TransformerMixin


class LabelEncoder(TransformerMixin, BaseEstimator):
    def fit(self, y):
        raise NotImplementedError

    def fit_transform(self, y):
        raise NotImplementedError

    def transform(self, y):
        raise NotImplementedError

    def inverse_transform(self, y):
        raise NotImplementedError
