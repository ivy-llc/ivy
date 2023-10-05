from ivy.functional.frontends.sklearn.base import BaseEstimator, TransformerMixin
import ivy


class LabelEncoder(TransformerMixin, BaseEstimator):
    def fit(self, y):
        shape = y.shape
        if len(shape) == 2 and shape[1] == 1:
            y = y.reshape(-1)
        elif len(shape) != 1:
            raise ValueError("y should be a 1d array, or column")
        self.classes_ = ivy.unique_values(y)
        return self

    def fit_transform(self, y):
        raise NotImplementedError

    def transform(self, y):
        raise NotImplementedError

    def inverse_transform(self, y):
        raise NotImplementedError
