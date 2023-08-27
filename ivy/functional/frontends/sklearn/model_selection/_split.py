class BaseCrossValidator:
    def split(self, X, y=None, groups=None):
        raise NotImplementedError

    def _iter_test_masks(self, X=None, y=None, groups=None):
        raise NotImplementedError

    def _iter_test_indices(self, X=None, y=None, groups=None):
        raise NotImplementedError


class KFold(BaseCrossValidator):
    def __init__(
        self,
        n_splits=5,
        *,
        shuffle=False,
        random_state=None,
    ):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        raise NotImplementedError

    def _iter_test_masks(self, X=None, y=None, groups=None):
        raise NotImplementedError

    def _iter_test_indices(self, X=None, y=None, groups=None):
        raise NotImplementedError
