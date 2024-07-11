from abc import ABCMeta, abstractmethod
import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back
from ivy.functional.frontends.sklearn.utils.validation import column_or_1d


class BaseCrossValidator(metaclass=ABCMeta):
    def split(self, X, y=None, groups=None):
        indices = ivy.arange(X.shape[0])
        for test_index in self._iter_test_masks(X, y, groups):
            train_index = indices[ivy.logical_not(test_index)]
            test_index = indices[test_index]
            yield train_index, test_index

    def _iter_test_masks(self, X=None, y=None, groups=None):
        for test_index in self._iter_test_indices(X, y, groups):
            test_mask = ivy.zeros(X.shape[0], dtype="bool")
            test_mask[test_index] = True
            yield test_mask

    def _iter_test_indices(self, X=None, y=None, groups=None):
        raise NotImplementedError

    @abstractmethod
    def get_n_splits(self, X=None, y=None, groups=None):
        pass


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

    def _iter_test_indices(self, X=None, y=None, groups=None):
        n_samples = X.shape[0]
        indices = ivy.arange(n_samples)
        if self.shuffle:
            indices = ivy.shuffle(indices, seed=self.random_state)

        n_splits = self.n_splits
        fold_sizes = ivy.full(
            n_splits, n_samples // n_splits, dtype=ivy.default_int_dtype()
        )
        fold_sizes[: n_samples % n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            yield indices[start:stop]
            current = stop

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class StratifiedKFold(KFold):
    def __init__(
        self,
        n_splits=5,
        *,
        shuffle=False,
        random_state=None,
    ):
        super().__init__(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
        )

    def _iter_test_indices(self, X=None, y=None, groups=None):
        ivy.seed(seed_value=self.random_state)
        y = ivy.array(y)
        y = column_or_1d(y)
        _, y_idx, y_inv, _ = ivy.unique_all(y)
        class_perm = ivy.unique_inverse(y_idx)
        y_encoded = class_perm[y_inv]

        n_classes = len(y_idx)
        y_order = ivy.sort(y_encoded)
        allocation = ivy.asarray(
            [
                ivy.bincount(y_order[i :: self.n_splits], minlength=n_classes)
                for i in range(self.n_splits)
            ]
        )
        test_folds = ivy.empty(len(y), dtype="int64")
        for k in range(n_classes):
            folds_for_class = ivy.arange(self.n_splits).repeat(allocation[:, k])
            if self.shuffle:
                folds_for_class = ivy.shuffle(folds_for_class)
            test_folds[y_encoded == k] = folds_for_class
        for i in range(self.n_splits):
            yield test_folds == i

    def split(self, X, y, groups=None):
        return super().split(X, y, groups)


@to_ivy_arrays_and_back
def train_test_split(
    *arrays,
    test_size=None,
    train_size=None,
    random_state=None,
    shuffle=True,
    stratify=None,
):
    # TODO: Make it concise
    # TODO: implement stratify
    if stratify is not None:
        raise NotImplementedError
    if len(arrays) == 0:
        raise ValueError("At least one array required as input")
    if test_size is None and train_size is None:
        test_size = 0.25
    n_samples = arrays[0].shape[0]
    n_train = (
        ivy.floor(train_size * n_samples)
        if isinstance(train_size, float)
        else float(train_size)
        if isinstance(train_size, int)
        else None
    )
    n_test = (
        ivy.ceil(test_size * n_samples)
        if isinstance(test_size, float)
        else float(test_size)
        if isinstance(test_size, int)
        else None
    )
    if train_size is None:
        n_train = n_samples - n_test
    elif test_size is None:
        n_test = n_samples - n_train

    n_train, n_test = int(n_train), int(n_test)
    indices = ivy.arange(0, n_train + n_test)
    if shuffle:
        if random_state is not None:
            ivy.seed(seed_value=random_state)
        indices = ivy.shuffle(indices)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    output = []
    for array in arrays:
        output.append(ivy.gather(array, train_indices, axis=0))
        output.append(ivy.gather(array, test_indices, axis=0))
    return tuple(output)
