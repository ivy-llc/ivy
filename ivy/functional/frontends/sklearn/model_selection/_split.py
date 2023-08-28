import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back

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


@to_ivy_arrays_and_back
def train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None):
    # TODO: Make it concise
    # TODO: implement stratify
    if stratify is not None:
        raise NotImplementedError
    if len(arrays) == 0:
        raise ValueError("At least one array required as input")
    if test_size is None and train_size is None:
        test_size = 0.25
    n_samples = arrays[0].shape[0]
    n_train = ivy.floor(train_size * n_samples) if isinstance(train_size, float) \
        else float(train_size) if isinstance(train_size, int) else None
    n_test = ivy.ceil(test_size * n_samples) if isinstance(test_size, float) \
        else float(test_size) if isinstance(test_size, int) else None
    if train_size is None:
        n_train = n_samples - n_test
    elif test_size is None:
        n_test = n_samples - n_train

    n_train, n_test = int(n_train), int(n_test)
    indices = ivy.arange(0,  n_train + n_test)
    if shuffle:
        if random_state is not None:
            ivy.seed(random_state)
        indices = ivy.shuffle(indices)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    output = []
    for array in arrays:
        output.append(ivy.gather(array, train_indices, axis=0))
        output.append(ivy.gather(array, test_indices, axis=0))
    return tuple(output)
