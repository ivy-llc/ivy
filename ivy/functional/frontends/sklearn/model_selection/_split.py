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
    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array required as input")
    if test_size is None and train_size is None:
        test_size = 0.25
    n_samples = arrays[0].shape[0]
    test_size_type, train_size_type = type(test_size), type(train_size)
    if "f" in str(test_size_type):
        n_test = ivy.ceil(test_size * n_samples)
    elif "i" in str(test_size_type):
        n_test = float(test_size)
    else:
        n_test = 0

    if "f" in str(train_size_type):
        n_train = ivy.floor(train_size * n_samples)
    elif "i" in str(train_size_type):
        n_train = float(train_size)
    else:
        n_train = 0

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
