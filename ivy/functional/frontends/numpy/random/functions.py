# local
import ivy


def random(size=None):
    return ivy.random_uniform(low=0.0, high=1.0, shape=size, dtype="float64")


def dirichlet(alpha, size=None):
    size = size if size is not None else 1
    ivy.assertions.check_true(type(alpha) not in [int, float],
                              f"object of type {type(alpha)} has no len()")
    ivy.assertions.check_true(all(x > 0 for x in alpha), "alpha <= 0")
    n = min(alpha)
    alpha = ivy.array(alpha)

    if type(size) == int:
        ivy.assertions.check_greater(
            size, 0, allow_equal=False,
            message="negative dimensions are not allowed")
        lst = []
        for i in range(0, size):
            alpha /= ivy.random_uniform(shape=alpha.shape)
            s = ivy.sum(alpha)
            lst.append((alpha / s).tolist())
        ret = ivy.array(lst, dtype="float64")
    elif type(size) in [tuple, list]:
        ivy.assertions.check_true(
            all(x >= 0 for x in size), "negative dimensions are not allowed")
        shape = tuple(size)
        shape = shape + (alpha.size, )
        uniform = ivy.random_uniform(low=n, shape=shape)
        flat = uniform.flatten().tolist()
        arr = ivy.array([flat[i:i + alpha.size]
                        for i in range(0, len(flat), alpha.size)])
        lst = []
        for i in range(0, arr.shape[0]):
            alpha /= arr[i]
            s = ivy.sum(alpha)
            lst.append((alpha / s).tolist())
        ret = ivy.array(lst, dtype="float64").reshape(shape)
    else:
        raise ivy.exceptions.IvyException(
            f"{type(size)} object is not iterable")
    return ret
