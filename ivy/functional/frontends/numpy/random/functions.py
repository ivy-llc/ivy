# local
import ivy


def random(size=None):
    return ivy.random_uniform(low=0.0, high=1.0, shape=size, dtype="float64")


def dirichlet(alpha, size=None):
    size = size if size is not None else 1
    n = min(alpha)
    alpha = ivy.array(alpha)

    if type(alpha) in [int, float]:
        assert False, f"object of type {type(alpha)} has no len()"
    if any(x<=0 for x in alpha.flat):
        raise ValueError("alpha<=0")
    if any(x<0 for x in size):
        raise ValueError("negative dimensions are not allowed")

    if type(size) == int:
        lst = []
        for i in range(0, size):
            alpha /= ivy.random_uniform(shape=alpha.shape)
            s = ivy.sum(alpha)
            lst.append((alpha / s).tolist())
        ret = ivy.array(lst, dtype="float64")        
    elif type(size) == tuple:
        shape = size + (alpha.size, )  
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
        assert False, f"{type(size)} object is not iterable"
    return ret
