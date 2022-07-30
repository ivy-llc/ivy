# local
import ivy

def sort(arrays,dtype=None):
    if dtype:
        arrays = [ivy.astype(ivy.array(a), ivy.as_ivy_dtype(dtype)) for a in arrays]
    return ivy.sort(arrays)
