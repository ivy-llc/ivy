import numpy as np
import ivy


def any(
    x,
    axis=None,
    out=None,
    keepdims=False,
    *,
    where=None
):
    xx = x.copy()
    if where is not None:
        w = np.array(where).copy()
        if len(w.shape) == 0:
            w = np.tile(w, xx.shape())
        else:
            currDim = -1
            listFinalShapes = []
            while (-currDim <= len(w.shape)):
                if w.shape[currDim] == xx.shape[currDim]:
                    listFinalShapes.append(1)
                elif w.shape[currDim] == 1:
                    listFinalShapes.append(x.shape[currDim])
                currDim -= 1
            while (-currDim <= len(xx.shape)):
                listFinalShapes.append(xx.shape[currDim]) 
                currDim -= 1
            w = np.tile(w, tuple(listFinalShapes[::-1])) 
        xx = xx * w
    xx = ivy.array(xx)
    return ivy.any(xx, axis, keepdims, out=out)	
