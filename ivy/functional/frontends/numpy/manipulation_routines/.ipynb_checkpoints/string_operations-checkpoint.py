#local

import ivy


def add(x1,
        x2,
        /,
        out=None,
        *,
        where=True,
        casting='same_kind',
        order='K',
        dtype=None,
        subok=True[, signature, extobj]) =                 <ufunc'add'>:
    return ivy.add(x1,x2, axis,out=out)

    arrays = [promote_types_of_numpy_inputs(a) for a in arrays]
    #x1, x2 = promote_types_of_numpy_inputs(x1, x2)
    ret = ivy.add(x1,x2, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret
