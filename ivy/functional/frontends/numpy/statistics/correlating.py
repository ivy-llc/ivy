# global
import ivy

def correlate(
    a,    
    v,
    mode = None,
    *,
    old_behavior = False
):

    mode = mode if mode is not None else "valid"
    assert a.ndim == 1 and v.ndim == 1
    n = min(a.shape[0], v.shape[0])
    m = max(a.shape[0], v.shape[0])
    if a.shape[0] > v.shape[0]:
        if mode == "full":
            r = n+m-1
            for j in range(0,n-1):
                a = ivy.concat((ivy.array(0),a),axis = None)
        elif mode == "same":
            r = m
            for j in range(0,(n-1)%2):
                a = ivy.concat((ivy.array(0),a),axis = None)
        elif mode == "valid":
            r = m-n+1
        else:
            assert False, "Invalid Mode"
        ret = ivy.array([(v[:n]*ivy.roll(a,-t)[:n]).sum().tolist() for t in range(0,r)])
    else:
        if mode == "full":
            r = n+m-1
            for j in range(0,n-1):
                v = ivy.concat((ivy.array(0),v),axis = None)
        elif mode == "same":
            r = m
            for j in range(0,(n-1)%2):
                v = ivy.concat((ivy.array(0),v),axis = None)
        elif mode == "valid":
            r = m-n+1
        else:
            assert False, "Invalid Mode"
        ret = ivy.flip(ivy.array([(a[:n]*ivy.roll(v,-t)[:n]).sum().tolist() for t in range(0,r)]))
    return ret
