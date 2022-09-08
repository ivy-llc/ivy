# global
import ivy

def correlate(
    x,    
    y,
    mode = None,
    *,
    old_behavior = False,
    out = None
):

    mode = mode if mode is not None else "valid"
    assert x.ndim == 1 and y.ndim == 1
    n = min(x.shape[0], y.shape[0])
    m = max(x.shape[0], y.shape[0])
    if x.shape[0] > y.shape[0]:
        if mode == "full":
            r = n+m-1
            for j in range(0,n-1):
                x = ivy.concat((ivy.array(0),x),axis = None)
        elif mode == "same":
            r = m
            for j in range(0,(n-1)//2):
                x = ivy.concat((ivy.array(0),x),axis = None)
        elif mode == "valid":
            r = m-n+1
        else:
            assert False, "Invalid Mode"
        ret = ivy.array([(y[:n]*ivy.roll(x,-t)[:n]).sum().tolist() for t in range(0,r)], out=out)
    else:
        if mode == "full":
            r = n+m-1
            for j in range(0,n-1):
                y = ivy.concat((ivy.array(0),y),axis = None)
        elif mode == "same":
            r = m
            for j in range(0,(n-1)//2):
                y = ivy.concat((ivy.array(0),y),axis = None)
        elif mode == "valid":
            r = m-n+1
        else:
            assert False, "Invalid Mode"
        ret = ivy.flip(ivy.array([(x[:n]*ivy.roll(y,-t)[:n]).sum().tolist() for t in range(0,r)], out=out), out=out)
    return ret
