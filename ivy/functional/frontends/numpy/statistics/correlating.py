# global
import ivy

def correlate(
    a,    
    v,
    *,
    mode = None,
    out = None
):

    n = min(len(a),len(v))
    m = max(len(a),len(v))
    
    if len(a)>len(v):
        if mode == "full":
            r = n+m-1
            for j in range(0,n-1):
                a = ivy.concat((ivy.array(0),a),axis = None)
        elif mode == "same":
            r = m
            for j in range(0,(n-1)//2):
                a = ivy.concat((ivy.array(0),a),axis = None)
        else:
            r = m-n+1
            
        ret = ivy.array([(v[:n]*ivy.roll(a,-t)[:n]).sum() for t in range(0,r)])

    else:
        if mode == "full":
            r = n+m-1
            for j in range(0,n-1):
                v = ivy.concat((ivy.array(0),v),axis = None)
        elif mode == "same":
            r = m
            for j in range(0,(n-1)//2):
                v = ivy.concat((ivy.array(0),v),axis = None)
        else:
            r = m-n+1
        
        ret = ivy.flip([(a[:n]*ivy.roll(v,-t)[:n]).sum() for t in range(0,r)])  
    return ret
