# global
import ivy

def correlate(
    x,    
    y,
    *,
    mode = None,
    out = None
):

    n = min(len(x),len(y))
    m = max(len(x),len(y))
    
    if len(x)>len(y):
        if mode == "full":
            r = n+m-1
            for j in range(0,n-1):
                x = ivy.concat((ivy.array(0),x),axis = None)
        elif mode == "same":
            r = m
            for j in range(0,(n-1)//2):
                x = ivy.concat((ivy.array(0),x),axis = None)
        else:
            r = m-n+1
            
        ret = ivy.array([(y[:n]*ivy.roll(x,-t)[:n]).sum() for t in range(0,r)])

    else:
        if mode == "full":
            r = n+m-1
            for j in range(0,n-1):
                y = ivy.concat((ivy.array(0),y),axis = None)
        elif mode == "same":
            r = m
            for j in range(0,(n-1)//2):
                y = ivy.concat((ivy.array(0),y),axis = None)
        else:
            r = m-n+1
        
        ret = ivy.flip([(x[:n]*ivy.roll(y,-t)[:n]).sum() for t in range(0,r)])  
    return ret
