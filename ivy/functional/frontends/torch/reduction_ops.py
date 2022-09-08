import ivy

def dist(input,other,p=2):
    try:
        abs_val = ivy.abs(input - other)
    except:
        return f'The size of tensor a ({input.size}) must match the size of tensor b ({other.size}) at non-singleton dimension'
        
    if p == 0:
        # removing zeroes from the abs_val array as ivy.pow([0.0],0) if coming to be 1
        # but should be zero for pnorm calculation.
        abs_val_without_zero = abs_val[abs_val != 0]
        if abs_val_without_zero.size == 0:
            return abs_val
        return ivy.sum(ivy.pow(abs_val_without_zero,p))
    
    pnorm = ivy.pow(ivy.sum(ivy.pow(abs_val,p)),1/p)
    return pnorm

dist.unsupported_dtypes = ("float16",)