import ivy
# function for bincount
def bincount(input, weights=None, minlength=0):
    return ivy.bincount(input, weights=weights, minlength=minlength)