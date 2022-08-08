import ivy

def dropout(input, p = 0.5, scale = True, out=None):
    return ivy.dropout(x = input, prob = p, scale = scale , out = out)