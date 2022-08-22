# local
import ivy

def is_nonzero(input):
    if ivy.array(input)==0:
        return False
    return True