# local
import ivy


def where(condition, /, x=None, y=None):
    if x== None and y ==None :
        # numpy where behaves as np.asarray(condition).nonzero() when x and y not included
        return ivy.asarray(condition).nonzero()
    elif x!= None and y!=None :
        return ivy.where(condition, x, y)
    else :
        raise TypeError(f'where takes either 1 or 3 arguments, {'x' if x == None else 'y'} not given')
