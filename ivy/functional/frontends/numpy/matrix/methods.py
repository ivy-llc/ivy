import ivy
import numpy as np

def matrix_all(a, axis=None, out=None, keepdims=np._NoValue, *, where=np._NoValue):
  if len(a)==0:
        return "NaN"
    else:
        res = ivy.all(a, axis, out=out, keepdims=keepdims, where)
        return res
    
