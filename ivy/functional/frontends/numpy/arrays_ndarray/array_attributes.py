import ivy
import numpy as np

def max(self,
        axis=None,
        out=None,
        keepdims=False,
        initial=np._NoValue,
        where=True):
    if (len(self)>0):
        return ivy.max(self, axis=axis, keepdims=keepdms, out=out)
    elif (len(self)==0 and initial==np._NoValue):
            raise Exception("An error has occurred. Set the parameter `initial` to return max on an empty array")
    else:
        return initial
