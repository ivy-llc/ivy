import ivy
import ivy.functional.ivy.losses as l
import numpy as np
def mse_loss_test(x,y):
    x = ivy.Array(x)
    y = ivy.Array(y)
    return l.mse_loss(x,y)




