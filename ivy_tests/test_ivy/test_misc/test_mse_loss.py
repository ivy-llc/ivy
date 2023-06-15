import ivy
import ivy.functional.ivy.losses as l
import numpy as np
x = np.array([[1,1,1],[1,1,1]])
y = np.array([[0,0,0],[0,0,0]])
x = ivy.Array(x)
y = ivy.Array(y)
print(l.mse_loss(x,y))