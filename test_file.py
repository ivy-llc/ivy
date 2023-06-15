import ivy
from ivy.functional.ivy.losses import mse_loss
x = ivy.Array([1,1,1])
y = ivy.Array([0,0,0])
print(mse_loss(x,y))