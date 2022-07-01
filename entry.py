import ivy
import numpy as np
import torch
import jax.numpy as jnp
def ivy_matmul(x,y):
    return ivy.matmul(x,y)
x = np.array([2, 3, 4])
y = np.array([6, 5, 4])
ivy.set_backend("jax")
jx = ivy.array(x)
jy = ivy.array(y)
jxm = ivy_matmul(jx,jy)

print("jax ivy:", ivy_matmul(jx,jy))
ivy.unset_backend()
ivy.set_backend("numpy")
nx = ivy.array(x)
ny = ivy.array(y)
nxm = ivy_matmul(nx,ny)
print("numpy ivy:", ivy_matmul(nx,ny))
ivy.unset_backend()


ivy.set_backend("torch")
tx = ivy.array(x)
ty = ivy.array(y)
txm = ivy_matmul(tx,ty)
print("torch ivy:", ivy_matmul(tx,ty))
ivy.unset_backend()

# ivy.set_backend("tensorflow")
# tfx = ivy.array(x)
# tfy = ivy.array(y)
# tfxm = ivy_matmul(tfx,tfy)
# print("tensorflow ivy:", ivy_matmul(tfx,tfy))
# ivy.unset_backend()


print("jax:", jnp.matmul(x,y))
print("numpy:", np.matmul(x,y))
print("torch:", torch.matmul(torch.tensor(x),torch.tensor(y)))
#print("tensorflow:", tf.matmul(tf.constant(x), tf.constant(y)))
