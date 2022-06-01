import ivy
from ivy.backend_handler import current_backend as _cur_backend
ivy.set_backend('torch')
x = ivy.random_normal(shape = (8, 6))
S = ivy.svdvals(x)
print(S.shape)
#(6, 3)
# Compare the singular value S by ivy.svdvals() with the result by ivy.svd().
_, SS, _ = ivy.svd(x)
print(SS.shape)
error = (SS - S).abs()
print(error)

x = ivy.asarray([[1.0, 2.0, 3.0],[2.0, 3.0, 4.0], [2.0, 1.0, 3.0], [3.0, 4.0, 5.0]])
print(x.shape)
S = ivy.svdvals(x)
print(S)
# Compare the singular value S by ivy.svdvals() with the result by ivy.svd().
_, SS, _ = ivy.svd(x)
print(SS)
error = (SS - S).abs()
print(error)