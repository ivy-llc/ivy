import torch
import ivy.functional.ivy.experimental as back
import ivy

ivy.set_backend("numpy")
a = torch.tensor([[1, 2], [3, float("nan")]])
# print(np.nanmin(a, axis=0,keepdims=True))
# print(np.nanmin(a, axis=1))
print(back.statistical.nanmin(a, axis=0, keepdims=True))
print(back.statistical.nanmin(a, axis=1))

# #[[1. 2.]]
# #[1. 3.]
