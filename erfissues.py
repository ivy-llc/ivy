import torch
print(torch.special.erfc(torch.tensor([0, -1., 10.])))
from ivy.functional.frontends import torch as ivy_torch
print(ivy_torch.special.erfc(ivy_torch.tensor([0, -1., 10.])))
print(ivy_torch.erfc(ivy_torch.tensor([0, -1., 10.])))