# in ivy/functional/frontends/torch/torch.py
@to_ivy_arrays_and_back

import torch
import ivy.functional.frontends.torch as torch_frontend

def to_cpu(self):
     if self.is_cuda:
          return torch_frontend.self.to_cpu()
     else:
          return self

torch.Tensor.to_cpu = to_cpu

x = torch.randn(2, 3).cuda()

y = x.to_cpu()

print(y.device)
