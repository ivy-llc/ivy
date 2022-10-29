import ivy
import torch

# import numpy as np
# import tensorflow as tf
# import jax


class Demo(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # self.add1 = np.add()

    def forward(self, x, y):
        return ivy.add(x, y)


model = Demo()
pred = model(torch.tensor([1, 2]), torch.tensor([3, 4]))
# print(pred)
# print(model.__dict__.keys())

torch_model = torch.nn.Linear(5, 10)
print(torch_model)

torch_to_ivy = ivy.to_ivy_module(torch_model)
print(torch_to_ivy)

ivy_to_torch = torch_to_ivy.to_torch_module()
print(ivy_to_torch)
