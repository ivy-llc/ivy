import ivy

# import torch

# import numpy as np
# import tensorflow as tf

# import jax


# torch
# class Demo(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         self.linear = torch.nn.Linear(5, 10)
#
#     def forward(self, x):
#         return self.linear(x)
#
# class Demo2(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         self.linear = Demo()
#         self.linear2 = Demo()
#
#     def forward(self, x):
#         return self.linear(x)


# model = Demo2()
# print([x for x in model.parameters()])
# print([x for x in model.linear.parameters()])
# pred = model(torch.tensor([1, 2]), torch.tensor([3, 4]))
# # print(pred)
# # print(model.__dict__.keys())
#
# torch_model = torch.nn.Linear(5, 10)
# # print(torch_model)
#
# torch_to_ivy = ivy.to_ivy_module(torch_model)
# # print(torch_to_ivy)
#
# ivy_to_torch = torch_to_ivy.to_torch_module()
# print(ivy_to_torch)


# ivy
class IvyLinearModule(ivy.Module):
    def __init__(self, in_size, out_size):
        # super().__init__()
        self._linear = ivy.Linear(in_size, out_size)
        super().__init__()

    def _forward(self, x):
        return self._linear(x)


class IvyDemoModule(ivy.Module):
    def __init__(self, in_size, out_size, device=None, hidden_size=64):
        # super().__init__()
        self.linear0 = IvyLinearModule(in_size, hidden_size)
        self.linear1 = IvyLinearModule(hidden_size, hidden_size)
        self.linear2 = IvyLinearModule(hidden_size, out_size)
        super().__init__()

    def _forward(self, x):
        # x = x.unsqueeze(0)
        x = ivy.tanh(self.linear0(x))
        x = ivy.tanh(self.linear1(x))
        return ivy.tanh(self.linear2(x))[0]


ivy.set_backend("torch")
ivy_demo_module = IvyDemoModule(5, 10)
# print(ivy_demo_module)
# print(ivy_demo_module(ivy.array([3,5,6,7,8])))

# print(ivy_demo_module.__dict__.keys())
# print('\n')
# print(ivy_demo_module.sub_mods())
# print(ivy_demo_module.submod_dict)
# print(ivy_demo_module.submod_call_order)
# print(ivy_demo_module._track_submod_call_order)
# print(ivy_demo_module.linear0)
# print(ivy_demo_module.linear1)
# print(ivy_demo_module.linear2)
# print(type(ivy_demo_module._linear0))
# print([x for x in ivy_demo_module.v.keys()])
# print(ivy_demo_module.show_structure())
# print('\n')

# torch_demo_module = ivy_demo_module.to_torch_module()
# print(torch_demo_module(torch.tensor([3,5,6,7,8])))
# optimizer = torch.optim.SGD(torch_demo_module.parameters(), lr=0.01, momentum=0.9)
# loss_fn = torch.nn.MSELoss()
#
# optimizer.zero_grad()
# output = torch_demo_module(torch.tensor([3,5,6,7,8]))
# loss = loss_fn(output, torch.tensor([3,7,9,1,2,4,2,3,1,4], dtype=torch.float32))
# loss.backward()
# optimizer.step()
# print(torch_demo_module.__dict__.keys())
# print(torch_demo_module.parameters())
# print([x for x in torch_demo_module.named_parameters()])
# print(torch_demo_module._modules)
# ivy.unset_backend()


# ivy.set_backend("tensorflow")
# keras_demo_module = ivy_demo_module.to_keras_model()
# print(keras_demo_module.__dict__.keys())
# print(keras_demo_module(tf.constant([3,5,6,7,8])))
# ivy.unset_backend()
