import ivy
from ivy.functional.frontends.torch.func_wrapper import to_ivy_arrays_and_back

@to_ivy_arrays_and_back
def result_type(tensor1, tensor2):
    return ivy.result_type(tensor1, tensor2)
