import ivy
import torch
ivy.set_backend("jax")

def normalize(x):
    mean = torch.mean(x)
    std = torch.std(x)
    return torch.div(torch.sub(x, mean), std)

# convert the function to Ivy code
ivy_normalize = ivy.unify(normalize)

# trace the Ivy code into jax functions
jax_normalize = ivy.trace_graph(ivy_normalize)