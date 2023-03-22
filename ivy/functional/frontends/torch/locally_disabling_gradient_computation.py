import ivy
import ivy.functional.frontends.torch as torch_frontend

def no_grad():
    return ivy.with_grads(with_grads = False)


def enable_grad():
    return ivy.with_grads(with_grads = True)


def set_grad_enabled(mode=True):
    return ivy.with_grads(with_grads = mode)

"""
def is_grad_enabled():
    return 

def inference_mode(mode=True):
    return 

def is_inference_mode_enabled():
    return
"""
