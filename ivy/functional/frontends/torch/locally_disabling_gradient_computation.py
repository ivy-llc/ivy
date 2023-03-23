import ivy
import ivy.functional.frontends.torch as torch_frontend
from ivy.functional.ivy.gradients import *

def no_grad():
    return GradientTracking(with_grads = False)

def enable_grad():
    return GradientTracking(with_grads = True)

def set_grad_enabled(mode=True):
    return GradientTracking(with_grads = mode)

def is_grad_enabled():
    return ivy.with_grads()

def inference_mode(mode=True):
    return GradientTracking(with_grads = not(mode)) 

def is_inference_mode_enabled():
    return not(ivy.with_grads())