import ivy
from ivy.functional.frontends.tensorflow.func_wrapper import to_ivy_arrays_and_back



@to_ivy_arrays_and_back
def resize(images, size, preserve_aspect_ratio=False, antialias=False,name=None):
    return ivy.resize(images, size)