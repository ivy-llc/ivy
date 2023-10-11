import ivy
import ivy.functional.ivy as ivy_back
import numpy as np


arr = np.array([[-1.0]], dtype="float32")
print(
    "Ground Truth value directly tested with np.min=",
    np.min(arr, axis=0, keepdims=True, initial=0, where=ivy.array([False])),
)

for backend in ["numpy", "jax", "tensorflow", "paddle", "torch"]:
    ivy.set_backend(backend)
    a = ivy.array([[-1.0]], dtype="float32")
    container_a = ivy.Container(array_a=a)
    print(
        "Value from"
        f" {backend} is={ivy_back.min(container_a, axis=0, keepdims=True, initial=0, where=ivy.array([False]))}"
    )
    ivy.unset_backend()
