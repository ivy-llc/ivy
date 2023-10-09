import ivy
import ivy.functional.ivy.experimental as ivy_back
import numpy as np


arr = np.array([[-1.0]], dtype="float32")
print(
    "Ground Truth value directly tested with np.nanmin=",
    np.nanmin(arr, axis=0, keepdims=True, initial=0),
)

for backend in ["numpy", "tensorflow", "jax", "paddle", "torch"]:
    ivy.set_backend(backend)
    a = ivy.array([[-1.0]], dtype="float32")
    print(
        f"Value from {backend} is={ivy_back.nanmin(a,axis=0,keepdims=True,initial=0)}"
    )
    ivy.unset_backend()
