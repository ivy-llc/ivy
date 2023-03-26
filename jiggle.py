import ivy
import numpy as np

for fw in ["torch", "jax", "tensorflow"]:
    try:
        ivy.set_backend(fw)
        a = ivy.array(1e-5)
        # c = ivy.to_numpy(a)
        b = np.array(a)
    except Exception as e:
        print(f"Error in {fw}: {e}")
