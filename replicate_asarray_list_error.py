# This file is just here to make it easy
# to replicate a minimal example of the
# bug found in the tests.

# This file is not intended to be merged to master at all.

# Create an array X with each backend.
# Then call `ivy.asarray([X])` with the
# same backend.
# Print them and their shapes at the end
# to ensure that a) there's no crashes
# and b) they're all the same.

import ivy

# -- JAX --
ivy.set_backend("jax")
jax_array = ivy.asarray([False])
try:
    jax_error_array = ivy.asarray([jax_array])
except Exception as e:
    print("Crash in Jax")
    print(e)
    print("\n")
    jax_error_array = None

try:
    print(f"Jax array : {jax_error_array} with shape {jax_error_array.shape}")
except Exception as e:
    print(e)


# -- TF --
ivy.set_backend("tensorflow")
tf_array = ivy.asarray([False])
try:
    tf_error_array = ivy.asarray([tf_array])
except Exception as e:
    print("Crash in TF")
    print(e)
    print("\n")
    tf_error_array = None

try:
    print(f"TF array : {tf_error_array} with shape {tf_error_array.shape}")
except Exception as e:
    print(e)


# -- Torch --
ivy.set_backend("torch")
torch_array = ivy.asarray([False])
try:
    torch_error_array = ivy.asarray([torch_array])
except Exception as e:
    print("Crash in Torch")
    print(e)
    print("\n")
    torch_error_array = None

try:
    print(f"Torch array : {torch_error_array} with shape {torch_error_array.shape}")
except Exception as e:
    print(e)

ivy.set_backend("numpy")
np_array = ivy.asarray([False])
try:
    np_error_array = ivy.asarray([np_array])
except Exception as e:
    print("Crash in NP")
    print(e)
    print("\n")
    np_error_array = None


try:
    print(f"NP array : {np_error_array} with shape {np_error_array.shape}")
except Exception as e:
    print(e)
