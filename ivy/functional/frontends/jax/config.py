jax_enable_x64 = False


def update(value, toggle):
    global jax_enable_x64
    if value == "jax_enable_x64":
        jax_enable_x64 = toggle
