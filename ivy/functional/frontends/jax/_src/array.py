import ivy.functional.frontends.jax as jax_frontend


# Dummy Array class to help with compilation, don't add methods here
class ArrayImpl(jax_frontend.Array):
    pass
