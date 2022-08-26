# global


import ivy.functional.frontends.jax as ivy_frontend


def reshape(self, new_sizes, dimensions=None):
    return ivy_frontend.reshape(self, new_sizes, dimensions)


def add(self, other):
    return ivy_frontend.add(self, other)
