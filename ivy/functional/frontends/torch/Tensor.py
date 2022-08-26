# global

# local
import ivy.functional.frontends.torch as ivy_frontend


def reshape(self, shape):
    return ivy_frontend.reshape(self, shape)


def add(self, other, *, alpha=1, out=None):
    return ivy_frontend.add(self, other * alpha, out=out)
