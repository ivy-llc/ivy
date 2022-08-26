# global

# local
import ivy.functional.frontends.tensorflow as ivy_frontend


def reshape(self, shape, name=None):
    return ivy_frontend.reshape(self, shape, name)


def add(self, y, name=None):
    return ivy_frontend.add(self, y, name)
