from .tensorflow__helpers import tensorflow_set_item


def tensorflow___setitem__(self, query, val):
    self = tensorflow_set_item(self, query, val)
