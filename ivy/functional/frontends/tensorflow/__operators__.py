import ivy.functional.frontends.tensorflow as tf_frontend


def add(x, y, name=None):
    return tf_frontend.math.add(x, y, name=name)
