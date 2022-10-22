import ivy
import ivy.functional.frontends.tensorflow as tf_frontend


def dct(input, type=2, n=None, axis=-1, norm=None, name=None ):
        return ivy.dct(input, type, n, axis, norm, name)
