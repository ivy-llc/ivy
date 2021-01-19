"""
Collection of MXNet general functions, wrapped to fit Ivy syntax and signature.
"""

# global
import mxnet as _mx
_round = round


def _mxnet_init_context(dev):
    if dev is None or dev.find("cpu") != -1:
        mx_dev = "cpu"
    elif dev.find("cuda") != -1:
        mx_dev = "gpu"
    else:
        raise Exception("dev type not supported.")
    if dev.find(":") != -1:
        mx_dev_id = int(dev[dev.find(":")+1:])
    else:
        mx_dev_id = 0
    return _mx.Context(mx_dev, mx_dev_id)


def array(object_in, dtype_str=None, dev=None):
    cont = _mxnet_init_context('cpu' if not dev else dev)
    return _mx.nd.array(object_in, cont, dtype=dtype_str)
