"""Collection of Jax network layers, wrapped to fit Ivy syntax and signature."""

# global
import jax.lax as jlax

# local
from ivy.functional.backends.jax import JaxArray
from typing import Union, Tuple, Optional

def multi_head_attention(
    x: JaxArray,
    scale: float,
    num_heads: int,
    context: Optional[JaxArray] = None,
    mask: Optional[JaxArray] = None,
    to_q_fn=None,
    to_kv_fn=None,
    to_out_fn=None,
    to_q_v: Optional[JaxArray] = None,
    to_kv_v: Optional[JaxArray] = None,
    to_out_v: Optional[JaxArray] = None,
) -> JaxArray:
    raise Exception("multi_head_attention not yet implemented for jax library")

def conv1d(
    x: JaxArray,
    filters: JaxArray,
    strides: int,
    padding: str,
    data_format: str = "NWC",
    dilations: int = 1,
) -> JaxArray:
    strides = (strides,) if isinstance(strides, int) else strides
    dilations = (dilations,) if isinstance(dilations, int) else dilations
    return jlax.conv_general_dilated(
        x, filters, strides, padding, None, dilations, (data_format, "WIO", data_format)
    )


def conv1d_transpose(*_):
    raise Exception("Convolutions not yet implemented for jax library")


def conv2d(
    x: JaxArray,
    filters: JaxArray,
    strides: Union[int, Tuple[int, int]],
    padding: str,
    data_format: str = "NHWC",
    dilations: int = 1,
) -> JaxArray:
    strides = [strides] * 2 if isinstance(strides, int) else strides
    dilations = [dilations] * 2 if isinstance(dilations, int) else dilations
    return jlax.conv_general_dilated(
        x,
        filters,
        strides,
        padding,
        None,
        dilations,
        (data_format, "HWIO", data_format),
    )


def depthwise_conv2d(*_):
    raise Exception("Convolutions not yet implemented for jax library")


def conv2d_transpose(*_):
    raise Exception("Convolutions not yet implemented for jax library")


def conv3d(*_):
    raise Exception("Convolutions not yet implemented for jax library")


def conv3d_transpose(*_):
    raise Exception("Convolutions not yet implemented for jax library")
