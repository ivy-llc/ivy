import ivy
from typing import Union, Optional
from ivy.func_wrapper import (
    to_native_arrays_and_back,
    handle_out_argument,
    handle_nestable,
    handle_array_like,
)
from ivy.exceptions import handle_exceptions
@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like
def sort(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
    out: Optional[ivy.Array] = None,
):
    if axis==1:
        x=ivy.sort(x,axis=1,out=out)
    if descending:
        x=ivy.sort(x,descending,out=out)
    if stable is False:
        x = ivy.sort(x, stable == False,out=out)
    return x
