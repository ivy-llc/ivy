# global
import ivy

# local
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_numpy_casting,
    handle_numpy_dtype,
    from_zero_dim_arrays_to_scalar,
    handle_numpy_out,
)

# This is the cosine (cos) function.
# It returns the cosine input using the ivy cosine function.
@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _cos(
    x,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="k",
    dtype=None,
    subok=True,
):
    # Perform the cosine operation on the input array x
    ret = ivy.cos(x, out=out)
    
    # Check if where argument is an array and conditionally update ret array
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    
    return ret # Return result


# This is the sine (sin) function
# It returns the sine input using the ivy sine function.
@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _sin(
    x,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="k",
    dtype=None,
    subok=True,
):
    # Perform the sine operation on the input array x
    ret = ivy.sin(x, out=out)
    
    # Check if where argument is an array and conditionally update ret array
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    
    return ret # Return result


# This is the tangent (tan) function
# It returns the tangent input using the ivy tangent function.
@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _tan(
    x,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="K",
    dtype=None,
    subok=True,
):
    # Perform the tangent operation on the input array x
    ret = ivy.tan(x, out=out)
    
    # # Check if where argument is an array and conditionally update ret array
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    
    return ret # Return result


# This is the inverse sine (arcsin) function
# It returns the inverse sine input using the ivy inverse sine function.
@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _arcsin(
    x,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="K",
    dtype=None,
    subok=True,
):
    # Perform the inverse sine operation on the input array x
    ret = ivy.asin(x, out=out)
    
    # Check if where argument is an array and conditionally update ret array
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    
    return ret # Return result


# This is the inverse cosine (arccos) function
# It returns the inverse cosine input using the ivy inverse cosine function.
@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _arccos(
    x,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="K",
    dtype=None,
    subok=True,
):
    # Perform the inverse cosine operation on the input array x
    ret = ivy.acos(x, out=out)
    
    # Check if where argument is an array and conditionally update ret array
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    
    return ret # Return result


# This is the inverse tangent (arctan) function
# It returns the inverse tangent input using the ivy inverse tangent function.
@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _arctan(
    x,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="K",
    dtype=None,
    subok=True,
):
    # Perform the inverse tangent operation on the input array x
    ret = ivy.atan(x, out=out)
    
    # Check if where argument is an array and conditionally update ret array
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    
    return ret # Return result



# This is the inverse tangent of y/x (arctan2) function
# It returns the inverse tangent of y/x input using the ivy inverse tangent of y/x function.
@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _arctan2(
    x1,
    x2,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="K",
    dtype=None,
    subok=True,
    signature=None,
    extobj=None,
):
    # Perform the inverse tangent of y/x operation on the input array x
    ret = ivy.atan2(x1, x2, out=out)
    
    # # Check if where argument is an array and conditionally update ret array
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)

    return ret # Return result


# This is the degree to radians function
# It returns the degree to radians input using the ivy degree to radians function.
@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _deg2rad(
    x,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="K",
    dtype=None,
    subok=True,
    signature=None,
    extobj=None,
):
    # Perform the degree to radian operation on the input array x
    ret = ivy.deg2rad(x, out=out)
    
    # Check if where argument is an array and conditionally update ret array
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    
    return ret # Return result


# This is the radians to degree function
# It returns the radians to degree input using the ivy radians to degree function.
@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _rad2deg(
    x,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="K",
    dtype=None,
    subok=True,
):
    # Perform the radian to degree operation on the input array x
    ret = ivy.rad2deg(x, out=out)
    
    # Check if where argument is an array and conditionally update ret array
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    
    return ret # Return result


# This is the degree function
# It returns the degree input using the ivy degree function. It's the same operation as the function above.
@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _degrees(
    x,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="K",
    dtype=None,
    subok=True,
):
    # Perform the radian to degree operation on the input array x
    ret = ivy.rad2deg(x, out=out)
    
    # Check if where argument is an array and conditionally update ret array
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    
    return ret # Return result


# This is the radian function
# It returns the radian input using the ivy radian function. It's the same operation as the _deg2rad function.
@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _radians(
    x,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="K",
    dtype=None,
    subok=True,
    signature=None,
    extobj=None,
):
    # Perform the radian operation on the input array x
    ret = ivy.deg2rad(x, out=ret)
    
    # Check if where argument is an array and conditionally update ret array
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    
    return ret # Return result


# This is the hypotenuse function
# It returns the hypotenuse input using the ivy hypotenuse function.
@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
def _hypot(
    x1,
    x2,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="K",
    dtype=None,
    subok=True,
): 
    # Perform the hypotenuse operation on the input array x
    ret = ivy.hypot(x1, x2, out=out)
    
    # # Check if where argument is an array and conditionally update ret array
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.zeros_like(ret), out=out)
    
    return ret # Return result
