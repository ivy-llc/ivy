# local
from typing import Union, Optional

# local
import ivy
from ivy.container.base import ContainerBase


# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class ContainerWithSearching(ContainerBase):
    pass

# implement container instance method
# implement container static method
# write docstring examples under newly created function
# (more of this explained in deep dive docs under docstring_examples.rst)

def argmax(
    x: Union[ivy.Array, ivy.NativeArray],
    axis: Optional[int] = None,
    keepdims: Optional[bool] = False,

):
    return ivy.Array.argmax(x, axis=axis, keepdims=keepdims)



    """
    
    ivy.<Container> <instance> method variant of ivy.argmax. 
    This method simply wraps the function, and so the docstring for ivy.argmax also applies to this method
    with minimal changes.
    
    Functional Examples
        --------

        With :code:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([-0., 1., -1.]))
        >>> y = ivy.argmax(x)
        >>> print(y)
        {
          a:ivy.array([1])
        }
        
        >>> x = ivy.array([-0., 1., -1.])
        >>> ivy.argmax(x,out=x)
        >>> print(x)
        ivy.array([1])

        >>> x = ivy.Container(a=ivy.array([1., -0., -1.]), b=ivy.array([-2., 3., 2.]))
        >>> y = ivy.argmax(x, axis= 1)
        >>> print(y)
        ivy.array([0, 1])
        
        >>> x = ivy.Container(a=ivy.array([-0., 1., -1.]))
        >>> y = ivy.argmax(x, axis)
        >>> print(y)
        a:ivy.array([1])
        
        >>> x = ivy.Container(a=ivy.array([-0., 1., -1.]))
        >>> y = ivy.argmax(x, axis, keepdims)
        >>> print(y)
        a:ivy.array([1])
        
        >>> x = ivy.Container(a=ivy.array([-0., 1., -1.]))
        >>> y = ivy.argmax(x, axis, keepdims = False)
        >>> print(y)
        a:ivy.array([1])
        
        >>> x = ivy.Container(a=ivy.array([4., 0., -1.]), b=ivy.array([2., -3., 6]))
        >>> y = ivy.argmax(x, axis= 1, keepdims = True)
        >>> print(y)
        ivy.array([[0], [2]])

        >>> x = ivy.Container(a=ivy.array([4., 0., -1.]), b=ivy.array([2., -3., 6]), c=ivy.array([2., -3., 6]))
        >>> z= ivy.zeros((1,3), dtype=ivy.int64)
         >>> y = ivy.argmax(x, axis= 1, keepdims= True, out= z)
        >>> print(y)
        {
          a:ivy.array([0])
          b:ivy.array([2])
          c:ivy.array([2])
        }
                   
        >>> x = ivy.Container(a=ivy.native_array([-0., 1., -1.]))
        >>> y = ivy.argmax(x)
        >>> print(y)
        ivy.array([1])
    
    
    """





def static_argmax(
    x: Union[ivy.Array, ivy.NativeArray],
    axis: Optional[int] = None,
    keepdims: Optional[bool] = False,

):
    return ivy.Array.argmax(x, axis=axis, keepdims=keepdims)


    """

    ivy.<Container> <static> method variant of ivy.argmax. 
    This method simply wraps the function, and so the docstring for ivy.argmax also applies to this method
    with minimal changes.
    
    Functional Examples
        --------

        With :code:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([-0., 1., -1.]))
        >>> y = ivy.Container.static_argmax(x)
        >>> print(y)
        {
          a:ivy.array([1])
        }
        
        >>> x = ivy.array([-0., 1., -1.])
        >>> ivy.Container.static_argmax(x,out=x)
        >>> print(x)
        ivy.array([1])

        >>> x = ivy.Container(a=ivy.array([1., -0., -1.]), b=ivy.array([-2., 3., 2.]))
        >>> y = ivy.Container.static_argmax(x, axis= 1)
        >>> print(y)
        ivy.array([0, 1])
        
        >>> x = ivy.Container(a=ivy.array([-0., 1., -1.]))
        >>> y = ivy.Container.static_argmax(x, axis)
        >>> print(y)
        a:ivy.array([1])
        
        >>> x = ivy.Container(a=ivy.array([-0., 1., -1.]))
        >>> y = ivy.Container.static_argmax(x, axis, keepdims)
        >>> print(y)
        a:ivy.array([1])
        
        >>> x = ivy.Container(a=ivy.array([-0., 1., -1.]))
        >>> y = ivy.Container.static_argmax(x, axis, keepdims = False)
        >>> print(y)
        a:ivy.array([1])
        
        >>> x = ivy.Container(a=ivy.array([4., 0., -1.]), b=ivy.array([2., -3., 6]))
        >>> y = ivy.Container.static_argmax(x, axis= 1, keepdims = True)
        >>> print(y)
        ivy.array([[0], [2]])

        >>> x = ivy.Container(a=ivy.array([4., 0., -1.]), b=ivy.array([2., -3., 6]), c=ivy.array([2., -3., 6]))
        >>> z= ivy.zeros((1,3), dtype=ivy.int64)
         >>> y = ivy.Container.static_argmax(x, axis= 1, keepdims= True, out= z)
        >>> print(y)
        {
          a:ivy.array([0])
          b:ivy.array([2])
          c:ivy.array([2])
        }
                   
        >>> x = ivy.Container(a=ivy.native_array([-0., 1., -1.]))
        >>> y = ivy.Container.static_argmax(x)
        >>> print(y)
        ivy.array([1])
        
        ##
        >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([3., 4., 5.]))
        >>> y = x.tan()
        >>> print(y)
        {
            a:ivy.array([0., 1.56, -2.19]),
            b:ivy.array([-0.143, 1.16, -3.38])
        }

    """




