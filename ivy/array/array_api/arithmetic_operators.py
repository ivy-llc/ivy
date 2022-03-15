# global
import abc

# ToDo: implement all Array API arithmetic operators here


class ArrayWithArrayAPIArithmeticOperators(abc.ABC):
    def __neg__(x: Union[ivy.Array, ivy.NativeArray],
            axes: Union[int, Tuple[int], List[int]],
            dtype: Optional[Union[ivy.Dtype, str]] = None,
            dev: Optional[Union[ivy.Dev, str]] = None) \
        -> ivy.Array:
        
    
    '''returns elementwise negative'''
    
         def call(*args, **kargs):
            
        
     '''we can use __call__ to create a new instance
        of this class, with the help of any constructor'''

    
            returns _cur_framework(x).__neg__(operator.neg,*args, **kargs)
        

    
                               
           
            
    
