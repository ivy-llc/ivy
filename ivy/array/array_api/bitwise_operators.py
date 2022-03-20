# global
import abc
import ivy 
from ivy.framework_handler import current_framework as cur_framework 

# ToDo: implement all Array API attributes here


class ArrayWithArrayAPIBitwiseOperators(abc.ABC):
    def __or__(X: Union[ivy.Array, ivy.NativeArray]):
        
        return __cur__framework(x). __bitwise_or__(x)
