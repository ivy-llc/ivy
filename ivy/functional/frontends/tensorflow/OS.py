import ivy
import tensorflow

@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def linear(x):
        x: Union[ivy.Array, ivy.NativeArray],
        
    
      
        return ivy.current_framework(x,f=tensorflow).keras.activation.linear(x)
    
    
 @to_native_arrays_and_back
@handle_out_argument
@handle_nestable  
def exponential(x):
    x: Union[ivy.Array, ivy.NativeArray],
    return ivy.current_framework(x,f=tensorflow).keras.activation.exponential(x) 
   

@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def hard_sigmoid(x):
    
    
    x: Union[ivy.Array, ivy.NativeArray],
    return ivy.current_framework(x,f=tensorflow).keras.activation.hard_sigmoid(x) 
   
    