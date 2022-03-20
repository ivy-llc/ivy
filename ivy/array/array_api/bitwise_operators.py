# global
import numpy as np 
import tensorflow as tf 
import jax 
import jax.numpy as jnp
import torch
from typing import union, Optional, Tuple, Iterable 
from ivy.framework_handler import current_framework as cur_framework 

# ToDo: implement all Array API attributes here
#local 
from ivy import inf 
from ivy.functional.backends.jax import jaxArray
from collection import namedtuple 

    def __or__(X: np.ndarray)/
      -->np.ndarray:
            
            return np.bitwise_or_x, a, b)
            
            def __or__(x: jnp.ndarray)\
            --->jnp.ndarray:
                return jnp.__bitwisw_or__(x, a, b)
            
            def __or__(x: tensor.torch)\
            --> tensor.torch:
                return torch.__bitwise_or__(x, a, b)
            
            def __or__(x: tensor)\
            --> tensor:
                return tf.experimental.numpy.__bitwise_or__(x, a,b)
            
            
                
            
            
            
        
            
      
