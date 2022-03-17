# global
import abc

# ToDo: implement all Array API arithmetic operators here


class ArrayWithArrayAPIArithmeticOperators(abc.ABC):
    def __neg__(x: torch.tensor)\
       --> torch.tensor:
        return[torch.__neg__(x)]
    
    def __neg__(x:tensor)\
    --> Tensor:
        return tf.numpy.__neg__(x)
    
    def __neg__(x: jaxArray)/
    --> jaxArray:
        return jax.numpy.__neg__(x)
    
    def __neg__(x: np.ndarray)/
      ---> np.ndarray:
            return -np.ndarray(x)
    
    
    
