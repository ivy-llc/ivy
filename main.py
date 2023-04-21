import ivy 
import numpy as np 
import torch

ivy.set_backend("numpy")

# t1 = tf.Variable(np.array([[[200, 4, 5], [20, 5, 70]],[[2, 3, 5], [5, 5, 7]]]), dtype = tf.float32, name = 'lables')
# t2 = tf.Variable(np.array([1,2]), dtype = tf.float32, name = 'predictions')
# result = tf.cond(tf.convert_to_tensor(False), lambda: t1, lambda: t2)
i = 0
j=1
test_fn = lambda i, j: i < 3
body_fn = lambda i, j: (i + 1, j * 2)
result = ivy.while_loop(test_fn, body_fn, vars= (i,j))
print(result)
#ivy.array(2)
