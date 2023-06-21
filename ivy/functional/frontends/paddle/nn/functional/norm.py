# local
import paddle.fluid as fluid


bn = fluid.layers.batch_norm

input_data = fluid.data(name='input', shape=[None, 10], dtype='float32')
output = bn(input_data, is_test=True)
print(output)
