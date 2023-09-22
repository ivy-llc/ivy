import paddle
import ivy.functional.frontends.paddle as paddle_ivy

data = paddle.to_tensor([[7]], dtype="int8")

fac = paddle.to_tensor([[3.1]], dtype="float32")

res = paddle.scale(data, scale=fac, bias=3, bias_after_scale=False)
res_ivy = paddle_ivy.scale(data, scale=fac, bias=3, bias_after_scale=False)

print("groundtruth=", res)

print("ivy paddle=", res_ivy)
