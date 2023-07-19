from ivy.functional.ivy.experimental import put_along_axis
import ivy
import paddle

ivy.set_backend("paddle")

x = paddle.to_tensor([[10, 30, 20], [60, 40, 50]])
index = paddle.to_tensor([[1], [0]])
value = 99
axis = 0

res = put_along_axis(x, index, value, axis, mode="assign")
# res = paddle.put_along_axis(x, index, value, axis, reduce="assign")

print(res)
