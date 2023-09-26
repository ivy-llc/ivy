# import paddle
# import ivy.functional.frontends.paddle as paddle_bc

# data=paddle.to_tensor([[7]], dtype='float32')
# # scale=paddle.to_tensor([[3.1]], dtype='float32')
# # bias=paddle.to_tensor([[3.2]], dtype='float32')
# # add=paddle.add(data,bias)
# # mult=paddle.multiply(scale,add)
# # print("add=",add)
# # print("mult=",mult)
# res = paddle.scale(data, scale=3.1, bias=3.2, bias_after_scale=False)
# res_bc=paddle_bc.scale(data, scale=3.1, bias=3.2, bias_after_scale=False)

# print("value of ground truth=", res)

# print("value from backend=",res_bc)
