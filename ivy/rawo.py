# # import ivy.functional.frontends.tensorflow as tf_frontend
# import ivy
# # import copy

# # def wrap_raw_ops_alias(fn: callable) -> callable:
# #     def _wraped_fn(*args, **kwargs):
# #         kwargs.update(zip(fn.__code__.co_varnames, args))
# #         return fn(**kwargs)

# #     return _wraped_fn

# # # n =  ivy.array(1)
# # # m = ivy.array(2)
# # # x = wrap_raw_ops_alias(tf_frontend.math.add)

# # # print(x(m, n))

# def dictup(dict1:dict, dict2: dict):
#     print(dict1.keys())
#     dict3 = copy.deepcopy(dict1)
#     for key, val in dict2.items():
#         for k in dict1.keys():
#             if key == k:
#                 temp_key = dict3[k]
#                 del dict3[k]
#                 dict3[val] = temp_key
# #     print(dict3.keys())

# # def rename(old_dict,old_name,new_name):
# #     new_dict = {}
# #     for key in old_dict.keys():
# #         new_key = key if key != old_name else new_name
# #         new_dict[new_key] = old_dict[key]
# #     return new_dict

# # def update_kwarg_keys(kwargs, to_update):
# #     new_dict = {}
# #     for key,value in to_update.items():
# #         for k in kwargs.keys():
# #             if k == key:
# #                 new_dict=rename(kwargs, k, value)
# #         kwargs = new_dict
# #     return new_dict

import ivy

# n = ivy.array(1)
# m = n
# ret = ivy.functional.frontends.tensorflow.raw_ops.Add(k=n, y=m)
# # x = ivy.array([1, 10, 26.9, 2.8, 166.32, 62.3])
# # ret = ivy.functional.frontends.tensorflow.raw_ops.ArgMax(input=x, dimension=0)
# print(ret)

curr_dict = {"x": ivy.array(2), "y": ivy.array(2)}
upd = ivy.functional.frontends.tensorflow.func_wrapper.update_kwarg_keys(
    {"x": "jk", "y": "pk"}, curr_dict
)
print(upd)
