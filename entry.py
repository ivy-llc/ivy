import ivy
ivy.set_backend("jax")
x = ivy.array([[[[3, 4], [1, 8]], [[10, 0], [2, 0]]]])
print(ivy.flatten(x))
print(x)
