import ivy
ivy.set_backend("numpy")

x = ivy.array([ivy.array([]), ivy.array([])])
# print(ivy.to_scalar(x))

y = (ivy.array([]), ivy.array([]))
z = (ivy.array([0]), ivy.array([0]))
print(ivy.shape(z[0]))
print(ivy.to_scalar(ivy.shape(z[0], as_array=True)))
# print(ivy.to_scalar(ivy.any(z)))