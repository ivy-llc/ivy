x = ivy.array([0., 1., 2.])
y = ivy.clip_vector_norm(x, 2.0)
print(y)