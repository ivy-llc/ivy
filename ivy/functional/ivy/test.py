x = ivy.array([0.0, 1.0, 2.0])
y = ivy.clip_vector_norm(x, 2.0)
print(y)
