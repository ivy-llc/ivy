import ivy
for fw in ['tensorflow', 'torch', 'jax', 'numpy']:
    ivy.set_backend(fw)
    print(fw)
    weights = ivy.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    indices = ivy.array([0, 2])
    print(ivy.embedding(weights, indices, max_norm=2))
    ivy.unset_backend()