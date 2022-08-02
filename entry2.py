import ivy
ivy.set_backend("numpy")
weights = ivy.Container(
    {'decoder':
         {'l0':
              {'b': ivy.array([0.]),
               'w': ivy.array([[0.]])},
          'l1':
              {'b': ivy.array([0.]),
               'w': ivy.array([[0.]])}},
     'encoder':
         {'l0':
              {'b': ivy.array([0.]),
               'w': ivy.array([[0.]])},
          'l1':
              {'b': ivy.array([0.]),
               'w': ivy.array([[0.]])}},
     'l0':
         {'b': ivy.array([0.]),
          'w': ivy.array([[0.]])},
     'l1':
         {'b': ivy.array([0.]),
          'w': ivy.array([[0.]])}})

print(weights.flatten_key_chains(below_depth=1))