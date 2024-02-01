import ivy

ivy.set_backend("torch")

x = ivy.array([-1.0, 0.2, 1.0])
y = x.hardtanh()
print(y)
