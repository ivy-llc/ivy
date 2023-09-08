import ivy
x = ivy.array([1,2,3])
z = ivy.l2_loss(x)
print(z)
a = ivy.array([[1,2,3],[4,5,6]])
b = ivy.l2_loss(a)
print(b)
