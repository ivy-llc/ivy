import ivy

x = ivy.Container(
    a=ivy.array([[1.0, 2.0], [3.0, 4.0]]), b=ivy.array([[1.0, 2.0], [2.0, 1.0]])
)
print(x.slogdet())
