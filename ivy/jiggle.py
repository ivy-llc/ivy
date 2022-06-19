import ivy

ivy.set_backend("torch")
x = ivy.Container(
    a=ivy.array([[1.0, 0.0, 3.0], [2.0, 3.0, 4.0]]),
    b=ivy.array([[5.0, 6.0, 7.0], [3.0, 4.0, 8.0]]),
)
y = ivy.Container(
    a=ivy.array([[2.0, 4.0, 5.0], [9.0, 10.0, 6.0]]),
    b=ivy.array([[1.0, 0.0, 3.0], [2.0, 3.0, 4.0]]),
)

ivy.Container.static_tensordot(x, y)
