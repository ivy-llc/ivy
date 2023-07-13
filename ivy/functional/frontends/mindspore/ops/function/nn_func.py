"""Includes Mindspore Frontend functions listed in the TODO list
https://github.com/unifyai/ivy/issues/14951."""

import ivy
@with_supported_dtypes({"2.0 and below": ("float16", "float32")}, "mindspore")
@to_ivy_arrays_and_back
def sigmoid(x):
    return ivy.sigmoid(x)



# test cases
x1 = ivy.array([-1.0, 0.0, 1.0])
y1 = sigmoid(x1)
print(y1)

x2 = ivy.array([-10.0, -5.0, 0.0, 5.0, 10.0])
y2 = ivy.sigmoid(x2)
print(y2)

x3 = ivy.array([0.5, 1.0, 1.5, 2.0, 2.5])
y3 = ivy.sigmoid(x3)
print(y3)

x4 = ivy.array([-100.0, -50.0, 0.0, 50.0, 100.0])
y4 = ivy.sigmoid(x4)
print(y4)

x5 = ivy.array([0.0, 0.1, 0.5, 0.9, 1.0])
y5 = ivy.sigmoid(x5)
print(y5)





