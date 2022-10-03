import ivy

ivy.set_framework('torch')

x = ivy.ivy.array([-1])
y = ivy.to_scalar(x)
print(y)