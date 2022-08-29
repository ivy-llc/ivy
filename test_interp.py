import ivy
ivy.set_backend('numpy')
xp = ivy.array([0, 1, 2])
yp = ivy.array([2, 3, 4])
i = -5
while i < 5:
    print(i, ivy.functional.frontends.numpy.interp(i, xp, yp, period=3))
    i += 0.5