import ivy

ivy.set_backend("paddle")
def f(x): return ivy.mean(2*x**4)
g = ivy.grad
c = ivy.array([5.])
a = g(g(g(f)))
print(a(c))
# def f(x): return 3*(x**3)+2*x
# a = g(g((f)))
# print(a(c))
# a = g(g(ivy.sin))
# print(a(c))