
import ivy
ivy.set_backend("torch")

x=ivy.array([[-0.8]])
offset=2
padding_value=4
# padding_value = ivy.astype(padding_value, ivy.dtype(x))

# d_x = ivy.diagflat(x, offset=offset, padding_value=padding_value)
# print(d_x)

# d_x = ivy.diag(x, k=offset)
# print(d_x)


xt = ivy.array([[0, 1, 2,8],
                [3, 4, 5,7],
                 [6, 7, 8,6]])
print(ivy.diag(xt, k=1))

# since the diagonal is backward_diagonal
# so the valid max offset values depend on the no of cols and not the no of rows.
# so the offset value will range from (0 to #cols-1)
# ie, offset should not be greater than cols-1
# ie, offset !> (x.shape[1] - 1)
# ie, if (offset > (x.shape[1] - 1)):
#         return ivy.array([])