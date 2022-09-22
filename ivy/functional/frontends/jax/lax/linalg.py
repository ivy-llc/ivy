import ivy
from typing import Union,Optional

def cholesky(x, /, *, symmetrize_input=True):
    def symmetrize(x):
        # TODO : Take Hermitian transpose after complex numbers added
        return (x + ivy.swapaxes(x, -1, -2)) / 2

    if symmetrize_input:
        x = symmetrize(x)

    return ivy.cholesky(x)





def matrix_power(m: Union[ivy.Array, ivy.NativeArray],
                 n: int,
                 /,
                 *,
                 out: Optional[ivy.Array] = None) -> ivy.Array:
    return ivy.matrix_power(m, n, out=out)



