import ivy


def cholesky(x, /, *, symmetrize_input=True):
    def symmetrize(x):
        # TODO : Take Hermitian transpose after complex numbers added
        return (x + ivy.swapaxes(x, -1, -2)) / 2

    if symmetrize_input:
        x = symmetrize(x)

    return ivy.cholesky(x)


def qr(x, /, *, full_matrices=True):
    mode = "complete" if full_matrices else "reduced"
    return ivy.qr(x, mode=mode)


qr.unsupported_dtypes = ("float16", "bfloat16")
