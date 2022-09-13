import ivy


def qr(x, *, full_matrices=True):
    mode = "complete" if full_matrices else "reduced"
    return ivy.qr(x, mode=mode)

qr.unsupported_dtypes=("float16", "bfloat16")