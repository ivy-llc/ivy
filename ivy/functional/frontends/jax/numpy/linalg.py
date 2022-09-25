import ivy


def eigvalsh(a, UPLO="L"):
    return ivy.eigvalsh(a, UPLO=UPLO)
