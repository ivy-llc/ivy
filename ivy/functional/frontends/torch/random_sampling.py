import ivy


def multinomial(input, num_samples, replacement=False, *,
                generator=None, out=None):
    return ivy.multinomial(input, num_samples, replace=replacement, out=out)
