import ivy


def cholesky(x, /, *, symmetrize_input=True):
    def symmetrize(x):
        # TODO : Take Hermitian transpose after complex numbers added
        return (x + ivy.swapaxes(x, -1, -2)) / 2

    if symmetrize_input:
        x = symmetrize(x)

    return ivy.cholesky(x)


def triangular_solve(a, b, *,
                     left_side=False, lower=False, transpose_a=False,
                     conjugate_a=False, unit_diagonal=False):
    def solveUpperTriangularMatrix(R, b):
        fR, fb = [], []
        for x, line in enumerate(R):
            fLine = []
            for y, el in enumerate(line):
                value = el
                fLine.append(value)
            fR.append(fLine)
        for el in b:
            fb.append(el)
        x = [0] * len(b)
        for step in range(len(b) - 1, 0 - 1, -1):
            if fR[step][step] == 0:
                if fb[step] != 0:
                    return "No solution"
                else:
                    return "Infinity solutions"
            else:
                x[step] = fb[step] / fR[step][step]
            for row in range(step - 1, 0 - 1, -1):
                fb[row] -= fR[row][step] * x[step]
        return x

    a_list = []
    for a_1 in list(a):
        a_list.append(list(a_1))
    result_temp = solveUpperTriangularMatrix(a_list, list(b))
    result_list = []
    for result in result_temp:
        result_list.append(float(result))
    ret = ivy.array(result_list, dtype=result.dtype)
    return ret
