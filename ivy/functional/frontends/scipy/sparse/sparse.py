import ivy


def isspmatrix_csr(x):
    if ivy.is_ivy_sparse_array(x):
        if x._format == "csr":
            return True

    return False
