import ivy


def det(input, name=None):
    return ivy.det(input)


det.unsupported_dtypes = ("float16", "bfloat16")

<<<<<<< HEAD



def eigvalsh(tensor, name=None):
    return ivy.eigvalsh(tensor)


eigvalsh.unsupported_dtypes = ("float16", "bfloat16")
=======
>>>>>>> 9f4596f592aa614b5a5d528a3418203660fb9146
