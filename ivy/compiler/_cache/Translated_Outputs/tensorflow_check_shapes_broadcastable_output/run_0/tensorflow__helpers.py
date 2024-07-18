def tensorflow_check_one_way_broadcastable(x1, x2):
    if len(x1) > len(x2):
        return False
    for a, b in zip(x1[::-1], x2[::-1]):
        if a in (1, b):
            pass
        else:
            return False
    return True
