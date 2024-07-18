def tensorflow__reverse_repeat_tuple(t, n):
    return tuple(x for x in reversed(t) for _ in range(n))
