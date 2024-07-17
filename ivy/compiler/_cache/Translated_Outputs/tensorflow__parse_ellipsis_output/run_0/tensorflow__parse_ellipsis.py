def tensorflow__parse_ellipsis(so, ndims):
    pre = list()
    for s in so:
        if s is Ellipsis:
            break
        pre.append(s)
    post = list()
    for s in reversed(so):
        if s is Ellipsis:
            break
        post.append(s)
    ret = list(
        pre
        + [slice(None, None, None) for _ in range(ndims - len(pre) - len(post))]
        + list(reversed(post))
    )
    return ret, (len(pre), ndims - len(post))
