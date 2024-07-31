def Translated__addindent(s_, numSpaces):
    s = s_.split("\n")
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * " " + line) for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s
