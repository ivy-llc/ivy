from .tensorflow__helpers import tensorflow_split_1


def tensorflow_retrieve_object(frame, name):
    if name is None:
        return name
    names = tensorflow_split_1(name, ".")
    obj = frame.f_locals.get(names[0]) or frame.f_globals.get(names[0])
    if obj is None:
        return None
    for attr in names[1:]:
        try:
            obj = getattr(obj, attr)
        except AttributeError:
            return None
    return obj
