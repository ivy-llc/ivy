from .tensorflow__helpers import tensorflow_check_all


def tensorflow_check_all_or_any_fn(
    *args,
    fn,
    type="all",
    limit=(0,),
    message="args must exist according to type and limit given",
    as_array=True,
):
    if type == "all":
        tensorflow_check_all([fn(arg) for arg in args], message, as_array=as_array)
    elif type == "any":
        count = 0
        for arg in args:
            count = count + 1 if fn(arg) else count
        if count not in limit:
            raise Exception(message)
    else:
        raise Exception("type must be all or any")
