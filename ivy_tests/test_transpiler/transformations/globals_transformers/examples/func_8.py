from .func_6 import ClippedGELUActivation, GeluActivation, ACT2FN

"""
helpers
"""


def get_act_fn(name):
    if name == "gelu":
        return GeluActivation()
    elif name == "gelu_10":
        return ClippedGELUActivation(min=-10.0, max=10.0)
    else:
        return None


def get_act_fn2(name):
    if name in ACT2FN:
        return ACT2FN[name]
    else:
        return None


class get_act_Cls:
    def find(self, name):
        if name == "gelu":
            return GeluActivation()
        elif name == "gelu_10":
            return ClippedGELUActivation(min=-10.0, max=10.0)
        else:
            return None


class get_act_Cls2:
    def find(self, name):
        if name in ACT2FN:
            return ACT2FN[name]
        else:
            return None


"""
examples
"""


def fn_with_cached_glob_used_as_obj(name):
    if "gelu" in ACT2FN:  # GeluActivation etc.. used as globals in mod1
        return ACT2FN[name]
    else:
        return get_act_fn(name)  # GeluActivation etc.. used as regular object in mod2


def fn_with_cached_glob_used_as_glob(name):
    if "gelu" in ACT2FN:  # GeluActivation etc.. used as globals in mod1
        return ACT2FN[name]
    else:
        return get_act_fn2(name)  # GeluActivation etc.. used as globals in mod2


class Class_with_cached_glob_used_as_obj:

    def get_activation(self, name):
        if "gelu" in ACT2FN:  # GeluActivation etc.. used as globals in mod1
            return ACT2FN[name]
        else:
            return get_act_Cls().find(
                name
            )  # GeluActivation etc.. used as regular object in mod2


class Class_with_cached_glob_used_as_glob:

    def get_activation(self, name):
        if "gelu" in ACT2FN:  # GeluActivation etc.. used as globals in mod1
            return ACT2FN[name]
        else:
            return get_act_Cls2().find(
                name
            )  # GeluActivation etc.. used as globals in mod2
