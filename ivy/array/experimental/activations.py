# global
import abc
import ivy


class ArrayWithActivationsExperimental(abc.ABC):
    def logit(self, /, *, eps=None, out=None):
        return ivy.logit(self, eps=eps, out=out)
