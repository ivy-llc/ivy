# global
import abc

import ivy


class ArrayWithNormsExperimental(abc.ABC):

    def l2_normalize(self, axis=None, out=None):
        """
        Normalizes the array to have unit L2 norm.
        """
        return ivy.l2_normalize(self, axis=axis, out=out)


