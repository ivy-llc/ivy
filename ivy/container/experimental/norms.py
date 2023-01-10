from ivy.container.base import ContainerBase


class ContainerWithNormsExperimental(ContainerBase):

    @staticmethod
    def static_l2_normalize(x, axis=None, out=None):
        """
        Normalizes the array to have unit L2 norm.
        """
        return ContainerBase.cont_multi_map_in_function(
            "l2_normalize",
            x,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def l2_normalize(self, axis=None, out=None):

        return self.static_l2_normalize(self, axis=axis, out=out)
