# local
from ivy.container.base import ContainerBase
import ivy

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class ContainerWithGeneral(ContainerBase):
    def clip_vector_norm(
        self,
        max_norm,
        p,
        global_norm=False,
        key_chains=None,
        to_apply=True,
        prune_unapplied=False,
        map_sequences=False,
        out=None,
    ):
        max_norm_is_container = isinstance(max_norm, ivy.Container)
        p_is_container = isinstance(p, ivy.Container)
        if global_norm:
            if max_norm_is_container or p_is_container:
                raise Exception(
                    """global_norm can only be computed for 
                    scalar max_norm and p_val arguments,"""
                    "but found {} and {} of type {} and {} respectively".format(
                        max_norm, p, type(max_norm), type(p)
                    )
                )
            vector_norm = self.vector_norm(p, global_norm=True)
            ratio = max_norm / vector_norm
            if ratio < 1:
                return self.handle_inplace(self * ratio, out)
            return self.handle_inplace(self.copy(), out)
        return self.handle_inplace(
            self.map(
                lambda x, kc: self._ivy.clip_vector_norm(
                    x,
                    max_norm[kc] if max_norm_is_container else max_norm,
                    p[kc] if p_is_container else p,
                )
                if self._ivy.is_native_array(x) or isinstance(x, ivy.Array)
                else x,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
            out,
        )
