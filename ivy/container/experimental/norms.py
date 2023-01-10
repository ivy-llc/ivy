from ivy.container.base import ContainerBase
from typing import Union, List, Dict, Optional
import ivy

class ContainerWithNormsExperimental(ContainerBase):

    @staticmethod
    def static_l2_normalize(x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
                            axis: int = None,
                            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
                            to_apply: bool = True,
                            prune_unapplied: bool = False,
                            map_sequences: bool = False,
                            out=None) -> ivy.Container:
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

    def l2_normalize(
            self,
            axis=None,
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            map_sequences: bool = False,
            out=None) -> ivy.Container:

        return self.static_l2_normalize(self,
                                        axis=axis,
                                        key_chains=key_chains,
                                        to_apply=to_apply,
                                        prune_unapplied=prune_unapplied,
                                        map_sequences=map_sequences,
                                        out=out)
