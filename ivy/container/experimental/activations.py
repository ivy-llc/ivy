# global
from typing import Union, Optional

# local
import ivy
from ivy.container.base import ContainerBase


class ContainerWithActivationExperimental(ContainerBase):
    @staticmethod
    def static_logit(
            x: Union[float, int, ivy.Container],
            /,
            *,
            eps: Optional[float] = None,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.cont_multi_map_in_function(
            "logit",
            x,
            eps=eps,
            out=out,
        )

    def logit(
            self: Union[float, int, ivy.Container],
            /,
            *,
            eps: Optional[float] = None,
            out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_logit(
            self,
            eps=eps,
            out=out)
