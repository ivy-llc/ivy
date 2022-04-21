# global
from typing import Optional, Union, List, Dict

# local
import ivy
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor,PyMethodParameters
class ContainerWithElementwise(ContainerBase):

    def abs(x: ContainerBase,
            key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
            to_apply: bool = True,
            prune_unapplied: bool = False,
            out: Optional[ContainerBase] = None) \
            -> ContainerBase:
        return x.handle_inplace(
            x.map(lambda x_, _: ivy.abs(x_) if ivy.is_array(x_) else x_, key_chains, to_apply, prune_unapplied), out)
