# global

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

    def all_equal(
        self,
        x,
        equality_matrix: bool = False
    ):
        """
        ivy.Container instance method variant of ivy.all_equal. This method simply wraps the
        function, and so the docstring for ivy.all_equal also applies to this method
        with minimal changes.

        Examples
        --------

        With one :code:`ivy.Container` instances:

        >>> x1 = ivy.Container(a=ivy.array([1, 0, 1, 1]), b=ivy.array([1, -1, 0, 0]))
        >>> x2 = ivy.array([1, 0, 1, 1])
        >>> y = x1.all_equal(x2, equality_matrix= False)
        >>> print(y)
        {
            a: true,
            b: false
        }

        >>> x1 = ivy.Container(a=ivy.array([1, 0, 1, 1]), b=ivy.array([1, -1, 0, 0]))
        >>> x2 = ivy.array([1, 0, 1, 1])
        >>> y = ivy.Container.equal(x1, x2)
        >>> print(y)
        {
            a: ivy.array([True, True, True, True]),
            b: ivy.array([True, False, False, False])
        }

        With multiple :code:`ivy.Container` instances:

        >>> x1 = ivy.Container(a=ivy.native_array([1, 0, 0]),\
                                b=ivy.array([1, 2, 3]))
        >>> x2 = ivy.Container(a=ivy.native_array([1, 0, 1]),\
                                b=ivy.array([1, 2, 3]))
        >>> y = x1.all_equal(x2, equality_matrix= False)
        >>> print(y)
        {
            a: false,
            b: true
        }

        >>> x1 = ivy.Container(a=ivy.native_array([1, 0, 0]),\
                                b=ivy.array([1, 2, 3]))
        >>> x2 = ivy.Container(a=ivy.native_array([1, 0, 1]),\
                                b=ivy.array([1, 2, 3]))
        >>> y = ivy.Container.equal(x1, x2)
        >>> print(y)
        {
            a: ivy.array([True, True, False]),
            b: ivy.array([True, True, True])
        }

        Static Method Examples
        ----------------------

        With one :code:`ivy.Container` input:

        >>> x1 = ivy.Container(a=ivy.array([1, 0, 1, 1]), b=ivy.array([1, -1, 0, 0]))
        >>> x2 = ivy.array([1, 0, 1, 1])
        >>> y = ivy.all_equal(x1, x2, equality_matrix= False)
        >>> print(y)
        {
            a: true,
            b: false
        }

        >>> x1 = ivy.Container(a=ivy.array([1, 0, 1, 1]), b=ivy.array([1, -1, 0, 0]))
        >>> x2 = ivy.array([1, 0, 1, 1])
        >>> y = ivy.Container.equal(x1, x2)
        >>> print(y)
        {
            a: ivy.array([True, True, True, True]),
            b: ivy.array([True, False, False, False])
        }

        With multiple :code:`ivy.Container` input:

        >>> x1 = ivy.Container(a=ivy.array([1, 0, 1, 1]), \
                                b=ivy.native_array([1, 0, 0, 1]))
        >>> x2 = ivy.Container(a=ivy.native_array([1, 0, 1, 1]), \
                                b=ivy.array([1, 0, -1, -1]))
        >>> y = ivy.all_equal(x1, x2, equality_matrix=False)
        >>> print(y)
        {
            a: true,
            b: false
        }

        >>> x1 = ivy.Container(a=ivy.array([1, 0, 1, 1]), \
                                b=ivy.native_array([1, 0, 0, 1]))
        >>> x2 = ivy.Container(a=ivy.native_array([1, 0, 1, 1]), \
                                b=ivy.array([1, 0, -1, -1]))
        >>> y = ivy.Container.equal(x1, x2)
        >>> print(y)
        {
            a: ivy.array([True, True, True, True]),
            b: ivy.array([True, True, False, False])
        }

        Container Static Method Examples
        --------------------------------

        With one :code:`ivy.Container` input:

        >>> x1 = ivy.Container(a=ivy.array([1, 0, 1, 1]), b=ivy.array([1, -1, 0, 0]))
        >>> x2 = ivy.array([1, 0, 1, 1])
        >>> y = ivy.Container.equal(x1, x2)
        >>> print(y)
        {
            a: ivy.array([True, True, True, True]),
            b: ivy.array([True, False, False, False])
        }

        With multiple :code:`ivy.Container` input:

        >>> x1 = ivy.Container(a=ivy.array([1, 0, 1, 1]), \
                                b=ivy.native_array([1, 0, 0, 1]))
        >>> x2 = ivy.Container(a=ivy.native_array([1, 0, 1, 1]), \
                                b=ivy.array([1, 0, -1, -1]))
        >>> y = ivy.Container.equal(x1, x2)
        >>> print(y)
        {
            a: ivy.array([True, True, True, True]),
            b: ivy.array([True, True, False, False])
        }

        """
        return ivy.all_equal(self, x, equality_matrix)
