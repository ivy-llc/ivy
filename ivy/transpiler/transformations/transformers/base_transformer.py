import gast


__all__ = []


class BaseTransformer(gast.NodeTransformer):
    def visit(self, node):
        if not isinstance(node, gast.AST):
            msg = f'Expected "gast.AST", but got "{type(node)}".'
            raise ValueError(msg)

        from ivy.transpiler.utils.origin_utils import ORIGI_INFO

        origin_info = getattr(node, ORIGI_INFO, None)

        result = super().visit(node)

        iter_result = result
        if iter_result is not node and iter_result is not None:
            if not isinstance(iter_result, (list, tuple)):
                iter_result = (iter_result,)
            if origin_info is not None:
                for n in iter_result:
                    setattr(n, ORIGI_INFO, origin_info)

        return result
