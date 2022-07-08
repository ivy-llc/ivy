# global
import ivy


class PreNorm(ivy.Module):
    def __init__(self, dim, fn, context_dim=None, epsilon=None, device=None, v=None):
        self._fn = fn
        self._norm = ivy.LayerNorm([dim], epsilon=epsilon, device=device)
        if isinstance(context_dim, int):
            context_dim = [context_dim]
        self._norm_context = ivy.LayerNorm(context_dim, epsilon=epsilon, device=device) if \
            ivy.exists(context_dim) else None
        ivy.Module.__init__(self, v=v, device=device)

    def _forward(self, x, **kwargs):
        x = self._norm(x)
        if ivy.exists(self._norm_context):
            kwargs.update(context=self._norm_context(kwargs['context']))
        return self._fn(x, **kwargs)


class FeedForward(ivy.Module):
    def __init__(self, dim, dropout=0., device=None, v=None):
        self._net = ivy.Sequential(
            ivy.Linear(dim, dim, device=device),
            ivy.GELU(),
            ivy.Linear(dim, dim, device=device),
            ivy.Dropout(dropout),
            device=device
        )
        ivy.Module.__init__(self, v=v)

    def _forward(self, x):
        return self._net(x)
