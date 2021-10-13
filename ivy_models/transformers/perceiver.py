# global
import ivy


# Helper Classes #
# ---------------#

class PreNorm(ivy.Module):
    def __init__(self, dim, fn, context_dim=None, dev_str=None, v=None):
        self._fn = fn
        self._norm = ivy.LayerNorm([dim], dev_str=dev_str)
        if isinstance(context_dim, int):
            context_dim = [context_dim]
        self._norm_context = ivy.LayerNorm(context_dim, dev_str=dev_str) if ivy.exists(context_dim) else None
        ivy.Module.__init__(self, v=v, dev_str=dev_str)

    def _forward(self, x, **kwargs):
        x = self._norm(x)
        if ivy.exists(self._norm_context):
            kwargs.update(context=self._norm_context(kwargs['context']))
        return self._fn(x, **kwargs)


class FeedForward(ivy.Module):
    def __init__(self, dim, mult=4, dropout=0., dev_str=None, v=None):
        self._net = ivy.Sequential(
            ivy.Linear(dim, dim * mult * 2, dev_str=dev_str),
            ivy.GEGLU(),
            ivy.Dropout(dropout),
            ivy.Linear(dim * mult, dim, dev_str=dev_str),
            dev_str=dev_str)
        ivy.Module.__init__(self, v=v, dev_str=dev_str)

    def _forward(self, x):
        return self._net(x)

# Specification class #
# --------------------#

class PerceiverSpec(ivy.Container):

    def __init__(self,

                 # input-output dependent
                 num_input_dims,
                 num_input_axes,
                 num_classes,

                 # input-output agnostic
                 network_depth=6,
                 num_latents=512,
                 latent_dim=512,
                 num_cross_att_heads=1,
                 num_self_att_heads=8,
                 cross_head_dim=64,
                 self_head_dim=64,
                 weight_tie_layers=False,
                 attn_dropout=0.,
                 ff_dropout=0.,
                 num_self_att_per_cross_attn=1,
                 with_final_head=True,
                 fourier_encode_input=True,
                 num_fourier_freq_bands=6,
                 max_fourier_freq=None,
                 device=None
                 ):

        if fourier_encode_input and not ivy.exists(max_fourier_freq):
            raise Exception('The input-dependent max_fourier_freq must be specified when fourier_encode_input is set.')

        device = ivy.default(device, ivy.default_device())

        super().__init__(num_input_dims=num_input_dims,
                         num_input_axes=num_input_axes,
                         num_classes=num_classes,
                         network_depth=network_depth,
                         num_latents=num_latents,
                         latent_dim=latent_dim,
                         num_cross_att_heads=num_cross_att_heads,
                         num_self_att_heads=num_self_att_heads,
                         cross_head_dim=cross_head_dim,
                         self_head_dim=self_head_dim,
                         weight_tie_layers=weight_tie_layers,
                         attn_dropout=attn_dropout,
                         ff_dropout=ff_dropout,
                         num_self_att_per_cross_attn=num_self_att_per_cross_attn,
                         with_final_head=with_final_head,
                         fourier_encode_input=fourier_encode_input,
                         num_fourier_freq_bands=num_fourier_freq_bands,
                         max_fourier_freq=max_fourier_freq,
                         device=device)


# Main Class #
# -----------#

class Perceiver(ivy.Module):

    def __init__(self, spec: PerceiverSpec, v: ivy.Container = None):
        self._spec = spec
        super(Perceiver, self).__init__(v=v)

    # noinspection PyUnusedLocal
    def _build(self, *args, **kwargs):
        self._fourier_encode_input = self._spec.fourier_encode_input
        fourier_channels = (self._spec.num_input_axes * ((self._spec.num_fourier_freq_bands * 2) + 1)) \
            if self._spec.fourier_encode_input else 0
        num_input_dims = fourier_channels + self._spec.num_input_dims

        self._latents = ivy.variable(
            ivy.random_uniform(shape=(self._spec.num_latents, self._spec.latent_dim), dev_str=self._spec.device))

        get_cross_attn = lambda: PreNorm(
            self._spec.latent_dim, ivy.MultiHeadAttention(
                self._spec.latent_dim, self._spec.num_cross_att_heads, self._spec.cross_head_dim,
                self._spec.attn_dropout, num_input_dims, dev_str=self._spec.device), context_dim=num_input_dims,
            dev_str=self._spec.device)
        get_cross_ff = lambda: PreNorm(
            self._spec.latent_dim, FeedForward(self._spec.latent_dim, dropout=self._spec.ff_dropout,
                                               dev_str=self._spec.device), dev_str=self._spec.device)
        get_latent_attn = lambda: PreNorm(
            self._spec.latent_dim, ivy.MultiHeadAttention(
                self._spec.latent_dim, self._spec.num_self_att_heads, self._spec.self_head_dim, self._spec.attn_dropout,
                dev_str=self._spec.device), dev_str=self._spec.device)
        get_latent_ff = lambda: PreNorm(self._spec.latent_dim, FeedForward(
            self._spec.latent_dim, dropout=self._spec.ff_dropout, dev_str=self._spec.device), dev_str=self._spec.device)

        get_cross_attn_cached, get_cross_ff_cached, get_latent_attn_cached, get_latent_ff_cached =\
            map(ivy.cache_fn, (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))

        self._layers = list()
        for i in range(self._spec.network_depth):
            should_cache = i > 0 and self._spec.weight_tie_layers

            self_attns = list()

            for _ in range(self._spec.num_self_att_per_cross_attn):
                self_attns.append([
                    get_latent_attn_cached() if should_cache else get_latent_attn(),
                    get_latent_ff_cached() if should_cache else get_latent_ff(),
                ])

            self._layers.append([
                get_cross_attn_cached() if should_cache else get_cross_attn(),
                get_cross_ff_cached() if should_cache else get_cross_ff(),
                self_attns
            ])

        self._to_logits = ivy.Sequential(
            ivy.LayerNorm([self._spec.latent_dim], dev_str=self._spec.device),
            ivy.Linear(self._spec.latent_dim, self._spec.num_classes, dev_str=self._spec.device),
            dev_str=self._spec.device
        ) if self._spec.with_final_head else lambda x: x

    def _forward(self, data, mask=None):
        # noinspection PyTupleAssignmentBalance
        b, *axis, _ = data.shape
        assert len(axis) == self._spec.num_input_axes, 'input data must have the right number of axis'

        if self._fourier_encode_input:
            # calculate fourier encoded positions in the range of [-1, 1], for all axis
            axis_pos = list(map(lambda size: ivy.linspace(-1., 1., size, dev_str=self._dev_str), axis))
            pos = ivy.stack(ivy.meshgrid(*axis_pos), -1)
            enc_pos = ivy.fourier_encode(pos, self._spec.max_fourier_freq, self._spec.num_fourier_freq_bands)
            enc_pos = ivy.einops_rearrange(enc_pos, '... n d -> ... (n d)')
            enc_pos = ivy.einops_repeat(enc_pos, '... -> b ...', b=b)

            data = ivy.concatenate([data, enc_pos], -1)

        # concat to channels of data and flatten axis

        data = ivy.einops_rearrange(data, 'b ... d -> b (...) d')

        x = ivy.einops_repeat(self._latents, 'n d -> b n d', b=b)

        # layers

        for cross_attn, cross_ff, self_attns in self._layers:
            x = cross_attn(x, context=data, mask=mask) + x
            x = cross_ff(x) + x

            for self_attn, self_ff in self_attns:
                x = self_attn(x) + x
                x = self_ff(x) + x

        x = ivy.reduce_mean(x, -2)
        return self._to_logits(x)
