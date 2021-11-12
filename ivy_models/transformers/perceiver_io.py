# global
import ivy

# local
from ivy_models.transformers.helpers import PreNorm, FeedForward


# Specification class #
# --------------------#

class PerceiverIOSpec(ivy.Container):

    def __init__(self,

                 # input-output dependent
                 input_dim,
                 num_input_axes,
                 output_dim,

                 # input-output agnostic
                 queries_dim=1024,
                 network_depth=6,
                 num_latents=512,
                 latent_dim=512,
                 num_cross_att_heads=1,
                 num_self_att_heads=8,
                 cross_head_dim=64,
                 latent_head_dim=64,
                 weight_tie_layers=False,
                 learn_query=False,
                 query_shape=None,
                 attn_dropout=0.,
                 fc_dropout=0.,
                 num_self_att_per_cross_attn=1,
                 with_decoder=True,
                 with_final_head=True,
                 fourier_encode_input=True,
                 num_fourier_freq_bands=6,
                 max_fourier_freq=None,
                 device=None
                 ):

        if fourier_encode_input and not ivy.exists(max_fourier_freq):
            raise Exception('The input-dependent max_fourier_freq must be specified when fourier_encode_input is set.')

        if learn_query and not ivy.exists(query_shape):
            raise Exception('if learn_query is set, then query_shape must be specified.')

        device = ivy.default(device, ivy.default_device())

        super().__init__(input_dim=input_dim,
                         num_input_axes=num_input_axes,
                         output_dim=output_dim,
                         queries_dim=queries_dim,
                         network_depth=network_depth,
                         num_latents=num_latents,
                         latent_dim=latent_dim,
                         num_cross_att_heads=num_cross_att_heads,
                         num_self_att_heads=num_self_att_heads,
                         cross_head_dim=cross_head_dim,
                         latent_head_dim=latent_head_dim,
                         weight_tie_layers=weight_tie_layers,
                         learn_query=learn_query,
                         query_shape=query_shape,
                         attn_dropout=attn_dropout,
                         fc_dropout=fc_dropout,
                         num_self_att_per_cross_attn=num_self_att_per_cross_attn,
                         with_decoder=with_decoder,
                         with_final_head=with_final_head,
                         fourier_encode_input=fourier_encode_input,
                         num_fourier_freq_bands=num_fourier_freq_bands,
                         max_fourier_freq=max_fourier_freq,
                         device=device)


# Main Class #
# -----------#

class PerceiverIO(ivy.Module):

    def __init__(self, spec: PerceiverIOSpec, v: ivy.Container = None):
        self._spec = spec
        ivy.Module.__init__(self, v=v)

    # noinspection PyUnusedLocal
    def _build(self, *args, **kwargs):
        self._fourier_encode_input = self._spec.fourier_encode_input
        fourier_channels = (self._spec.num_input_axes * ((self._spec.num_fourier_freq_bands * 2) + 1)) \
            if self._spec.fourier_encode_input else 0
        input_dim = fourier_channels + self._spec.input_dim

        self._latents = ivy.variable(
            ivy.random_uniform(shape=(self._spec.num_latents, self._spec.latent_dim), dev_str=self._spec.device))

        # ToDo: set the correct initializatin scheme for the query here
        self._queries = ivy.variable(ivy.random_uniform(shape=self._spec.query_shape + [self._spec.queries_dim]))\
            if self._spec.learn_query else None

        get_cross_attn = lambda: PreNorm(
            self._spec.latent_dim, ivy.MultiHeadAttention(
                self._spec.latent_dim, self._spec.num_cross_att_heads, self._spec.cross_head_dim,
                self._spec.attn_dropout, input_dim, dev_str=self._spec.device), context_dim=input_dim,
            dev_str=self._spec.device)
        get_cross_fc = lambda: PreNorm(
            self._spec.latent_dim, FeedForward(self._spec.latent_dim, dropout=self._spec.fc_dropout,
                                               dev_str=self._spec.device), dev_str=self._spec.device)
        get_latent_attn = lambda: PreNorm(
            self._spec.latent_dim, ivy.MultiHeadAttention(
                self._spec.latent_dim, self._spec.num_self_att_heads, self._spec.latent_head_dim, self._spec.attn_dropout,
                dev_str=self._spec.device), dev_str=self._spec.device)
        get_latent_fc = lambda: PreNorm(self._spec.latent_dim, FeedForward(
            self._spec.latent_dim, dropout=self._spec.fc_dropout, dev_str=self._spec.device), dev_str=self._spec.device)

        get_cross_attn_cached, get_cross_fc_cached, get_latent_attn_cached, get_latent_fc_cached =\
            map(ivy.cache_fn, (get_cross_attn, get_cross_fc, get_latent_attn, get_latent_fc))

        self._layers = list()
        for i in range(self._spec.network_depth):
            should_cache = i > 0 and self._spec.weight_tie_layers

            self_attns = list()

            for _ in range(self._spec.num_self_att_per_cross_attn):
                self_attns.append([
                    get_latent_attn_cached() if should_cache else get_latent_attn(),
                    get_latent_fc_cached() if should_cache else get_latent_fc(),
                ])

            self._layers.append([
                get_cross_attn_cached() if should_cache else get_cross_attn(),
                get_cross_fc_cached() if should_cache else get_cross_fc(),
                self_attns
            ])

        self._decoder_cross_attn = PreNorm(self._spec.queries_dim, ivy.MultiHeadAttention(
            self._spec.queries_dim, self._spec.num_cross_att_heads, self._spec.cross_head_dim,
            context_dim=self._spec.latent_dim), context_dim = self._spec.latent_dim)
        self._decoder = PreNorm(self._spec.queries_dim, FeedForward(self._spec.queries_dim))\
            if self._spec.with_decoder else None

        self._to_logits = ivy.Linear(self._spec.queries_dim, self._spec.output_dim, dev_str=self._spec.device)\
            if self._spec.with_final_head else lambda x: x

    def _forward(self, data, mask=None, queries=None):
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

        for cross_attn, cross_fc, self_attns in self._layers:
            x = cross_attn(x, context=data, mask=mask) + x
            x = cross_fc(x) + x

            for self_attn, self_fc in self_attns:
                x = self_attn(x) + x
                x = self_fc(x) + x

        # queries
        if not ivy.exists(queries):
            if ivy.exists(self._queries):
                queries = ivy.einops_repeat(self._queries, '... -> b ...', b=b)
            else:
                raise Exception('If learn_query is not set as True, the queries must be provided explicitly'
                                'during the forward pass.')

        queries_shape = list(queries.shape)

        queries = ivy.einops_rearrange(queries, 'b ... d -> b (...) d')

        # cross attend from decoder queries to latents

        latents = self._decoder_cross_attn(queries, context=x)

        # optional decoder feedforward

        if ivy.exists(self._decoder):
            latents = latents + self._decoder(latents)

        # final linear out

        ret = self._to_logits(latents)

        # reshape to correct number of axes
        return ivy.reshape(ret, queries_shape[:-1] + [self._spec.output_dim])
