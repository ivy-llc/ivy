# local
import ivy


def embedding(input, weight, padding_idx=None, max_norm=None,
              norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    return ivy.embedding(input, weight, padding_idx=None, max_norm=None,
                         norm_type=2.0, scale_grad_by_freq=False, sparse=False)