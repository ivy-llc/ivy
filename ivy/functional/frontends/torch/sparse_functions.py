# local
import ivy


def embedding(num_embeddings, embedding_dim, padding_idx=None,
              max_norm=None, norm_type=2.0, scale_grad_by_freq=False,
              sparse=False, _weight=None, device=None, dtype=None):
    return ivy.embedding(num_embeddings, embedding_dim, padding_idx=None,
                         max_norm=None, norm_type=2.0, scale_grad_by_freq=False,
                         sparse=False, _weight=None, device=None, dtype=None)


