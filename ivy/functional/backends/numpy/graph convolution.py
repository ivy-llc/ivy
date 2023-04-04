import ivy
import ivy.numpy as np

class GCNConv:
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = ivy.variable((in_channels, out_channels))

    def __call__(self, x, edge_index):
        # normalize adjacency matrix
        row, col = edge_index
        deg = np.zeros(x.shape[0])
        deg = ivy.scatter_add(deg, col, ivy.ones_like(col, dtype=np.float32))
        deg_inv_sqrt = ivy.sqrt(1.0 / ivy.clip(deg, 1))
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        norm = ivy.reshape(norm, (edge_index.shape[1], 1))

        # perform message passing
        x = ivy.matmul(x, self.weight)
        x = ivy.transpose(x, (1, 0))
        x = norm * x[col]
        x = ivy.transpose(x, (1, 0))
        out = ivy.scatter_add(x, row, x)
        return out
