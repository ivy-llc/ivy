import torch
import torch.nn.functional as F


def convolutional_pipeline(x):
    x = F.conv2d(x, torch.randn(64, 3, 3, 3), stride=1, padding=1)
    # x = F.batch_norm(x, torch.randn(64), torch.randn(64), torch.randn(64), torch.randn(64))
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2, stride=2)
    return x
