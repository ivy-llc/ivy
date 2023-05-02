# global
import ivy


class ResidualBlock(ivy.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        Helper module that is used in the ResNet implementation.
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param stride: stride used in the convolutions
        :param downsample: downsample function used in the residual path
        """
        super(ResidualBlock, self).__init__()
        self.conv1 = ivy.Sequential(
            ivy.Conv2D(in_channels, out_channels, [3, 3], stride, 1),
            ivy.BatchNorm2D(out_channels),
            ivy.ReLU(),
        )
        self.conv2 = ivy.Sequential(
            ivy.Conv2D(out_channels, out_channels, [3, 3], 1, 1),
            ivy.BatchNorm2D(out_channels),
        )
        self.downsample = downsample
        self.relu = ivy.ReLU()
        self.out_channels = out_channels

    def _forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out = ivy.add(out, residual)
        out = self.relu(out)
        return out


class ResNet(ivy.Module):
    def __init__(self, block, layers, num_classes=10):
        """
        ResNet implementation.
        :param block: residual block used in the network
        :param layers: list containing the number of blocks per layer
        :param num_classes: number of classes in the dataset
        """
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = ivy.Sequential(
            ivy.Conv2D(3, 64, [7, 7], 2, 3), ivy.BatchNorm2D(64), ivy.ReLU()
        )
        self.maxpool = ivy.MaxPool2D(3, 2, 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = ivy.AvgPool2D(7, 1, 0)
        self.fc = ivy.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = ivy.Sequential(
                ivy.Conv2D(
                    self.inplanes, planes, [1, 1], stride, 0
                ),  # check if padding is correct
                ivy.BatchNorm2D(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return ivy.Sequential(*layers)

    def _forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.reshape((x.shape[0], -1))
        x = self.fc(x)
        return x


def resnet_18():
    """
    ResNet-18 model
    """
    return ResNet(ResidualBlock, [2, 2, 2, 2])


def resnet_34():
    """
    ResNet-34 model
    """
    return ResNet(ResidualBlock, [3, 4, 6, 3])


def resnet_101():
    """
    ResNet-101 model
    """
    return ResNet(ResidualBlock, [3, 4, 23, 3])


def resnet_152():
    """
    ResNet-152 model
    """
    return ResNet(ResidualBlock, [3, 8, 36, 3])
