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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        self.stride = stride
        self.relu = ivy.ReLU()
        super(ResidualBlock, self).__init__()

    def _build(self, *args, **kwrgs):
        self.conv1 = ivy.Sequential(
            ivy.Conv2D(self.in_channels, self.out_channels, [3, 3], self.stride, 1),
            ivy.BatchNorm2D(self.out_channels),
            ivy.ReLU(),
        )
        self.conv2 = ivy.Sequential(
            ivy.Conv2D(self.out_channels, self.out_channels, [3, 3], 1, 1),
            ivy.BatchNorm2D(self.out_channels),
        )

    def _forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out = ivy.add(out, residual)
        out = self.relu(out)
        return out


class ResLayer(ivy.Module):
    def __init__(
        self,
        block,
        planes,
        blocks,
        current_inplanes,
        stride=1,
    ):
        """Helper module that is used in the ResNet implementation."""
        self.block = block
        self.planes = planes
        self.blocks = blocks
        self.stride = stride
        self.current_inplanes = current_inplanes
        super(ResLayer, self).__init__()

    def _build(self, *args, **kwargs):
        downsample = None
        layers = []
        if self.stride != 1 or self.current_inplanes != self.planes:
            downsample = ivy.Sequential(
                ivy.Conv2D(
                    self.current_inplanes, self.planes, [1, 1], self.stride, 0
                ),  # check if padding is correct
                ivy.BatchNorm2D(self.planes),
            )

        layers.append(
            self.block(self.current_inplanes, self.planes, self.stride, downsample)
        )

        self.current_inplanes = self.planes
        for _ in range(1, self.blocks):
            layers.append(self.block(self.current_inplanes, self.planes))
        self.res_layer = ivy.Sequential(*layers)

    def _forward(self, inputs):
        return self.res_layer(inputs)


class ResNet(ivy.Module):
    def __init__(self, block, layers, num_classes=10):
        """
        Resnet implementation.
        :param block: residual block used in the network
        :param layers: list containing the number of blocks per layer
        :param num_classes: number of classes in the dataset
        """
        self.num_classes = num_classes
        self.block = block
        self.layers = layers
        self.inplanes = 64
        self.conv1 = ivy.Sequential(
            ivy.Conv2D(3, 64, [7, 7], 2, 3), ivy.BatchNorm2D(64), ivy.ReLU()
        )
        self.maxpool = ivy.MaxPool2D(3, 2, 1)

        self.layer0 = ResLayer(
            self.block, 64, self.layers[0], current_inplanes=64, stride=1
        )
        self.layer1 = ResLayer(
            self.block, 128, self.layers[1], current_inplanes=64, stride=2
        )
        self.layer2 = ResLayer(
            self.block, 256, self.layers[2], current_inplanes=128, stride=2
        )
        self.layer3 = ResLayer(
            self.block, 512, self.layers[3], current_inplanes=256, stride=2
        )

        self.avgpool = ivy.AvgPool2D(7, 1, 0)
        self.fc = ivy.Linear(512, self.num_classes)

        super(ResNet, self).__init__()

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
