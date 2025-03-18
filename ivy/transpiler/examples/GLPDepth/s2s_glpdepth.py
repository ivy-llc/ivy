import torch
import torch.nn as nn

from .mit import mit_b4


class GLPDepth(nn.Module):
    def __init__(self, max_depth=10.0, is_train=False):
        super().__init__()
        self.max_depth = max_depth

        self.encoder = mit_b4()
        channels_in = [512, 320, 128]
        channels_out = 64

        self.decoder = Decoder(channels_in, channels_out)

        self.last_layer_depth = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        conv1, conv2, conv3, conv4 = self.encoder(x)
        out = self.decoder(conv1, conv2, conv3, conv4)
        out_depth = self.last_layer_depth(out)
        out_depth = torch.sigmoid(out_depth) * self.max_depth

        return {"pred_d": out_depth}


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.bot_conv = nn.Conv2d(
            in_channels=in_channels[0], out_channels=out_channels, kernel_size=1
        )
        self.skip_conv1 = nn.Conv2d(
            in_channels=in_channels[1], out_channels=out_channels, kernel_size=1
        )
        self.skip_conv2 = nn.Conv2d(
            in_channels=in_channels[2], out_channels=out_channels, kernel_size=1
        )

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.fusion1 = SelectiveFeatureFusion(out_channels)
        self.fusion2 = SelectiveFeatureFusion(out_channels)
        self.fusion3 = SelectiveFeatureFusion(out_channels)

    def forward(self, x_1, x_2, x_3, x_4):
        x_4_ = self.bot_conv(x_4)
        out = self.up(x_4_)

        x_3_ = self.skip_conv1(x_3)
        out = self.fusion1(x_3_, out)
        out = self.up(out)

        x_2_ = self.skip_conv2(x_2)
        out = self.fusion2(x_2_, out)
        out = self.up(out)

        out = self.fusion3(x_1, out)
        out = self.up(out)
        out = self.up(out)

        return out


class SelectiveFeatureFusion(nn.Module):
    def __init__(self, in_channel=64):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=int(in_channel * 2),
                out_channels=in_channel,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=int(in_channel / 2),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(int(in_channel / 2)),
            nn.ReLU(),
        )

        self.conv3 = nn.Conv2d(
            in_channels=int(in_channel / 2),
            out_channels=2,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_local, x_global):
        x = torch.cat((x_local, x_global), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        attn = self.sigmoid(x)

        out = x_local * attn[:, 0, :, :].unsqueeze(1) + x_global * attn[
            :, 1, :, :
        ].unsqueeze(1)

        return out
