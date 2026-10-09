"""Project-owned 2.5D U-Net: five physical axial planes -> central Heart logits."""
import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, norm_groups):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.GroupNorm(norm_groups, output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GroupNorm(norm_groups, output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, image):
        return self.layers(image)


class HeartUNet25D(nn.Module):
    def __init__(self, input_channels=5, encoder_channels=(16, 32, 64, 128), norm_groups=8, output_channels=1):
        super().__init__()
        if input_channels <= 0 or output_channels <= 0 or len(encoder_channels) != 4 or norm_groups <= 0:
            raise ValueError('Positive input channels and four encoder stages required')
        if any(c <= 0 or c % norm_groups for c in encoder_channels):
            raise ValueError('Encoder channels must be divisible by GroupNorm groups')
        self.input_channels = input_channels
        incoming = (input_channels, *encoder_channels[:-1])
        self.encoder = nn.ModuleList(
            ConvBlock(a, b, norm_groups) for a, b in zip(incoming, encoder_channels)
        )
        self.pool = nn.MaxPool2d(2)
        self.decoder = nn.ModuleList(
            ConvBlock(encoder_channels[i + 1] + encoder_channels[i], encoder_channels[i], norm_groups)
            for i in (2, 1, 0)
        )
        self.output = nn.Conv2d(encoder_channels[0], output_channels, 1)

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != self.input_channels or min(image.shape[-2:]) < 8:
            raise ValueError('Expected BCHW with configured input channels and H/W >= 8')
        skips = []
        for index, block in enumerate(self.encoder):
            image = block(image)
            if index < 3:
                skips.append(image)
                image = self.pool(image)
        for block, skip in zip(self.decoder, reversed(skips)):
            image = F.interpolate(image, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            image = block(torch.cat((image, skip), dim=1))
        return self.output(image)
