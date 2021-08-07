import torch
import torch.nn as nn


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=True, padding_mode="reflect"), # k=1, p=1, padding by input's reflection (According to paper, helps reducing artifacts)
            nn.InstanceNorm2d(out_channels), # performs normalization across the feature map itself (hence the instance)
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial_block = nn.Sequential( # initial disc's block, won't hold instance-normalization
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(negative_slope=0.2),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(DiscriminatorBlock(in_channels, feature, stride=1 if feature==features[-1] else 2)) # s=1 only on the last block
            in_channels = feature
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"))

        self.model = nn.Sequential(*layers) # unwraps the list's content to single parameters

    def forward(self, x):
        x = self.initial_block(x)
        return torch.sigmoid(self.model(x)) # recall that the disc' returns a patchgan-output and not a singular value.

""" TEST
x = torch.randn((5,3,256,256))
model = Discriminator(in_channels=3)
preds = model(x)
print(preds.shape)
"""
