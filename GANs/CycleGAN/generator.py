import torch
import torch.nn as nn

# Will be for both up-sampling and down-sampling:
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            # we'll upsample or downsample according to "down" parameter
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs) #kwargs will hold kernel, stride and padding sizes and in that order
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity() # nn.Identity() literally just passes through values without any changes to it
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1), # According to paper, where they apply activation-func over 1st block but not over the 2nd one
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, img_channels, num_features=64, num_residuals=9):
        super().__init__()
        self.initial_block = nn.Sequential( # a regular Conv block, just without the instance-normalization
            nn.Conv2d(img_channels, 64, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.ReLU(inplace=True),
        )

        # Creating the down part of the generator:
        self.down_blocks = nn.ModuleList( # ModuleList can be indexed like a regular Python list, but modules it contains are properly registered, and will be visible by all Module methods.
            [
                ConvBlock(num_features, num_features*2, down=True, kernel_size=3, stride=2, padding=1),
                ConvBlock(num_features*2, num_features*4, down=True, kernel_size=3, stride=2, padding=1),
            ]
        )

        # Creating the bottleneck of the generator:
        self.residual_blocks = nn.Sequential(
            *[ ResidualBlock(num_features*4) for _ in range(num_residuals) ] # creates multiple residual-blocks and unwraps it from list
        )

        # Creating the up part of the generator:
        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(num_features*4, num_features*2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
                ConvBlock(num_features*2, num_features*1, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
            ]
        )

        self.last_layer = nn.Conv2d(num_features*1, img_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect")

    def forward(self, x):
        # Propagate through the 1st layer:
        x = self.initial_block(x)

        # Propagate through the down-part:
        for layer in self.down_blocks:
            x = layer(x)

        # Propagate through the bottleneck:
        x = self.residual_blocks(x)

        # Propagate through the up-part:
        for layer in self.up_blocks:
            x = layer(x)

        # Propagate through the last layer:
        return torch.tanh(self.last_layer(x))

