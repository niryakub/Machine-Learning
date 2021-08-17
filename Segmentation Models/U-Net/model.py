import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False), #kernel=3, stride=1, padding=1, bias=False since we'll be using BatchNorm
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),  # kernel=3, stride=1, padding=1, bias=False since we'll be using BatchNorm
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]): #in_channels=3=RGB, out_channels=1=???
        super(UNET, self).__init__()
        self.ups = nn.ModuleList() #ModuleList will be helpful when applying model.train/model.eval in the future
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET: (creating DoubleConv instance for every step in the down-part, and appends it to ModuleList.
        for feature in features:
            self.downs.append(DoubleConv(in_channels,feature))
            in_channels = feature

        # Up part of UNET:
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(in_channels=feature*2, out_channels=feature, kernel_size=2, stride=2) ) # this way we double the height&weight of the input
            self.ups.append(DoubleConv(feature*2, feature))

        # Bottle neck part & final layer of UNET:
        self.bottleneck = DoubleConv(features[-1], features[-1]*2) # with channels of size 512, 1024...
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1) # the very final layer...

    def forward(self, x):
        skip_connections = [] # the first cell will hold the highest-resolution block and so on...

        # Perform the Down-part of the Unet
        for down in self.downs:
            x = down(x) # perform the i-th layer in the Down-UNET-part
            skip_connections.append(x) # save the output for the future skip-connection
            x = self.pool(x)

        # Perform the bottleneck of the Unet
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1] # reverse the list...

        for idx in range(0, len(self.ups), 2): # in jumps of 2, since we're performing both  UpSampling & Double-Conv-Layer
            x = self.ups[idx](x) # Upsample
            skip_connection = skip_connections[idx//2] # //=floor division, so we're indexing correctly because of jumps of 2

            # We're going to solve here the following issue: i.e, in the down-part, we're max-pooling from 161x161 by 2, hence we'll get 80x80... on the upsampling accordingly,
            # ... we'll go from 80x80 to 160x160, thus we'll attempt to concat 161x161 to 160x160 channels.
            if x.shape != skip_connection.shape :
                x = TF.resize(x, size=skip_connection.shape[2:]) # resizing x to the height and width of skip_connection's

            concat_skip = torch.cat((skip_connection,x), dim=1) # concat' along the Channel-Dimension.. (Batch,Channel,Height,Width)
            x = self.ups[idx+1](concat_skip) # DoubleConv

        return self.final_conv(x) # traverse through last layer.


def test():
    x = torch.randn((3,1,160,160))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape


if __name__ == "__main__":
    test()










