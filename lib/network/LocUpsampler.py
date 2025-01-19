import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        if x2 is not None:
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        return self.conv(x)
        

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act_fn = nn.Sigmoid()

    def forward(self, x):
        y = self.act_fn(self.conv(x))
        return y

class Upsampler(nn.Module):
    def __init__(self, input_dim, output_dim, network_capacity=32):
        super(Upsampler, self).__init__()

        dims = [network_capacity * s for s in [4, 8, 4, 2, 1]]
        
        self.inc = DoubleConv(input_dim, dims[0])
        self.down1 = Down(dims[0], dims[1])
        self.up1 = Up(dims[0] + dims[1], dims[2])
        self.up2 = Up(dims[2], dims[3])
        self.up3 = Up(dims[3], dims[4])
        self.outc = OutConv(dims[4], output_dim)

    def forward(self, x):
        x1 = self.inc(x) # res == 128
        x2 = self.down1(x1) # res == 64
        y = self.up1(x2, x1) # res == 128
        y = self.up2(y) # res == 256
        y = self.up3(y) # res == 512
        y = self.outc(y) # res == 512
        return y


class Upsampler2(nn.Module):
    def __init__(self, input_dim, output_dim, network_capacity=32):
        super(Upsampler2, self).__init__()

        dims = [network_capacity * s for s in [4, 8, 4, 2, 1]]
        
        self.inc = DoubleConv(input_dim, dims[0])
        
        # Downsampling
        self.down1 = Down(dims[0], dims[1])
        self.down2 = Down(dims[1], dims[2])
        
        # Bridge
        self.bridge = DoubleConv(dims[2], dims[3])
        
        # Upsampling
        self.up1 = Up(dims[3] + dims[2], dims[4])
        self.up2 = Up(dims[4], dims[4])
        
        # Additional Upsampling to get to 512x512 resolution
        self.up3 = Up(dims[4], dims[4])
        
        self.up4 = Up(dims[4], dims[4])
        
        self.up5 = Up(dims[4], dims[4])
        self.outc = OutConv(dims[4], output_dim)

    def forward(self, x):
        x1 = self.inc(x)          # 128x128
        x2 = self.down1(x1)       # 64x64
        x3 = self.down2(x2)       # 32x32
        
        x4 = self.bridge(x3)      # 32x32
        
        y = self.up1(x4, x3)      # 32x32
        y = self.up2(y)           # 64x64
        y = self.up3(y)           # 128x128
        y = self.up4(y)           # 256x256
        y = self.up5(y)           # 512x512
        y = self.outc(y)
        return y

class ResidualDoubleConv(nn.Module):
    """
    A series of convolutions followed by a residual connection.
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualDoubleConv, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        residual = self.residual_conv(x)
        x = self.double_conv(x)
        x = residual + x
        return x

class Upsampler3(nn.Module):
    def __init__(self, input_dim, output_dim, network_capacity=32):
        super(Upsampler3, self).__init__()

        dims = [network_capacity * s for s in [4, 8, 4, 2, 1]]
        
        self.inc = ResidualDoubleConv(input_dim, dims[0])
        
        # Downsampling
        self.down1 = Down(dims[0], dims[1])
        self.down2 = Down(dims[1], dims[2])
        
        # Bridge
        self.bridge = ResidualDoubleConv(dims[2], dims[3])
        
        # Upsampling with skip connections
        self.up1 = Up(dims[3] + dims[2], dims[4])
        self.up2 = Up(dims[4] + dims[1], dims[4])
        self.up3 = Up(dims[4] + dims[0], dims[4])
        self.up4 = Up(dims[4], dims[4])
        self.up5 = Up(dims[4], dims[4])
        
        self.outc = OutConv(dims[4], output_dim)

    def forward(self, x):
        x1 = self.inc(x)          # 128x128
        x2 = self.down1(x1)       # 64x64
        x3 = self.down2(x2)       # 32x32
        
        x4 = self.bridge(x3)      # 32x32
        
        y = self.up1(x4, x3)      # 64x64
        y = self.up2(y, x2)       # 128x128
        y = self.up3(y, x1)       # 256x256
        y = self.up4(y)           # 256x256
        y = self.up5(y)           # 512x512
        y = self.outc(y)
        return y

class ExtendedResidualDoubleConv(nn.Module):
    """
    An extended version of ResidualDoubleConv with more layers for increased depth.
    """
    def __init__(self, in_channels, out_channels):
        super(ExtendedResidualDoubleConv, self).__init__()
        self.conv1 = ResidualDoubleConv(in_channels, out_channels)
        self.conv2 = ResidualDoubleConv(out_channels, out_channels)
        self.conv3 = ResidualDoubleConv(out_channels, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.conv3(x)


class ExtendedDown(nn.Module):
    """Extended downscaling with more convolutions"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ExtendedResidualDoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class ExtendedUp(nn.Module):
    """Extended upsampling with more convolutions"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ExtendedResidualDoubleConv(in_channels, out_channels)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        if x2 is not None:
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        return self.conv(x)


class LocUpsampler(nn.Module):
    def __init__(self, input_dim, output_dim, network_capacity=32):
        super(LocUpsampler, self).__init__()

        dims = [network_capacity * s for s in [4, 8, 4, 2, 1]]
        
        # Initial convolution block
        self.inc = ExtendedResidualDoubleConv(input_dim, dims[0])
        
        # Extended Downsampling
        self.down1 = ExtendedDown(dims[0], dims[1])
        self.down2 = ExtendedDown(dims[1], dims[2])
        
        # Bridge
        self.bridge = ExtendedResidualDoubleConv(dims[2], dims[3])
        
        # Extended Upsampling with skip connections
        self.up1 = ExtendedUp(dims[3] + dims[2], dims[4])
        self.up2 = ExtendedUp(dims[4] + dims[1], dims[4])
        self.up3 = ExtendedUp(dims[4] + dims[0], dims[4])
        
        # Output convolution
        self.outc = OutConv(dims[4], output_dim)
        

    def forward(self, x):
        x1 = self.inc(x)          # 128x128
        x2 = self.down1(x1)       # 64x64
        x3 = self.down2(x2)       # 32x32
        
        x4 = self.bridge(x3)      # 32x32
        
        import pdb; pdb.set_trace()
        y = self.up1(x4, x3)      # 32x32
        y = self.up2(y, x2)       # 64x64
        y = self.up3(y, x1)       # 128x128
        y = self.outc(y)
        return y

if __name__ == "__main__":
    # Checking the modified U-Net model
    upsampler_enhanced = Upsampler2(32, 3).cuda()
    x = torch.randn(1, 32, 128, 128).cuda()
    y = upsampler_enhanced(x)
    # save the model
    torch.save(upsampler_enhanced.state_dict(), 'upsampler2.pth')
    
    
    # chekc the upsample3
    upsampler3 = Upsampler3(32, 3).cuda()
    y = upsampler3(x)
    torch.save(upsampler3.state_dict(), 'upsampler3.pth')
    
    
    # check the upsample4
    upsampler4 = LocUpsampler(64, 3).cuda()
    y = upsampler4(x)
    torch.save(upsampler4.state_dict(), 'upsampler4.pth')
    
    # check the upsample
    upsampler = Upsampler(32, 3).cuda()
    y = upsampler(x)
    torch.save(upsampler.state_dict(), 'upsampler.pth')