import torch
from torch import nn, einsum
from torch.nn import functional as F
from math import floor, log2
from functools import partial
from einops import rearrange


def exists(val):
    return val is not None

def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)

def Decompose3D(x):
    
    x_ = x.view(-1, 3, x.shape[1], x.shape[2], x.shape[3])
    x_xy, x_yz, x_zx = x_[:, 0], x_[:, 1], x_[:, 2]
    B, _, H, W = x.shape
    x_zy = x_yz.permute(0,1,3,2)
    x_xz = x_zx.permute(0,1,3,2)
    x_yx = x_xy.permute(0,1,3,2)

    x_zy_pz = x_zy.mean(dim=-1, keepdim=True).repeat(1,1,1,x_xy.shape[-1])
    x_xz_pz = x_xz.mean(dim=-2, keepdim=True).repeat(1,1,x_xy.shape[-2],1)
    x_xy_ = torch.cat([x_xy, x_zy_pz, x_xz_pz], 1)

    x_yx_px = x_yx.mean(dim=-2, keepdim=True).repeat(1,1,x_yz.shape[-2],1)
    x_xz_px = x_xz.mean(dim=-1, keepdim=True).repeat(1,1,1,x_yz.shape[-1])
    x_yz_ = torch.cat([x_yx_px, x_yz, x_xz_px], 1)

    x_yx_py = x_yx.mean(dim=-1, keepdim=True).repeat(1,1,1,x_zx.shape[-1])
    x_zy_py = x_zy.mean(dim=-2, keepdim=True).repeat(1,1,x_zx.shape[-2],1)
    x_zx_ = torch.cat([x_yx_py, x_zy_py, x_zx], 1)

    x = torch.cat([x_xy_[:, None], x_yz_[:, None], x_zx_[:, None]], 1).view(B, -1, H, W)
    return x


class Conv2DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, eps = 1e-8, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        self.eps = eps
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, h, w = x.shape

        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)
        return x

class GeneratorBlock(nn.Module):
    def __init__(self, latent_dim, input_channels, filters, upsample=True):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else None

        # Style layers
        self.to_style1 = nn.Linear(latent_dim, input_channels)
        self.to_style2 = nn.Linear(latent_dim, filters)
        self.to_style3 = nn.Linear(latent_dim, filters)
        
        # Convolutional layers
        self.conv1 = Conv2DMod(input_channels, filters, 3)
        self.conv2 = Conv2DMod(filters, filters, 3)
        self.conv3 = Conv2DMod(filters, filters, 3)
        
        self.activation = leaky_relu()

    def forward(self, x, istyle):
        if self.upsample is not None:
            x = self.upsample(x)

        # First convolution with style
        style1 = self.to_style1(istyle)
        x1 = self.conv1(x, style1)
        x1 = self.activation(x1)

        # Second convolution with style
        style2 = self.to_style2(istyle)
        x2 = self.conv2(x1, style2)
        x2 = self.activation(x2)
        
        # Third convolution with style
        style3 = self.to_style3(istyle)
        x3 = self.conv3(x2, style3)
        
        # Skip connection
        x3 += x1
        x3 = self.activation(x3)

        return x3

class ToTriPlane(torch.nn.Module):

    def __init__(self, latent_dim, input_channels, triplane_dim):
        super().__init__()

        # style layers
        self.to_style0 = nn.Linear(latent_dim, input_channels)
        self.to_style1 = nn.Linear(latent_dim, triplane_dim * 3)
        self.to_style2 = nn.Linear(latent_dim, triplane_dim * 3)
        
        #conv layers (conv0 is the projection to the triplane dimension)
        self.conv0 = Conv2DMod(input_channels, triplane_dim, 3)
        self.conv1 = Conv2DMod(triplane_dim, triplane_dim // 3, 3)
        self.conv2 = Conv2DMod(triplane_dim, triplane_dim // 3, 3)

        self.activation = leaky_relu()

    def forward(self, x, istyle):
        bs = x.shape[0]
        
        # project to tri-plane dimension
        style0 = self.to_style0(istyle)
        x = self.conv0(x, style0) # (bs, 32, 256, 256)
        x = self.activation(x)
        x = x.view(bs * 3, -1, x.shape[2], x.shape[3]) # (bs*3, 32, 256, 256)
        
        # do 3D decompositions of the triplane
        style1 = self.to_style1(istyle)
        x = Decompose3D(x) # (bs*3, 96, 512, 512)
        style1 = style1.view(style1.shape[0], 3, style1.shape[1]//3).view(style1.shape[0]*3, style1.shape[1]//3) # (bs*3, 96)
        x = self.conv1(x, style1) # (bs*3, 96, 512, 512) -> (bs*3, 32, 512, 512)
        x = self.activation(x)

        # do 3D decomposition of the triplane
        style2 = self.to_style2(istyle)
        x = Decompose3D(x) # (bs*3, 96, 512, 512)
        style2 = style2.view(style2.shape[0], 3, style2.shape[1]//3).view(style2.shape[0]*3, style2.shape[1]//3) # (bs*3, 96)
        x = self.conv2(x, style2) # (bs*3, 96, 512, 512) -> (bs*3, 32, 512, 512)
        x = self.activation(x)

        return x.view(bs, -1, x.shape[2], x.shape[3]) # (bs, 96, 512, 512)


class Generator(nn.Module):
    def __init__(self, image_size, latent_dim, network_capacity=48, triplane_dim=32, triplane_num=3, transparent=False, fmap_max=256):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.num_layers = int(log2(image_size) - 1)

        filters = [network_capacity * (2 ** (i)) for i in range(self.num_layers)][::-1]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        init_channels = filters[0]
        filters = [init_channels, *filters]

        in_out_pairs = zip(filters[:-1], filters[1:])

        self.to_initial_block = nn.ConvTranspose2d(latent_dim, init_channels, 4, 1, 0, bias=False)
        self.initial_conv = nn.Conv2d(filters[0], filters[0], 3, padding=1)
        self.blocks = nn.ModuleList([]) # 6 generator blocks now, 4 -> 8 -> 16 -> 32 -> 64 -> 128 -> 256
        self.tri_blocks = nn.ModuleList([]) # 3 triplane blocks now, 128 -> 256 -> 512
        tri_list = []
        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != (self.num_layers - 1)

            block = GeneratorBlock(
                latent_dim,
                in_chan,
                out_chan,
                upsample = not_first
            )
            self.blocks.append(block)
            tri_list.append(out_chan)
        self.plane_num = triplane_num
        self.tri_filters = tri_list[-triplane_num:]
        
        for in_chan in self.tri_filters:
           block = ToTriPlane(
                latent_dim,
                in_chan,
                triplane_dim * 3
            )
           self.tri_blocks.append(block)

    def forward(self, style):

        features = []
        avg_style = style[:, :, None, None] # (bs, 128, 1, 1)
        x = self.to_initial_block(avg_style) # (bs, 256, 4, 4)

        x = self.initial_conv(x) # (bs, 256, 4, 4)

        for block in self.blocks:
            x = block(x, style)
            if x.shape[1] in self.tri_filters:
                features.append(x)

        # select features that we want to generate multi-level triplanes
        for i, (feat, block) in enumerate(zip(features, self.tri_blocks)):
            tri_feat = block(feat, style)
            features[i] = tri_feat
        return features

if __name__ == "__main__":
    
    latent_dim = 128
    image_size = 512
    triplane_num = 2
    triplane_dim = 96
    network_capacity = 48
    transparent = False
    fmap_max = 256

    generator = Generator(image_size, latent_dim, network_capacity, transparent, fmap_max, triplane_num, triplane_dim)
    generator.cuda()
    generator.eval()
    
    style = torch.randn(1, latent_dim).cuda()
    features = generator(style)