import torch
from torch import nn
from torch.nn import functional as F

from lib.network.AutoEncoder import Encoder
from lib.network.MLP import MLP
from lib.module.HeadModule import HeadModule
from lib.network.Upsampler import Upsampler5
from lib.network.Generator import Generator


class AvatarModule(nn.Module):
    def __init__(self, opt):
        super(AvatarModule, self).__init__()

        self.exp_dim_2d = opt.exp_dim_2d
        self.encoder = Encoder(opt.exp_dim_2d, opt.encoder_dims)
        self.mapping_mlp = MLP(opt.mapping_dims)
        self.headmodule = HeadModule(opt.headmodule)
        self.upsampler = Upsampler5(opt.headmodule.color_mlp[-1], 3, opt.upsampler_capacity)
    
    def encode(self, input):
        return self.encoder(input - 0.5)

    def mapping(self, exp_code_2d):
        return self.mapping_mlp(exp_code_2d)

    def head(self, data):
        return self.headmodule(data)

    def upsample(self, feature_map):
        return self.upsampler(feature_map)

    def forward(self, func, data):
        if func == 'encode':
            return self.encode(data)
        elif func == 'mapping':
            return self.mapping(data)
        elif func == 'head':
            return self.head(data)
        elif func == 'upsample':
            return self.upsample(data)
        elif func == 'locupsample':
            return self.locupsample(data)
    
    def get_generator(self):
        return self.headmodule.generator