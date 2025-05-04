"""Superresolution network architectures from the paper
"Efficient Geometry-aware 3D Generative Adversarial Networks"."""

import torch
from training.networks_stylegan2 import Conv2dLayer, SynthesisLayer, ToRGBLayer
from torch_utils.ops import upfirdn2d
from torch_utils import persistence
from torch_utils import misc

from training.networks_stylegan2 import SynthesisBlock
import numpy as np
import torch

#----------------------------------------------------------------------------

# for 512x512 generation
class SuperresolutionHybrid8X(torch.nn.Module):
    def __init__(self, channels, img_resolution, sr_num_fp16_res, sr_antialias,
                num_fp16_res=4, conv_clamp=None, channel_base=None, channel_max=None,# IGNORE
                **block_kwargs):
        super().__init__()
        assert img_resolution == 512

        use_fp16 = sr_num_fp16_res > 0
        self.input_resolution = 128
        self.sr_antialias = sr_antialias
        self.block0 = SynthesisBlock(channels, 128, w_dim=512, resolution=256,
                img_channels=3, is_last=False, use_fp16=use_fp16, conv_clamp=(256 if use_fp16 else None), **block_kwargs)
        self.block1 = SynthesisBlock(128, 64, w_dim=512, resolution=512,
                img_channels=3, is_last=True, use_fp16=use_fp16, conv_clamp=(256 if use_fp16 else None), **block_kwargs)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter([1,3,3,1]))

    def forward(self, rgb, x, ws, **block_kwargs):
        ws = ws[:, -1:, :].repeat(1, 3, 1)

        if x.shape[-1] != self.input_resolution:
            x = torch.nn.functional.interpolate(x, size=(self.input_resolution, self.input_resolution),
                                                  mode='bilinear', align_corners=False, antialias=self.sr_antialias)
            rgb = torch.nn.functional.interpolate(rgb, size=(self.input_resolution, self.input_resolution),
                                                  mode='bilinear', align_corners=False, antialias=self.sr_antialias)

        x, rgb = self.block0(x, rgb, ws, **block_kwargs)
        x, rgb = self.block1(x, rgb, ws, **block_kwargs)
        return rgb

def maskErosion(self, mask, erosionFactor):
    offsetY = int(erosionFactor * 40)
    # throat
    mask2 = mask[:,:,0:-offsetY,:]
    mask2 = torch.cat([torch.ones_like(mask[:,:,0:offsetY,:]), mask2], 2)
    # forehead
    offsetY = int(erosionFactor * 8) #<<<<
    mask3 = mask[:,:,offsetY:,:]
    mask3 = torch.cat([mask3, torch.ones_like(mask[:,:,0:offsetY,:])], 2)
    mask = mask * mask2 * mask3

    offsetX = int(erosionFactor * 15)
    # left
    mask4 = mask[:,:,:,0:-offsetX]
    mask4 = torch.cat([torch.ones_like(mask[:,:,:,0:offsetX]), mask4], 3)
    # right
    mask5 = mask[:,:,:,offsetX:]
    mask5 = torch.cat([mask5,torch.ones_like(mask[:,:,:,0:offsetX])], 3)
    return mask * mask4 * mask5
