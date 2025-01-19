import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from pytorch3d.transforms import so3_exponential_map
from lib.network.Generator import Generator

from lib.network.MLP import MLP, OSGDecoder
from lib.network.PositionalEmbedding import get_embedder

class HeadModule(nn.Module):
    def __init__(self, opt):
        super(HeadModule, self).__init__()
        self.density_mlp = MLP(opt.density_mlp, last_op=None)
        self.color_mlp = MLP(opt.color_mlp, last_op=None)
        self.pos_embedding, _ = get_embedder(opt.pos_freq)
        self.view_embedding, _ = get_embedder(opt.view_freq)
        self.generator = Generator(opt.triplane_res, opt.exp_dim_3d, opt.triplane_dim * 3 // 2, opt.triplane_dim, opt.triplane_n)
        self.noise = opt.noise
        self.bbox = opt.bbox

    def forward(self, data):
        B, C, N = data['query_pts'].shape
        query_pts = data['query_pts']
        query_viewdirs = data['query_viewdirs']
        if 'pose' in data:
            R = so3_exponential_map(data['pose'][:, :3])
            T = data['pose'][:, 3:, None]
            S = data['scale'][:, :, None]
            query_pts = torch.bmm(R.permute(0,2,1), (query_pts - T)) / S
            query_viewdirs = torch.bmm(R.permute(0,2,1), query_viewdirs)
        query_viewdirs_embedding = self.view_embedding(rearrange(query_viewdirs, 'b c n -> (b n) c'))
        
        triplanes = data['triplane']
        feature = self.sample_from_triplane(triplanes, query_pts)
        density, color = self.decode_feature(feature, query_viewdirs_embedding, B)

        data['density'] = density
        data['color'] = color
        return data

    
    def sample_from_triplane(self, triplanes, query_pts):
        if type(triplanes) != list:
            plane_dim = triplanes.shape[1] // 3
            planes = [triplanes[:, i*plane_dim:(i+1)*plane_dim, :, :] for i in range(3)]
            triplanes = torch.cat(planes, dim=0) # (bs, 3, c, h, w)
        else:
            for i in range(len(triplanes)):
                plane_dim = triplanes[i].shape[1] // 3
                triplanes[i] = torch.cat([triplanes[i][:, j*plane_dim:(j+1)*plane_dim, :, :] for j in range(3)], dim=0)
        
        # Compute the coordinates for each plane
        bbox_deltas = [(self.bbox[i][1] - self.bbox[i][0]) for i in range(3)]
        bbox_midpoints = [(0.5 * (self.bbox[i][0] + self.bbox[i][1])) for i in range(3)]
        coords_list = [
            (query_pts[:, i:i+1] - bbox_midpoints[i]) / (0.5 * bbox_deltas[i]) for i in range(3)
        ]

        # Combine coordinates for different plane projections
        combinations = [(1, 2), (0, 2), (0, 1)]  # vw, uw, uv
        coor_combinations = [torch.cat([coords_list[i], coords_list[j]], dim=1) for i, j in combinations] # (bs, )
        coor_combinations = [rearrange(comb, 'b (t c) n -> b n t c', t=1) for comb in coor_combinations]
        coor_combinations = torch.cat(coor_combinations, dim=0)
        
        def sampling(triplane, coor_combinations):
            # Sample all features at once
            sampled_features_stacked = torch.nn.functional.grid_sample(triplane, coor_combinations, align_corners=True, mode='bilinear')
            sampled_features_stacked = rearrange(sampled_features_stacked, 'b c n t -> b c (n t)')

            # Split and reshape to get individual plane features
            sampled_features = torch.chunk(sampled_features_stacked, chunks=3, dim=0)  # (3, bs, c, n*t)
            feature = sum(sampled_features) # (bs, c, n*t)
            return rearrange(feature, 'b c n -> (b n) c')
        
        if type(triplanes) == list:
            feature = 0
            for i in range(len(triplanes)):
                feature += sampling(triplanes[i], coor_combinations)
        else:
            feature = sampling(triplanes, coor_combinations)
        return feature
    
    def decode_feature(self, feature, view_dir, B):
        density = rearrange(self.density_mlp(feature), '(b n) c -> b c n', b=B)
        if self.training:
            density = density + torch.randn_like(density) * self.noise
        
        color_input = torch.cat([feature, view_dir], 1)
        color = rearrange(self.color_mlp(color_input), '(b n) c -> b c n', b=B)

        return density, color