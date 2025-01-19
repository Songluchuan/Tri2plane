import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from einops import rearrange

from pytorch3d.transforms import so3_exponential_map


class NeuralCameraModule(nn.Module):
    def __init__(self, avatarmodule, opt):
        super(NeuralCameraModule, self).__init__()

        self.avatarmodule = avatarmodule
        self.model_bbox = opt.model_bbox
        self.image_size = opt.image_size
        self.N_samples = opt.N_samples
        self.near_far = opt.near_far
        try:
            self.generator = avatarmodule.module.get_generator()
        except:
            self.generator = avatarmodule.get_generator()
        self.plane_num = self.generator.plane_num

    @staticmethod
    def gen_part_rays(extrinsic, intrinsic, resolution, image_size, local_scale=1):
         # resolution (width, height)
        rays_o_list = []
        rays_d_list = []
        rot = extrinsic[:, :3, :3].transpose(1, 2)
        trans = -torch.bmm(rot, extrinsic[:, :3, 3:])
        c2w = torch.cat((rot, trans.reshape(-1, 3, 1)), dim=2)
        local_info = []
        for b in range(intrinsic.shape[0]):
            fx, fy, cx, cy = intrinsic[b, 0, 0], intrinsic[b, 1, 1], intrinsic[b, 0, 2], intrinsic[b, 1, 2]
            res_w = resolution[b, 0].int().item()
            res_h = resolution[b, 1].int().item()
            W = image_size[b, 0].int().item()
            H = image_size[b, 1].int().item()
            if local_scale != 1:
                H_resize = int(H * local_scale)
                W_resize = int(W * local_scale)

                # first choose the loca region from the whole H_resize, W_resize (select a squre region of size (H, W) from the H_resize, W_resize)
                x_start = torch.randint(0, int(W - W_resize), (1,)).item()
                y_start = torch.randint(0, int(H - H_resize), (1,)).item()
                
                # select the local region
                x_end = x_start + W_resize
                y_end = y_start + H_resize

                # build the meshgrid for the local region for the rays to do super-resolution
                i, j = torch.meshgrid(
                    torch.linspace(x_start + 0.5, x_end - 0.5, res_w, device=c2w.device),
                    torch.linspace(y_start + 0.5, y_end - 0.5, res_h, device=c2w.device)
                )
                # add the local region offset to the local info
                local_info.append([x_start, x_end, y_start, y_end])
                
            else:
                i, j = torch.meshgrid(torch.linspace(0.5, W-0.5, res_w, device=c2w.device), torch.linspace(0.5, H-0.5, res_h, device=c2w.device))  # pytorch's meshgrid has indexing='ij'
            i = i.t()
            j = j.t()
            dirs = torch.stack([(i-cx)/fx, (j-cy)/fy, torch.ones_like(i)], -1)
            # Rotate ray directions from camera frame to the world frame
            rays_d = torch.sum(dirs.unsqueeze(-2) * c2w[b, :3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
            # Translate camera frame's origin to the world frame. It is the origin of all rays.
            rays_o = c2w[b, :3,-1].expand(rays_d.shape) # (bs, res_w, res_h, 3)
            rays_o_list.append(rays_o.unsqueeze(0))
            rays_d_list.append(rays_d.unsqueeze(0))
        rays_o_list = torch.cat(rays_o_list, dim=0)
        rays_d_list = torch.cat(rays_d_list, dim=0)
        # rays [B, C, H, W]
        return rearrange(rays_o_list, 'b h w c -> b c h w'), rearrange(rays_d_list, 'b h w c -> b c h w'), local_info
    
    
    @staticmethod
    def gen_scale_rays(extrinsic, intrinsic, resolution, image_size, local_scale=1):

        rot = extrinsic[:, :3, :3].transpose(1, 2)
        trans = -torch.bmm(rot, extrinsic[:, :3, 3:])
        c2w = torch.cat((rot, trans.reshape(-1, 3, 1)), dim=2)
        local_info = []
        
        X_indexes = []
        Y_indexes = []
        ray_o_lists = []
        ray_d_lists = []
        for i in range(int(1/local_scale)):
            X_start = int(i * local_scale * image_size[0][0])
            X_end = int((i+1) * local_scale * image_size[0][0])
            X_indexes.append([X_start, X_end])
            
            Y_start = int(i * local_scale * image_size[0][1])
            Y_end = int((i+1) * local_scale * image_size[0][1])
            Y_indexes.append([Y_start, Y_end])
        
        for X_index in X_indexes:
            x_start, x_end = X_index
            for Y_index in Y_indexes:
                y_start, y_end = Y_index
                # add the local region offset to the local info
                local_info.append([x_start, x_end, y_start, y_end])
                rays_o_list = []
                rays_d_list = []
                for b in range(intrinsic.shape[0]):
                    fx, fy, cx, cy = intrinsic[b, 0, 0], intrinsic[b, 1, 1], intrinsic[b, 0, 2], intrinsic[b, 1, 2]
                    res_w = resolution[b, 0].int().item()
                    res_h = resolution[b, 1].int().item()

                    # build the meshgrid for the local region for the rays to do super-resolution
                    i, j = torch.meshgrid(
                        torch.linspace(x_start + 0.5, x_end - 0.5, res_w, device=c2w.device),
                        torch.linspace(y_start + 0.5, y_end - 0.5, res_h, device=c2w.device)
                    )

                    i = i.t()
                    j = j.t()
                    dirs = torch.stack([(i-cx)/fx, (j-cy)/fy, torch.ones_like(i)], -1)
                    # Rotate ray directions from camera frame to the world frame
                    rays_d = torch.sum(dirs.unsqueeze(-2) * c2w[b, :3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
                    # Translate camera frame's origin to the world frame. It is the origin of all rays.
                    rays_o = c2w[b, :3,-1].expand(rays_d.shape) # (bs, res_w, res_h, 3)
                    rays_o_list.append(rays_o.unsqueeze(0))
                    rays_d_list.append(rays_d.unsqueeze(0))
                rays_o_list = torch.cat(rays_o_list, dim=0)
                rays_d_list = torch.cat(rays_d_list, dim=0)
                
                rays_o_list, rays_d_list = rearrange(rays_o_list, 'b h w c -> b c h w'), rearrange(rays_d_list, 'b h w c -> b c h w')
                ray_o_lists.append(rays_o_list)
                ray_d_lists.append(rays_d_list)
        
        return ray_o_lists, ray_d_lists, local_info

    @staticmethod
    def coords_select(image, coords):
        select_rays = []
        for i in range(image.shape[0]):
            select_rays.append(image[i, :, coords[i, :, 1], coords[i, :, 0]].unsqueeze(0))
        select_rays = torch.cat(select_rays, dim=0)
        return select_rays

    @staticmethod
    def gen_near_far_fixed(near, far, samples, batch_size, device):
        nf = torch.zeros((batch_size, 2, samples), device=device)
        nf[:, 0, :] = near
        nf[:, 1, :] = far
        return nf

    def gen_near_far(self, rays_o, rays_d, R, T, S):
        """calculate intersections with 3d bounding box for batch"""
        B = rays_o.shape[0]
        rays_o_can = torch.bmm(R.permute(0,2,1), (rays_o - T)) / S
        rays_d_can = torch.bmm(R.permute(0,2,1), rays_d) / S
        bbox = torch.tensor(self.model_bbox, dtype=rays_o.dtype, device=rays_o.device)
        mask_in_box_batch = []
        near_batch = []
        far_batch = []
        for b in range(B):
            norm_d = torch.linalg.norm(rays_d_can[b], axis=0, keepdims=True)
            viewdir = rays_d_can[b] / norm_d
            viewdir[(viewdir < 1e-5) & (viewdir > -1e-10)] = 1e-5
            viewdir[(viewdir > -1e-5) & (viewdir < 1e-10)] = -1e-5
            tmin = (bbox[:, :1] - rays_o_can[b, :, :1]) / viewdir
            tmax = (bbox[:, 1:2] - rays_o_can[b, :, :1]) / viewdir
            t1 = torch.minimum(tmin, tmax)
            t2 = torch.maximum(tmin, tmax)
            near = torch.max(t1, 0)[0]
            far = torch.min(t2, 0)[0]
            mask_in_box = near < far
            mask_in_box_batch.append(mask_in_box)
            near_batch.append((near / norm_d[0]))
            far_batch.append((far / norm_d[0]))
        mask_in_box_batch = torch.stack(mask_in_box_batch)
        near_batch = torch.stack(near_batch)
        far_batch = torch.stack(far_batch)
        return near_batch, far_batch, mask_in_box_batch
    
    @staticmethod
    def unify_samples(depths1, colors1, densities1, depths2, colors2, densities2):
        all_depths = torch.cat([depths1, depths2], dim = -2)
        all_colors = torch.cat([colors1, colors2], dim = -2)
        all_densities = torch.cat([densities1, densities2], dim = -2)

        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))

        return all_depths, all_colors, all_densities
    
    @staticmethod
    def sample_pdf(density, z_vals, rays_d, N_importance):
        r"""sample_pdf function from another concurrent pytorch implementation
        by yenchenlin (https://github.com/yenchenlin/nerf-pytorch).
        """
        bins = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])

        _, _, _, weights = NeuralCameraModule.integrate(density, z_vals, rays_d)
        weights = weights[..., 1:-1] + 1e-5
        pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

        u = torch.rand(list(cdf.shape[:-1]) + [N_importance], dtype=weights.dtype, device=weights.device)

        u = u.contiguous()
        cdf = cdf.contiguous()
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds - 1), inds - 1)
        above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
        inds_g = torch.stack((below, above), dim=-1)

        matched_shape = (inds_g.shape[0], inds_g.shape[1], cdf.shape[-1])
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        sample_z = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
        sample_z, _ = torch.sort(sample_z, dim=-1)

        return sample_z
    
    
    @staticmethod
    def integrate(density, z_vals, rays_d, color=None):
        '''Transforms module's predictions to semantically meaningful values.
        Args:
            density: [num_rays, num_samples along ray, 4]. Prediction from module.
            z_vals: [num_rays, num_samples along ray]. Integration time.
            rays_d: [num_rays, 3]. Direction of each ray.
        Returns:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            acc_map: [num_rays]. Sum of weights along each ray.
            depth_map: [num_rays]. Estimated distance to object.
        '''
        dists = (z_vals[...,1:] - z_vals[...,:-1]) * 1e2 # [N_rays, N_samples-1]
        dists = torch.cat([dists, torch.ones(1, device=density.device).expand(dists[..., :1].shape) * 1e10], -1)  # [N_rays, N_samples]
        dists = dists * torch.norm(rays_d[...,None,:], dim=-1) # [N_rays, N_samples]
        alpha = 1 - torch.exp(-F.relu(density[...,0])*dists) # [N_rays, N_samples]
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=density.device), 1.-alpha + 1e-10], -1), -1)[:, :-1] # [N_rays, N_samples+1]
        acc_map = torch.sum(weights, -1)
        depth_map = torch.sum(weights * z_vals, -1)
        if color is None:
            return None, acc_map, depth_map, weights
        rgb_map = torch.sum(weights[..., None] * color, -2)
        return rgb_map, acc_map, depth_map, weights
    
    def sample_importance(self, z_vals, weights, N_importance):
        """
        Return depths of importance sampled points along rays. See NeRF importance sampling for more.
        """
        with torch.no_grad():
            batch_size, num_rays, samples_per_ray, _ = z_vals.shape

            z_vals = z_vals.reshape(batch_size * num_rays, samples_per_ray)
            weights = weights.reshape(batch_size * num_rays, -1)

            # smooth weights
            weights = torch.nn.functional.max_pool1d(weights.unsqueeze(1).float(), 2, 1, padding=1)
            weights = torch.nn.functional.avg_pool1d(weights, 2, 1).squeeze()
            weights = weights + 0.01

            z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
            importance_z_vals = self.sample_pdf(z_vals_mid, weights[:, 1:-1],
                                             N_importance).detach().reshape(batch_size, num_rays, N_importance)
        return importance_z_vals

    def render_rays(self, data, N_samples=64, triplane=[], scale=1, coarse2fine=False, do_detach=False):
        B, C, N = data['rays_o'].shape

        rays_o = rearrange(data['rays_o'], 'b c n -> (b n) c')
        rays_d = rearrange(data['rays_d'], 'b c n -> (b n) c')
        N_rays = rays_o.shape[0]
        rays_nf = rearrange(data['rays_nf'], 'b c n -> (b n) c')
        near, far = rays_nf[...,:1], rays_nf[...,1:] # [-1,1]
        
        if coarse2fine:
            t_vals = torch.linspace(0., 1., steps=N_samples, device=rays_o.device).unsqueeze(0)
            z_vals = near*(1-t_vals) + far*t_vals
            z_vals = z_vals.expand([N_rays, N_samples])
            
            # get coarse density
            query_pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
            query_pts = rearrange(query_pts, '(b n) s c -> b c (n s)', b=B)
            query_viewdirs = rays_d / torch.norm(rays_d, dim=1, keepdim=True)
            query_viewdirs = rearrange(query_viewdirs.unsqueeze(1).repeat(1, N_samples, 1), '(b n) s c -> b c (n s)', b=B)
            data['query_pts'] = query_pts
            data['query_viewdirs'] = query_viewdirs
            
            data['triplane'] = triplane[0]
            with torch.no_grad():
                data = self.avatarmodule('head', data)
            density_coarse = rearrange(data['density'], 'b c (n s) -> (b n) s c', n=N)
            z_vals = NeuralCameraModule.sample_pdf(density_coarse, z_vals, rays_d, N_samples)
            z_vals_fine = z_vals.clone()
        else:
            t_vals = torch.linspace(0., 1., steps=N_samples, device=rays_o.device).unsqueeze(0)
            z_vals = near*(1-t_vals) + far*t_vals
            z_vals = z_vals.expand([N_rays, N_samples])
            z_vals_fine = z_vals.clone()
        

        # ---------- fine (doing tri-plane) ----------
        query_pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
        query_pts = rearrange(query_pts, '(b n) s c -> b c (n s)', b=B) # (2, 16384, 64, 3) -> (2, 3, (16384, 64))
        query_viewdirs = rays_d / torch.norm(rays_d, dim=1, keepdim=True)
        query_viewdirs = rearrange(query_viewdirs.unsqueeze(1).repeat(1, N_samples, 1), '(b n) s c -> b c (n s)', b=B) # (2, 16384, 64, 3) -> (2, 3, (16384, 64))
        data['query_pts'] = query_pts
        data['query_viewdirs'] = query_viewdirs
        
        if len(triplane) == 1:
            data['triplane'] = triplane[0]
            # get the triplane features
            data = self.avatarmodule('head', data)
            
            density_fine = rearrange(data['density'], 'b c (n s) -> (b n) s c', n=N)
            color_fine = rearrange(data['color'], 'b c (n s) -> (b n) s c', n=N)
            
            # add a large value to the end of each ray
            density_fine = torch.cat([density_fine, torch.ones([density_fine.shape[0], 1, density_fine.shape[2]], device=density_fine.device) * 1e8], 1)
            color_fine = torch.cat([color_fine, torch.ones([color_fine.shape[0], 1, color_fine.shape[2]], device=color_fine.device)], 1)
            z_vals_fine = torch.cat([z_vals_fine, torch.ones([z_vals_fine.shape[0], 1], device=z_vals_fine.device) * 1e8], 1)
            
            render_image, render_mask, _, weights = NeuralCameraModule.integrate(density_fine, z_vals_fine, rays_d, color_fine)
            
            render_image = rearrange(render_image, '(b n) c -> b c n', b=B)
            render_mask = rearrange(render_mask, '(b n c) -> b c n', b=B, c=1)
            
            # if do_detach:
            #     render_image = render_image.detach()
            #     render_mask = render_mask.detach()
            
            data.update({f'render_image': render_image, 'render_mask': render_mask})

        if len(triplane) > 1:

            # data['query_pts'] = query_pts
            data['triplane'] = triplane

            # the rest is the same as above
            data = self.avatarmodule('head', data)
            
            density_fine = rearrange(data['density'], 'b c (n s) -> (b n) s c', n=N)
            color_fine = rearrange(data['color'], 'b c (n s) -> (b n) s c', n=N)
            
            # # add a large value to the end of each ray
            density_fine = torch.cat([density_fine, torch.ones([density_fine.shape[0], 1, density_fine.shape[2]], device=density_fine.device) * 1e8], 1)
            color_fine = torch.cat([color_fine, torch.ones([color_fine.shape[0], 1, color_fine.shape[2]], device=color_fine.device)], 1)
            z_vals_fine = torch.cat([z_vals_fine, torch.ones([z_vals_fine.shape[0], 1], device=z_vals_fine.device) * 1e8], 1)
            
            render_image, render_mask, _, weights = NeuralCameraModule.integrate(density_fine, z_vals_fine, rays_d, color_fine)
            render_image = rearrange(render_image, '(b n) c -> b c n', b=B)
            render_mask = rearrange(render_mask, '(b n c) -> b c n', b=B, c=1)
            
            if do_detach:
                render_image = render_image.detach()
                render_mask = render_mask.detach()
            
            data.update({f'local_{scale}_render_image': render_image, f'local_{scale}_render_mask': render_mask})
        return data
    

    def forward(self, data, resolution, local_enhance=False, coarse2fine=True, active_num=4):
        B = data['exp_code_3d'].shape[0]
        H = W = ray_size = resolution // 4 # ray size
        device = data['exp_code_3d'].device
        data['scale_list'] = []
        
        triplanes = self.generate_triplane(data['exp_code_3d'])
        
        
        local_infos = []
        render_res = {}
        
        for i, scale in enumerate([1, 0.5, 0.25]):
            render_data, local_info = self.generate_rays(data, H, W, B, device, scale=scale)
            # render the rays for the whole image
            render_data = self.render_rays(render_data, N_samples=self.N_samples, triplane=triplanes[:i+1], scale=scale, coarse2fine=coarse2fine)
            local_infos.append(local_info)
            
            # use the render data to update render_res
            render_res.update(render_data)
        
        # get the render image
        render_feature = rearrange(render_res['render_image'], 'b c (h w) -> b c h w', h=H) # [B, 32, 128, 128]
        render_mask = rearrange(render_res['render_mask'], 'b c (h w) -> b c h w', h=H)
        data['render_feature'] = render_feature
        data['render_mask'] = render_mask
        
        if len(render_res['scale_list']) > 1:
            remain_scale_list = render_res['scale_list'][1:]
            for scale in remain_scale_list:
                data[f'render_local_{scale}_feature'] = rearrange(render_res[f'local_{scale}_render_image'], 'b c (h w) -> b c h w', h=H) # [B, 32, 128, 128]
                data[f'render_local_{scale}_mask'] = rearrange(render_res[f'local_{scale}_render_mask'], 'b c (h w) -> b c h w', h=H)
                
                # enhance the local features
                # 1. downsample the render_img to be 128x128
                # 2. concat the render_img with the local feature
                if local_enhance:
                    down_render_img = F.interpolate(render_image, size=(128, 128), mode='bilinear', align_corners=False)
                    local_feature = torch.cat([down_render_img, data[f'render_local_{scale}_feature']], dim=1)
                    data[f'render_local_{scale}_enhance'] = self.avatarmodule('locupsample', local_feature)
            data['local_info'] = local_infos[1:]
        
        # the third tri-plane rendering
        
        local_scale = 0.25
        combined_resolution = int(ray_size / local_scale)
        combined_fake_image = torch.zeros((B, 32, combined_resolution, combined_resolution), device=device)
        combined_fake_mask = torch.zeros((B, 1, combined_resolution, combined_resolution), device=device)
        rays_o_lists, rays_d_lists, local_infos = self.gen_scale_rays(data['extrinsic'], 
                                                      data['intrinsic'], 
                                                      torch.FloatTensor([[H, W]]).repeat(B, 1), 
                                                      torch.FloatTensor([[self.image_size, self.image_size]]).repeat(B, 1), local_scale=local_scale)

        num_rays = len(rays_o_lists)     
            
        # randomly select active rays, with the num of active rays equals to undetached_num, from 16 patches
        active_indexes = np.random.choice(num_rays, active_num, replace=False)
        detach_list = [False if i in active_indexes else True for i in range(num_rays)]

        for i, (rays_o_grid, rays_d_grid, local_info) in enumerate(zip(rays_o_lists, rays_d_lists, local_infos)):
            rays_o = rearrange(rays_o_grid, 'b c h w -> b c (h w)')
            rays_d = rearrange(rays_d_grid, 'b c h w -> b c (h w)')
            
            rays_nf = self.gen_near_far_fixed(self.near_far[0], self.near_far[1], rays_o.shape[2], B, device)
            R = so3_exponential_map(data['pose'][:, :3])
            T = data['pose'][:, 3:, None] # for X.shape==Bx3XN : RX+T ; R^-1(X-T)
            S = data['scale'][:, :, None]
            rays_near_bbox, rays_far_bbox, mask_in_box = self.gen_near_far(rays_o, rays_d, R, T, S)
            for b in range(B):
                rays_nf[b, 0, mask_in_box[b]] = rays_near_bbox[b, mask_in_box[b]]
                rays_nf[b, 1, mask_in_box[b]] = rays_far_bbox[b, mask_in_box[b]]
            
            render_data = {
                'exp_code_3d': data['exp_code_3d'],
                'pose': data['pose'],
                'scale': data['scale'],
                'rays_o': rays_o,
                'rays_d': rays_d,
                'rays_nf': rays_nf
            }
            
            render_res_local = self.render_rays(render_data, N_samples=self.N_samples, triplane=triplanes[:2], scale=0.25, coarse2fine=coarse2fine, do_detach=detach_list[i])
            render_local_feature = rearrange(render_res_local[f'local_{local_scale}_render_image'], 'b c (h w) -> b c h w', h=H) # [B, 32, 128, 128]
            render_local_mask = rearrange(render_res_local[f'local_{local_scale}_render_mask'], 'b c (h w) -> b c h w', h=H)
            local_info = [int(item * (ray_size / local_scale / resolution)) for item in local_info]
            combined_fake_image[:, :, local_info[2]:local_info[3], local_info[0]:local_info[1]] = render_local_feature
            combined_fake_mask[:, :, local_info[2]:local_info[3], local_info[0]:local_info[1]] = render_local_mask
        
        render_image = self.avatarmodule('upsample', combined_fake_image) # [B, 3, 512, 512]
        
        data['render_image'] = render_image
        data['render_local_combined'] = combined_fake_image
        data['render_mask_combined'] = combined_fake_mask
        data['render_feature'] = render_feature
        data['render_mask'] = render_mask
                
        return data
    
    # def local_reenact(self, data, resolution):
        
    #     lr_scale = 0.25

    #     B = data['exp_code_3d'].shape[0]
    #     H = W = int(resolution * lr_scale)#4
    #     device = data['exp_code_3d'].device
        
    #     triplanes = []
    #     data['scale_list'] = []
    #     # generate triplanes
    #     triplane = super_triplane = None
        
    #     triplanes = self.generate_triplane(data['exp_code_3d'])
            
    #     combined_fake_image = torch.zeros((B, 3, resolution, resolution), device=device)
        
        
    #     render_data, local_info = self.generate_rays(data, H, W, B, device, scale=1)
            
    #     # render the rays for the whole image
    #     render_data = self.render_rays(render_data, N_samples=self.N_samples, triplane=triplanes[:1], scale=1)
        
    #     render_feature = rearrange(render_data['render_image'], 'b c (h w) -> b c h w', h=H) # [B, 32, 128, 128]
    #     render_image = self.avatarmodule('upsample', render_feature) # [B, 3, 512, 512]
        
    #     rays_o_lists, rays_d_lists, local_infos = self.gen_scale_rays(data['extrinsic'], 
    #                                                   data['intrinsic'], 
    #                                                   torch.FloatTensor([[H, W]]).repeat(B, 1), 
    #                                                   torch.FloatTensor([[self.image_size, self.image_size]]).repeat(B, 1), local_scale=lr_scale)

        
    #     for rays_o_grid, rays_d_grid, local_info in zip(rays_o_lists, rays_d_lists, local_infos):
    #         rays_o = rearrange(rays_o_grid, 'b c h w -> b c (h w)')
    #         rays_d = rearrange(rays_d_grid, 'b c h w -> b c (h w)')
            
    #         rays_nf = self.gen_near_far_fixed(self.near_far[0], self.near_far[1], rays_o.shape[2], B, device)
    #         R = so3_exponential_map(data['pose'][:, :3])
    #         T = data['pose'][:, 3:, None] # for X.shape==Bx3XN : RX+T ; R^-1(X-T)
    #         S = data['scale'][:, :, None]
    #         rays_near_bbox, rays_far_bbox, mask_in_box = self.gen_near_far(rays_o, rays_d, R, T, S)
    #         for b in range(B):
    #             rays_nf[b, 0, mask_in_box[b]] = rays_near_bbox[b, mask_in_box[b]]
    #             rays_nf[b, 1, mask_in_box[b]] = rays_far_bbox[b, mask_in_box[b]]
            
    #         render_data = {
    #             'exp_code_3d': data['exp_code_3d'],
    #             'pose': data['pose'],
    #             'scale': data['scale'],
    #             'rays_o': rays_o,
    #             'rays_d': rays_d,
    #             'rays_nf': rays_nf
    #         }
    #         render_res_local = self.render_rays(render_data, N_samples=self.N_samples, triplane=triplanes[:2], scale=lr_scale)
    #         render_local_feature = rearrange(render_res_local[f'local_{lr_scale}_render_image'], 'b c (h w) -> b c h w', h=H) # [B, 32, 128, 128]
    #         render_local_mask = rearrange(render_res_local[f'local_{lr_scale}_render_image'], 'b c (h w) -> b c h w', h=H)
            
    #         down_render_img = F.interpolate(render_image, size=(128, 128), mode='bilinear', align_corners=False)
    #         local_feature = torch.cat([down_render_img, render_local_feature], dim=1)
    #         render_fine_local = self.avatarmodule('locupsample', local_feature) # [B, 3, 512, 512]

    #         combined_fake_image[:, :, local_info[2]:local_info[3], local_info[0]:local_info[1]] = render_fine_local
        
    
    #     data['render_image'] = combined_fake_image
    #     data['render_feature'] = render_image
    #     data['render_mask'] = torch.ones_like(combined_fake_image)
    #     return data
    
    
    def generate_triplane(self, code):
        triplanes = self.generator(code)
        return triplanes
    
    def generate_rays(self, data, H, W, B, device, scale=1):
        rays_o_grid, rays_d_grid, local_info = self.gen_part_rays(data['extrinsic'], 
                                                      data['intrinsic'], 
                                                      torch.FloatTensor([[H, W]]).repeat(B, 1), 
                                                      torch.FloatTensor([[self.image_size, self.image_size]]).repeat(B, 1), local_scale=scale)

        rays_o = rearrange(rays_o_grid, 'b c h w -> b c (h w)')
        rays_d = rearrange(rays_d_grid, 'b c h w -> b c (h w)')

        rays_nf = self.gen_near_far_fixed(self.near_far[0], self.near_far[1], rays_o.shape[2], B, device)
        R = so3_exponential_map(data['pose'][:, :3])
        T = data['pose'][:, 3:, None] # for X.shape==Bx3XN : RX+T ; R^-1(X-T)
        S = data['scale'][:, :, None]
        rays_near_bbox, rays_far_bbox, mask_in_box = self.gen_near_far(rays_o, rays_d, R, T, S)
        for b in range(B):
            rays_nf[b, 0, mask_in_box[b]] = rays_near_bbox[b, mask_in_box[b]]
            rays_nf[b, 1, mask_in_box[b]] = rays_far_bbox[b, mask_in_box[b]]

        data['scale_list'].append(scale)
        render_data = {
            'exp_code_3d': data['exp_code_3d'],
            'pose': data['pose'],
            'scale': data['scale'],
            'rays_o': rays_o,
            'rays_d': rays_d,
            'rays_nf': rays_nf,
            'scale_list': data['scale_list']
        }
        return render_data, local_info
