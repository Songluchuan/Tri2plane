import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import copy
import lpips
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

import torchvision


class AvatarTrainer():
    def __init__(self, dataloader, avatarmodule, camera, optimizer, recorder, gpu_id):
        self.dataloader = dataloader
        self.avatarmodule = avatarmodule
        self.camera = camera
        self.optimizer = optimizer
        self.recorder = recorder
        self.device = torch.device('cuda:%d' % gpu_id)
        self.fn_lpips = lpips.LPIPS(net='vgg').to(self.device)
    
    def train(self, start_epoch=0, epochs=1):
        for epoch in range(start_epoch, epochs):
            for idx, data in tqdm(enumerate(self.dataloader)):

                to_cuda = ['image', 'front_mtex', 'front_norm', 'mask', 'intrinsic', 'extrinsic', 'pose', 'scale', 'exp', 'img_index', 'index']
                for data_item in to_cuda:
                    data[data_item] = data[data_item].to(device=self.device)

                condition_image = torch.cat((data['front_mtex'], data['front_norm']), 1)
                condition_image = torch.clamp(condition_image + torch.randn_like(condition_image) * 0.05, 0.0, 1.0)
                
                exp_code = data['exp']
                if condition_image is None:
                    continue
                
                image = data['image']
                if idx < 100 and epoch == 0:
                    image = image - torch.randn_like(image).abs() * 0.1
                
                image_coarse = F.interpolate(image, scale_factor=0.25)
                mask = data['mask']

                exp_code_2d = self.avatarmodule('encode', condition_image)
                exp_code = torch.cat([exp_code_2d, exp_code], dim=1)
                exp_code_3d = self.avatarmodule('mapping', exp_code)
                data['exp_code_3d'] = exp_code_3d

                data = self.camera(data, image.shape[2])
                render_image = data['render_image']
                render_mask = data['render_mask_combined']

                render_image_coarse = data['render_feature'][:,0:3,:,:]
                
                loss_Rgb = torch.tensor(0.0).to(self.device)
                loss_Perp = torch.tensor(0.0).to(self.device)
                if 'local_info' in data:
                    for scale, local_info in zip(data['scale_list'][1:], data['local_info']):
                        render_local_feature = data[f'render_local_{scale}_feature'][:,0:3,:,:]
                       
                        local_reals = []
                        for i in range(len(local_info)):
                            local_real = image[i:i+1, :, local_info[i][2]:local_info[i][3], local_info[i][0]:local_info[i][1]]
                            if local_real.shape[2] != render_local_feature[0].shape[2]:
                                local_real = F.interpolate(local_real, size=render_local_feature.shape[2:], mode='bilinear', align_corners=True)
                            local_reals.append(local_real)
                        local_reals = torch.cat(local_reals, dim=0)
                        data[f'local_{scale}_real'] = local_reals
                        loss_Rgb += F.l1_loss(render_local_feature, local_reals)
                        
                if 'render_local_combined' in data:
                    render_local_combined = data['render_local_combined'][:,0:3,:,:]
                    scaled_img = F.interpolate(image, size=render_local_combined.shape[2:], mode='bilinear', align_corners=True)
                    loss_Rgb += F.l1_loss(render_local_combined, scaled_img)
                        
                loss_Rgb += F.l1_loss(render_image_coarse, image_coarse)*10
                loss_Perp += F.l1_loss(render_image, image)
                loss_Perp += self.fn_lpips(render_image - 0.5, image - 0.5).mean() * 1e-1
                loss_Mask = F.mse_loss(render_mask, mask)
                
                loss = loss_Perp + loss_Rgb + loss_Mask * 1e-1

                print('[epoch][idx]:', epoch, idx, ' -[loss]: %.3f'%loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                data['condition_image'] = condition_image
                data['render_image'] = render_image
                data['render_image_coarse'] = render_image_coarse
                log = {
                    'data' : data,
                    'avatarmodule' : self.avatarmodule,
                    'camera' : self.camera,
                    'loss_rgb': loss_Rgb,
                    'loss_rgb_coarse': loss_Rgb,
                    'loss_vgg': loss_Perp,
                    'epoch' : epoch,
                    'iter' : idx + epoch * len(self.dataloader)
                }
                self.recorder.log(log)
