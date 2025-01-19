import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import copy
import lpips
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
# from lib.preprocess.FaceAlignment import FaceAlignment

import torchvision


class AvatarTrainer():
    def __init__(self, dataloader, avatarmodule, camera, latent, optimizer, recorder, gpu_id):
        self.dataloader = dataloader
        self.avatarmodule = avatarmodule
        self.camera = camera
        self.optimizer = optimizer
        self.recorder = recorder
        self.device = torch.device('cuda:%d' % gpu_id)
        # self.facealignment = FaceAlignment(device='cuda:%d' % gpu_id)
        self.fn_lpips = lpips.LPIPS(net='vgg').to(self.device)
        self.latent_code = latent
    
    def train(self, start_epoch=0, epochs=1):
        for epoch in range(start_epoch, epochs):
            for idx, data in tqdm(enumerate(self.dataloader)):


                to_cuda = ['image', 'front_mtex', 'front_norm', 'mask', 'intrinsic', 'extrinsic', 'pose', 'scale', 'exp', 'img_index', 'index']
                for data_item in to_cuda:
                    data[data_item] = data[data_item].to(device=self.device)

                condition_image = torch.cat((data['front_mtex'], data['front_norm']), 1)
                condition_image = torch.clamp(condition_image + torch.randn_like(condition_image) * 0.05, 0.0, 1.0)

                img_index = [data['img_index'][i_img].item() for i_img in range(condition_image.shape[0])]
                latent_code = self.latent_code[img_index]
                exp_code = data['exp']
                if condition_image is None:
                    continue
                
                image = data['image']
                if idx < 100 and epoch == 0:
                    image = image - torch.randn_like(image).abs() * 0.1
                

                image_coarse = F.interpolate(image, scale_factor=0.25)
                mask = data['mask']
                
                mask_coarse = F.interpolate(mask, scale_factor=0.25)

                exp_code_2d = self.avatarmodule('encode', condition_image)
                exp_code = torch.cat([exp_code_2d, exp_code, latent_code], dim=1)
                exp_code_3d = self.avatarmodule('mapping', exp_code)
                data['exp_code_3d'] = exp_code_3d

                data = self.camera(data, image.shape[2])
                render_image = data['render_image']
                render_image_coarse = data['render_feature'][:,0:3,:,:]
                render_mask_coarse = data['render_mask']
                
                loss_local_vgg = torch.tensor(0.0).to(self.device)
                if 'local_info' in data:
                    for scale, local_info in zip(data['scale_list'][1:], data['local_info']):
                        render_local_feature = data[f'render_local_{scale}_feature'][:,0:3,:,:]
                       
                        local_reals = []
                        local_fakes = []
                        # import pdb; pdb.set_trace()
                        for i in range(len(local_info)):
                            local_real = image[i:i+1, :, local_info[i][2]:local_info[i][3], local_info[i][0]:local_info[i][1]]
                            local_fake = render_image[i:i+1, :, local_info[i][2]:local_info[i][3], local_info[i][0]:local_info[i][1]]
                            if local_real.shape[2] != render_local_feature[0].shape[2]:
                                local_real = F.interpolate(local_real, size=render_local_feature.shape[2:], mode='bilinear', align_corners=True)
                            local_reals.append(local_real)
                            local_fakes.append(local_fake)
                        local_reals = torch.cat(local_reals, dim=0)
                        local_fakes = torch.cat(local_fakes, dim=0)

                        data[f'local_{scale}_real'] = local_reals
                        loss_local_vgg += (self.fn_lpips(render_local_feature - 0.5, local_reals - 0.5).mean() + F.l1_loss(render_local_feature, local_reals))
                        
                
                if 'render_local_combined' in data:
                    render_local_combined = data['render_local_combined'][:,0:3,:,:]
                    scaled_img = F.interpolate(image, size=render_local_combined.shape[2:], mode='bilinear', align_corners=True)
                    loss_local_vgg += (self.fn_lpips(render_local_combined - 0.5, scaled_img - 0.5).mean() + F.l1_loss(render_local_combined, scaled_img))
                        
                loss_rgb = F.l1_loss(render_image, image)
                loss_rgb_coarse = F.l1_loss(render_image_coarse, image_coarse)
                loss_mask = F.mse_loss(render_mask_coarse, mask_coarse)
                loss_vgg = self.fn_lpips(render_image - 0.5, image - 0.5).mean()
                
                loss = loss_rgb + loss_rgb_coarse * 10 + loss_vgg * 1e-1 + loss_local_vgg + loss_mask * 1e-1#+ loss_local_enhance * 0.1 #+ loss_mask * 1e-1
                print('[epoch][idx]:', epoch, idx, ' -[loss_rgb]: %.3f'%loss_rgb.item(), ' -[loss_vgg]: %.3f'%loss_vgg.item(), ' -[loss_rgb_coarse]: %.3f'%loss_rgb_coarse.item(), ' -[loss_local_vgg]: %.3f'%loss_local_vgg.item())

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
                    'loss_rgb': loss_rgb,
                    'loss_rgb_coarse': loss_rgb_coarse,
                    'loss_vgg': loss_vgg,
                    'epoch' : epoch,
                    'iter' : idx + epoch * len(self.dataloader)
                }
                self.recorder.log(log)
