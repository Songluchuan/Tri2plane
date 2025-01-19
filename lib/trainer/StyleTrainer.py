import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import copy
import lpips
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
# from lib.preprocess.FaceAlignment import FaceAlignment

import torchvision

def set_requires_grad(model, flag=False):
    for param in model.parameters():
        param.requires_grad = flag


class StyleTrainer():
    def __init__(self, dataloader, avatarmodule, camera, discriminator, optimizer, optimizer_D, recorder, gpu_id):
        self.dataloader = dataloader
        self.avatarmodule = avatarmodule
        self.camera = camera
        self.discriminator = discriminator
        self.optimizer = optimizer
        self.optimizer_D = optimizer_D
        self.recorder = recorder
        self.device = torch.device('cuda:%d' % gpu_id)
        # self.facealignment = FaceAlignment(device='cuda:%d' % gpu_id)
        self.fn_lpips = lpips.LPIPS(net='vgg').to(self.device)
    
    def train(self, start_epoch=0, epochs=1):
        for epoch in range(start_epoch, epochs):
            for idx, data in tqdm(enumerate(self.dataloader)):

                to_cuda = ['image', 'iuv', 'mask', 'intrinsic', 'extrinsic', 'pose', 'scale', 'exp']
                for data_item in to_cuda:
                    data[data_item] = data[data_item].to(device=self.device)

                ### v2: condition_image from the iuv face
                condition_image = data['iuv']
                exp_code = data['exp']

                # torchvision.utils.save_image(condition_image, 'condition_image.jpg', normalize=True)
                if condition_image is None:
                    continue
                
                image = data['image']
                if idx < 100 and epoch == 0:
                    image = image - torch.randn_like(image).abs() * 0.1
                

                image_coarse = F.interpolate(image, scale_factor=0.25)
                mask = data['mask']
                mask_coarse = F.interpolate(mask, scale_factor=0.5)

                exp_code_2d = self.avatarmodule('encode', condition_image)
                # import pdb; pdb.set_trace()
                exp_code_3d = self.avatarmodule('mapping', exp_code)
                data['exp_code_3d'] = exp_code_3d
                
                #### train discriminator ###
                set_requires_grad(self.discriminator, True)
                
                data = self.camera.self_styleunet(data, image.shape[2])
                render_image = data['render_image'][:, :3, :, :]
                render_image_coarse = data['render_feature'][:,0:3,:,:]
                render_image_coarse = F.interpolate(render_image_coarse, size=render_image.shape[2:], mode='bilinear', align_corners=True)
                render_image2 = data['render_image2'][:, :3, :, :]
                render_image2 = F.interpolate(render_image2, size=render_image.shape[2:], mode='bilinear', align_corners=True)
                
                fake_output = torch.cat([render_image2.detach(), render_image_coarse.detach(), condition_image[:, :-1, :, :]], dim=1)
                fake_pred = self.discriminator(fake_output)
                
                image_coarse_upsample = F.interpolate(image_coarse, size=image.shape[2:], mode='bilinear', align_corners=True)
                real_output = torch.cat([image, image_coarse_upsample, condition_image[:, :-1, :, :]], dim=1)
                
                d_loss = F.softplus(fake_pred).mean() + F.softplus(-self.discriminator(real_output)).mean()
                
                self.optimizer_D.zero_grad()
                d_loss.backward()
                self.optimizer_D.step()
                

                ### train generator ###
                set_requires_grad(self.discriminator, False)
                
                loss_local_vgg = torch.tensor(0.0).to(self.device)
                
                if 'local_info' in data:
                    for scale, local_info in zip(data['scale_list'][1:], data['local_info']):
                        render_local_feature = data[f'render_local_{scale}_feature'][:,0:3,:,:]
                        local_reals = []
                        local_fakes = []
                        for i in range(len(local_info)):
                            local_real = image[i:i+1, :, local_info[i][2]:local_info[i][3], local_info[i][0]:local_info[i][1]]
                            local_fake = render_image[i:i+1, :, local_info[i][2]:local_info[i][3], local_info[i][0]:local_info[i][1]]
                            if local_real.shape[2] != render_local_feature[0].shape[2]:
                                # upsample the local_real or downsample the local_real
                                local_real = F.interpolate(local_real, size=render_local_feature.shape[2:], mode='bilinear', align_corners=True)
                            local_reals.append(local_real)
                            local_fakes.append(local_fake)

                        local_reals = torch.cat(local_reals, dim=0)
                        local_fakes = torch.cat(local_fakes, dim=0)

                        data[f'local_{scale}_real'] = local_reals
                        
                        # loss_local_real = F.l1_loss(render_local_feature, local_reals)
                        loss_local_vgg += self.fn_lpips(render_local_feature - 0.5, local_reals - 0.5).mean()

                if image.shape[2] != render_image.shape[2]:
                    render_image = F.interpolate(render_image, size=image.shape[2:], mode='bilinear', align_corners=True)
                loss_rgb = F.l1_loss(render_image, image)
                if image.shape[2] != render_image2.shape[2]:
                    render_image2 = F.interpolate(render_image2, size=image.shape[2:], mode='bilinear', align_corners=True)
                loss_super = F.l1_loss(render_image2, image)
                if render_image_coarse.shape[2] == 512:
                    image_coarse = image
                loss_rgb_coarse = F.l1_loss(render_image_coarse, image_coarse)
                # loss_mask = F.mse_loss(render_mask_coarse, mask_coarse)
                loss_vgg = self.fn_lpips(render_image - 0.5, image - 0.5).mean() + self.fn_lpips(render_image2 - 0.5, image - 0.5).mean()
                
                render_image_coarse = F.interpolate(render_image_coarse, size=render_image.shape[2:], mode='bilinear', align_corners=True)
                fake_output = torch.cat([render_image2, render_image_coarse, condition_image[:, :-1, :, :]], dim=1)
                fake_pred = self.discriminator(fake_output)
                loss_g = F.softplus(-fake_pred).mean() * 0.1
                
                loss = loss_rgb + loss_rgb_coarse * 10 + loss_vgg * 1e-1 + loss_local_vgg * 1e-1 + loss_super + loss_g #+ loss_mask * 1e-1
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                print('[epoch][idx]:', epoch, idx, ' -[loss]: %.3f'%loss.item(), ' -[loss_rgb]: %.3f'%loss_rgb.item(), ' -[loss_rgb_coarse]: %.3f'%loss_rgb_coarse.item(), ' -[loss_local_vgg]: %.3f'%loss_local_vgg.item(), ' -[loss_vgg]: %.3f'%loss_vgg.item(), ' -[loss_super]: %.3f'%loss_super.item(), ' -[loss_g]: %.3f'%loss_g.item(), ' -[d_loss]: %.3f'%d_loss.item())

                

                data['condition_image'] = condition_image
                data['render_image2'] = render_image2
                data['render_image'] = render_image
                data['render_image_coarse'] = render_image_coarse
                log = {
                    'data' : data,
                    'avatarmodule' : self.avatarmodule,
                    'discriminator' : self.discriminator,
                    'camera' : self.camera,
                    'loss_rgb': loss_rgb,
                    'loss_rgb_coarse': loss_rgb_coarse,
                    'loss_vgg': loss_vgg,
                    'epoch' : epoch,
                    'iter' : idx + epoch * len(self.dataloader)
                }
                self.recorder.log(log)
