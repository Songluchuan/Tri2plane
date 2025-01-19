import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import copy
import lpips
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import torchvision

def set_requires_grad(model, flag=False):
    for param in model.parameters():
        param.requires_grad = flag

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class SuperTrainer():
    def __init__(self, dataloader, avatarmodule, camera, discriminator, optimizer, optimizer_D, recorder, gpu_id):
        self.dataloader = dataloader
        self.avatarmodule = avatarmodule
        self.camera = camera
        self.discriminator = discriminator
        self.optimizer = optimizer
        self.optimizer_D = optimizer_D
        self.recorder = recorder
        self.device = torch.device('cuda:%d' % gpu_id)
        self.fn_lpips = lpips.LPIPS(net='vgg').to(self.device)
        self.gan_criteria = GANLoss(use_lsgan=True, tensor = torch.cuda.FloatTensor).to(self.device)
    
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

                data = self.camera(data, image.shape[2])
                render_image = data['render_image'][:, :3, :, :]
                render_image_coarse = data['render_feature'][:,0:3,:,:]
                # render_mask_coarse = data['render_mask']
                
                #### train discriminator ###
                set_requires_grad(self.discriminator, True)
                
                upsample_render_image_coarse = F.interpolate(render_image_coarse, size=render_image.shape[2:], mode='bilinear', align_corners=True)
                fake_output = torch.cat([render_image.detach(), upsample_render_image_coarse.detach(), condition_image[:, :-1, :, :]], dim=1)
                fake_pred = self.discriminator(fake_output)
                loss_d_fake = self.gan_criteria(fake_pred, False)
                
                image_coarse_upsample = F.interpolate(image_coarse, size=image.shape[2:], mode='bilinear', align_corners=True)
                real_output = torch.cat([image, image_coarse_upsample, condition_image[:, :-1, :, :]], dim=1)
                real_pred = self.discriminator(real_output)
                loss_d_real = self.gan_criteria(real_pred, True)
                
                loss_d = (loss_d_fake + loss_d_real) * 0.5
                
                self.optimizer_D.zero_grad()
                loss_d.backward()
                self.optimizer_D.step()
                
                #### train generator ###
                
                set_requires_grad(self.discriminator, False)
                
                loss_local_vgg = torch.tensor(0.0).to(self.device)
                loss_local_enhance = torch.tensor(0.0).to(self.device)
                
                if 'local_info' in data:
                    for scale, local_info in zip(data['scale_list'][1:], data['local_info']):
                        render_local_feature = data[f'render_local_{scale}_feature'][:,0:3,:,:]
                        render_local_enhance = data[f'render_local_{scale}_enhance'][:,0:3,:,:]
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
                        loss_local_enhance += (self.fn_lpips(render_local_enhance - 0.5, local_reals - 0.5).mean() + F.l1_loss(render_local_enhance, local_reals))

                if image.shape[2] != render_image.shape[2]:
                    render_image = F.interpolate(render_image, size=image.shape[2:], mode='bilinear', align_corners=True)
                loss_rgb = F.l1_loss(render_image, image)
                if render_image_coarse.shape[2] == 512:
                    image_coarse = image
                loss_rgb_coarse = F.l1_loss(render_image_coarse, image_coarse)
                # loss_mask = F.mse_loss(render_mask_coarse, mask_coarse)
                loss_vgg = self.fn_lpips(render_image - 0.5, image - 0.5).mean()
                
                
                fake_output = torch.cat([render_image, upsample_render_image_coarse, condition_image[:, :-1, :, :]], dim=1)
                fake_pred = self.discriminator(fake_output)
                loss_gan = self.gan_criteria(fake_pred, True) * 0.1
                
                loss = loss_rgb + loss_rgb_coarse * 10 + loss_vgg * 1e-1 + loss_local_vgg * 1e-1 + loss_gan + loss_local_enhance * 0.1#+ loss_mask * 1e-1
                print('[epoch][idx]:', epoch, idx, ' -[loss]: %.3f'%loss.item(), ' -[loss_rgb]: %.3f'%loss_rgb.item(), ' -[loss_rgb_coarse]: %.3f'%loss_rgb_coarse.item(),  ' -[loss_vgg]: %.3f'%loss_vgg.item(), ' -[loss_gan]: %.3f'%loss_gan.item(),
                      ' -[loss_local_vgg]: %.3f'%loss_local_vgg.item(), ' -[loss_local_enhance]: %.3f'%loss_local_enhance.item())

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
