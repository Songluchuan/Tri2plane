import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import copy
import lpips
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

import torchvision


class AvatarReenact():
    def __init__(self, dataloader, avatarmodule, camera, optimizer, recorder, gpu_id):
        self.dataloader = dataloader
        self.avatarmodule = avatarmodule
        self.camera = camera
        self.optimizer = optimizer
        self.recorder = recorder
        self.device = torch.device('cuda:%d' % gpu_id)
    
    def Reenact(self):
        for idx, data in tqdm(enumerate(self.dataloader)):

            to_cuda = ['image', 'front_mtex', 'front_norm', 'mask', 'intrinsic', 'extrinsic', 'pose', 'scale', 'exp', 'img_index', 'index']
            for data_item in to_cuda:
                data[data_item] = data[data_item].to(device=self.device)

            condition_image = torch.cat((data['front_mtex'], data['front_norm']), 1)
            exp_code = data['exp']
            
            image = data['image']

            with torch.no_grad():
                exp_code_2d = self.avatarmodule('encode', condition_image)
                exp_code = torch.cat([exp_code_2d, exp_code], dim=1)
                exp_code_3d = self.avatarmodule('mapping', exp_code)
                data['exp_code_3d'] = exp_code_3d
                data = self.camera(data, image.shape[2])

            render_image = data['render_image']
            render_image_coarse = data['render_feature'][:,0:3,:,:]
            render_mask_coarse = data['render_mask']

            # torchvision.utils.save_image(render_image, 'render_image.jpg')
            # torchvision.utils.save_image(data['render_local_combined'][:,:3], 'render_local_combined.jpg')

            data['condition_image'] = condition_image
            data['render_image'] = render_image
            data['render_image_coarse'] = render_image_coarse
            log = {
                'data' : data,
                # 'avatarmodule' : self.avatarmodule,
                # 'camera' : self.camera,
                # 'loss_rgb': loss_RGB,
                # 'loss_rgb_coarse': loss_RGB,
                # 'loss_vgg': loss_Perp,
                # 'epoch' : epoch,
                'fid' : idx,
            }
            self.recorder.log(log)
