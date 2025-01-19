from tensorboardX import SummaryWriter
import torch
import os
import numpy as np
import cv2

class TrainRecorder():
    def __init__(self, opt):
        self.logdir = opt.logdir
        self.logger = SummaryWriter(self.logdir)

        self.name = opt.name
        self.checkpoint_path = opt.checkpoint_path
        self.result_path = opt.result_path
        
        self.save_freq = opt.save_freq
        self.show_freq = opt.show_freq

        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs('%s/%s' % (self.checkpoint_path, self.name), exist_ok=True)
        os.makedirs('%s/%s' % (self.result_path, self.name), exist_ok=True)
    
    def log(self, log_data):
        self.logger.add_scalar('loss_rgb', log_data['loss_rgb'], log_data['iter'])
        self.logger.add_scalar('loss_rgb_coarse', log_data['loss_rgb_coarse'], log_data['iter'])
        self.logger.add_scalar('loss_vgg', log_data['loss_vgg'], log_data['iter'])

        if log_data['iter'] % self.save_freq == 0:
            print('saving checkpoint.')
            torch.save(log_data['avatarmodule'].module.state_dict(), '%s/%s/latest' % (self.checkpoint_path, self.name))
            torch.save(log_data['avatarmodule'].module.state_dict(), '%s/%s/epoch_%d' % (self.checkpoint_path, self.name, log_data['epoch']))
            if 'discriminator' in log_data:
                torch.save(log_data['discriminator'].state_dict(), '%s/%s/d_latest' % (self.checkpoint_path, self.name))
                torch.save(log_data['discriminator'].state_dict(), '%s/%s/d_epoch_%d' % (self.checkpoint_path, self.name, log_data['epoch']))
            
        if log_data['iter'] % self.show_freq == 0:
            print('saving recon results.')
            image = log_data['data']['image'][0].detach().permute(1,2,0).cpu().numpy()
            image = (image * 255).astype(np.uint8)[:,:,::-1]

            render_image = log_data['data']['render_image'][0].detach().permute(1,2,0).cpu().numpy()
            render_image = (render_image * 255).astype(np.uint8)[:,:,::-1]
            
            
            condition_image = torch.clamp(log_data['data']['condition_image'][:,:3,], 0.0, 1.0)[0].permute(1,2,0).detach().cpu().numpy()
            condition_image = (condition_image * 255).astype(np.uint8)[:,:,::-1]
            condition_image = cv2.resize(condition_image, (render_image.shape[0], render_image.shape[1]))

            render_image_coarse = torch.clamp(log_data['data']['render_image_coarse'], 0.0, 1.0)[0].detach().permute(1,2,0).cpu().numpy()
            render_image_coarse = (render_image_coarse * 255).astype(np.uint8)[:,:,::-1]
            render_image_coarse = cv2.resize(render_image_coarse, (render_image.shape[0], render_image.shape[1]))
            
            if 'local_info' in log_data['data']:

                image_list = [condition_image, render_image_coarse]

                if 'render_local_combined' in log_data['data']:
                    render_local_combined = log_data['data']['render_local_combined'][0][:3, :, :].detach().permute(1,2,0).cpu().numpy()
                    render_local_combined = (render_local_combined * 255).astype(np.uint8)[:,:,::-1]
                    render_local_combined = cv2.resize(render_local_combined, (render_image.shape[0], render_image.shape[1]))
                    image_list.append(render_local_combined)

                image_list.append(render_image)
                image_list.append(image)
                result = np.hstack(image_list)
            else:
                if 'render_image2' in log_data['data']:
                    render_image2 = log_data['data']['render_image2'][0].detach().permute(1,2,0).cpu().numpy()
                    render_image2 = (render_image2 * 255).astype(np.uint8)[:,:,::-1]
                    result = np.hstack((condition_image, render_image_coarse, render_image, render_image2, image))
                else:
                    result = np.hstack((condition_image, render_image_coarse, render_image, image))

            cv2.imwrite('%s/%s/result_%05d.jpg' % (self.result_path, self.name, log_data['iter']), result)


class InferRecorder():
    def __init__(self, opt):
        self.name = opt.name
        self.result_path = opt.result_path

        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs('%s/%s' % (self.result_path, self.name), exist_ok=True)

    def log(self, log_data):
        data = log_data['data']

        image = data['condition_image'][0].permute(1,2,0).cpu().numpy()[:,:,:3]
        image = cv2.resize((image * 255).astype(np.uint8)[:,:,::-1], (512,512))

        real_image = data['image'][0].permute(1,2,0).cpu().numpy()
        real_image = cv2.resize((real_image * 255).astype(np.uint8)[:,:,::-1], (512,512))

        image_list = []
        render_image = data['render_image'][0].permute(1,2,0).cpu().numpy()
        render_image = (render_image * 255).astype(np.uint8)[:,:,::-1]

        if 'render_local_combined' in log_data['data']:
            render_local_combined = log_data['data']['render_local_combined'][0][:3, :, :].detach().permute(1,2,0).cpu().numpy()
            render_local_combined = (render_local_combined * 255).astype(np.uint8)[:,:,::-1]
            render_local_combined = cv2.resize(render_local_combined, (render_image.shape[0], render_image.shape[1]))
            image_list = [image, render_image, render_local_combined]

        if 'local_info' in log_data['data']:
            for scale in log_data['data']['scale_list'][1:]:
                render_local = log_data['data'][f'render_local_{scale}_feature'][0][:3, :, :].detach().permute(1,2,0).cpu().numpy()
                render_local = (render_local * 255).astype(np.uint8)[:,:,::-1]
                render_local = cv2.resize(render_local, (render_image.shape[0], render_image.shape[1]))
                image_list.append(render_local)

        image_list.append(real_image)
        result = np.hstack(image_list)

        cv2.imwrite('%s/%s/image_%04d.jpg' % (self.result_path, self.name, log_data['fid']), result)

