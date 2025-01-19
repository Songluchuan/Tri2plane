import torch
import torchvision as tv
from torch.utils.data import Dataset
import numpy as np
import glob
import os
import random

class TrainAvatarDataset(Dataset):

    def __init__(self, opt):
        super(TrainAvatarDataset, self).__init__()

        self.mode = opt.mode
        self.dataroot = opt.dataroot
        self.resolution = opt.resolution
        self.loader = tv.datasets.folder.default_loader
        self.transform = tv.transforms.Compose([tv.transforms.Resize(self.resolution), tv.transforms.ToTensor()])
        self.trans_cond = tv.transforms.Compose([tv.transforms.Resize(128), tv.transforms.ToTensor()])
        self.trans_src  = tv.transforms.Compose([tv.transforms.ToTensor()])

        self.intrinsic = torch.tensor([[5.0000e+03, 0.0000e+00, 2.5600e+02],
                                       [0.0000e+00, 5.0000e+03, 2.5600e+02],
                                       [0.0000e+00, 0.0000e+00, 1.0000e+00]]).float()
        self.extrinsic = torch.tensor([[1.0000,  0.0000,  0.0000,  0.0000],
                                       [0.0000, -1.0000,  0.0000,  0.0000],
                                       [0.0000,  0.0000, -1.0000,  4.0000]]).float()
        self.calibra_1 = torch.tensor([[[5.0000e+03,  0.0000e+00, -2.5600e+02],
                                        [0.0000e+00, -5.0000e+03, -2.5600e+02],
                                        [0.0000e+00,  0.0000e+00, -1.0000e+00]]]).float()#.to(device)
        self.calibra_1_inv = torch.inverse(self.calibra_1)
        
        self.samples = []
        video_folder = os.path.join(self.dataroot, opt.video_name)

        mode = opt.mode
        # mode = 'train'
        full_head_paths = sorted(glob.glob(os.path.join(video_folder, 'full_head_image', mode, '*')))
        self.i_train = 0

        for i, image_path in enumerate(full_head_paths):
            
            param_path = image_path.replace('full_head_image/%s/'%mode, 'crop_head_coeff/%s/params_'%mode).replace('jpg', 'npz')
            self.i_train += 1
            if not os.path.exists(param_path):
                continue
            
            front_mtex_path = image_path.replace('full_head_image', 'FrontCanonical_mtex').replace('.jpg', '.png')
            front_norm_path = image_path.replace('full_head_image', 'FrontCanonical_norm').replace('.jpg', '.png')
            front_mesh_path = image_path.replace('full_head_image', 'FrontCanonical_mesh').replace('.jpg', '.png')

            crop_head_path = image_path.replace('full_head_image', 'crop_head_image')
            
            mask_path = image_path.replace('full_head_image', 'full_head_masks')

            param = np.load(param_path)
            pose = torch.from_numpy(param['pose'])

            scale = torch.from_numpy(param['scale'])
            exp = torch.from_numpy(param['exp_coeff'])

            img_index = int(crop_head_path.split('/')[-1].replace('.jpg', ''))-1

            sample = (image_path, mask_path, pose, scale, exp, front_mtex_path, front_norm_path, front_mesh_path, crop_head_path, img_index)
            self.samples.append(sample)


    def get_item(self, index):
        data = self.__getitem__(index)
        return data
    
    def __getitem__(self, index):
        sample = self.samples[index]
        

        image_path = sample[0]
        full_image = self.trans_src(self.loader(image_path))
        full_mask_path = sample[1]
        full_mask = self.trans_src(self.loader(full_mask_path))
        full_image = full_image * full_mask + torch.ones_like(full_image) * (1 - full_mask)

        # depth = 4-pose[:,-1].item()
        # move = torch.bmm(calibra_1, delta_t.repeat(calibra_1.shape[0], 1, 1).permute(0,2,1))/depth

        exp = sample[4][0]
        pose = sample[2][0]
        scale = sample[3]

        if self.mode == 'train':
            move_x = random.randint(-127, 127)
            move_y = random.randint(-127, 127)
            move_point = torch.tensor([[[move_x],[move_y],[0]]])
            T_z = 4 - pose[-1].item()
            
            delta_Txy = torch.bmm(self.calibra_1_inv, T_z*move_point).view(-1)
            new_pose = pose.clone()
            new_pose[3:6] += delta_Txy

            ### move the bbox with : x --> -move_x and y --> -move_y
            image = full_image[:, 128-move_y: 128-move_y + 512, 128-move_x: 128-move_x + 512]
            mask  = full_mask[:, 128-move_y: 128-move_y + 512, 128-move_x: 128-move_x + 512]
            ## torchvision.utils.save_image(mask, 'debug.png')
        
        else:
            move_x = 0
            move_y = 0
            new_pose = pose.clone()
            image = full_image[:, 128-move_y: 128-move_y + 512, 128-move_x: 128-move_x + 512]
            mask  = full_mask[:, 128-move_y: 128-move_y + 512, 128-move_x: 128-move_x + 512]
            

        front_mtex = self.trans_cond(self.loader(sample[5]))
        front_norm = self.trans_cond(self.loader(sample[6]))
        front_mesh = self.trans_cond(self.loader(sample[7]))

        crop_head = self.trans_src(self.loader(sample[8]))

        intrinsic = self.intrinsic
        extrinsic = self.extrinsic
        index = torch.tensor(index).long()

        img_index = torch.tensor(sample[9]).long()


        return {'image': image,
                'mask': mask,
                'front_mtex': front_mtex,
                'front_norm': front_norm,
                'front_mesh': front_mesh,
                'intrinsic': intrinsic,
                'extrinsic': extrinsic,
                'exp': exp,
                'pose': new_pose,
                'crop_head': crop_head,
                'scale': scale,
                'img_index': img_index,
                'index': index}

    def __len__(self):
        return len(self.samples)



class TestAvatarDataset(Dataset):

    def __init__(self, opt):
        super(TestAvatarDataset, self).__init__()

        self.dataroot = opt.dataroot
        self.resolution = opt.resolution
        self.loader = tv.datasets.folder.default_loader
        self.transform = tv.transforms.Compose([tv.transforms.Resize(self.resolution), tv.transforms.ToTensor()])
        self.tran_cond = tv.transforms.Compose([tv.transforms.Resize(128), tv.transforms.ToTensor()])

        self.intrinsic = torch.tensor([[5.0000e+03, 0.0000e+00, 2.5600e+02],
                                       [0.0000e+00, 5.0000e+03, 2.5600e+02],
                                       [0.0000e+00, 0.0000e+00, 1.0000e+00]]).float()
        self.extrinsic = torch.tensor([[1.0000,  0.0000,  0.0000,  0.0000],
                                       [0.0000, -1.0000,  0.0000,  0.0000],
                                       [0.0000,  0.0000, -1.0000,  4.0000]]).float()
        
        self.samples = []
        video_folder = os.path.join(self.dataroot, opt.video_name)
        image_paths = sorted(glob.glob(os.path.join(video_folder, 'img_*')))

        train_video_folder = os.path.join(self.dataroot, opt.train_video_name)
        train_image_paths = sorted(glob.glob(os.path.join(train_video_folder, 'img_*')))
        train_param_path = train_image_paths[0].replace('img', 'params').replace('jpg', 'npz')
        train_param = np.load(train_param_path)
        train_pose = torch.from_numpy(train_param['pose'])
        train_scale = torch.from_numpy(train_param['scale'])
        
        for i, image_path in enumerate(image_paths):
            front_mtex_path = image_path.replace('img', 'FrontCanonical_mtex').replace('.jpg', '.png')
            front_norm_path = image_path.replace('img', 'FrontCanonical_norm').replace('.jpg', '.png')
            front_mesh_path = image_path.replace('img', 'FrontCanonical_mesh').replace('.jpg', '.png')

            mask_path = image_path.replace('img', 'mask')
            param_path = image_path.replace('img', 'params').replace('jpg', 'npz')
            if not os.path.exists(param_path):
                continue
            param = np.load(param_path)

            pose = torch.from_numpy(param['pose'])
            scale = train_scale#torch.from_numpy(param['scale'])
            exp = torch.from_numpy(param['exp_coeff'])
            sample = (image_path, mask_path, pose, scale, exp, front_mtex_path, front_norm_path, front_mesh_path)
            self.samples.append(sample)
        

    def get_item(self, index):
        data = self.__getitem__(index)
        return data
    
    def __getitem__(self, index):
        sample = self.samples[index]
        
        image_path = sample[0]
        image = self.transform(self.loader(image_path))
        mask_path = sample[1]
        mask = self.transform(self.loader(mask_path))
        image = image * mask + torch.ones_like(image) * (1 - mask)

        exp = sample[4][0]
        pose = sample[2][0]
        scale = sample[3]

        front_mtex = self.tran_cond(self.loader(sample[5]))
        front_norm = self.tran_cond(self.loader(sample[6]))
        front_mesh = self.tran_cond(self.loader(sample[7]))

        intrinsic = self.intrinsic
        extrinsic = self.extrinsic
        index = torch.tensor(index).long()

        return {'image': image,
                'iuv': front_mtex,
                'exp': exp,
                'mask': mask,
                'intrinsic': intrinsic,
                'extrinsic': extrinsic,
                'pose': pose,
                'scale': scale,
                'index': index}

    def __len__(self):
        return len(self.samples)



