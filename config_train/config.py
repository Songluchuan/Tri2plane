import os
from yacs.config import CfgNode as CN
 

class config_base():

    def __init__(self):
        self.cfg = CN()
    
    def get_cfg(self):
        return  self.cfg.clone()
    
    def load(self,config_file):
         self.cfg.defrost()
         self.cfg.merge_from_file(config_file)
         self.cfg.freeze()


class config_avatar(config_base):

    def __init__(self):
        super(config_avatar, self).__init__()

        self.cfg.start_epoch = 0
        self.cfg.load_avatarmodule_checkpoint = ''
        self.cfg.load_discriminator_checkpoint = ''
        self.cfg.load_yvaemodule_checkpoint = ''
        self.cfg.lr = 0.0
        self.cfg.gpu_ids = [0]
        self.cfg.batch_size = 1

        self.cfg.dataset = CN()
        self.cfg.dataset.dataroot = ''
        self.cfg.dataset.video_name = ''
        self.cfg.dataset.mode = ''
        self.cfg.dataset.resolution = 512
        self.cfg.dataset.views = []
        self.cfg.dataset.condition_views = []
        self.cfg.dataset.condition_folder = ''

        self.cfg.yvaemodule = CN()
        self.cfg.yvaemodule.exp_dim_2d = 256
        self.cfg.yvaemodule.encoder_dims = [3, 32, 64, 128, 256]
        self.cfg.yvaemodule.neck_dims = [512, 512]
        self.cfg.yvaemodule.decoder_dims = [512, 256, 128, 64, 4]
        self.cfg.yvaemodule.mapping_dims = [256, 256, 256, 128]
        self.cfg.yvaemodule.domains = []
        
        self.cfg.avatarmodule = CN()
        self.cfg.avatarmodule.exp_dim_2d = 256
        self.cfg.avatarmodule.encoder_dims = [3, 32, 64, 128, 256]
        self.cfg.avatarmodule.mapping_dims = [256, 256]
        self.cfg.avatarmodule.headmodule = CN()
        self.cfg.avatarmodule.headmodule.density_mlp = [256, 1]
        self.cfg.avatarmodule.headmodule.color_mlp = [256, 128, 128, 3]
        self.cfg.avatarmodule.headmodule.pos_freq = 10
        self.cfg.avatarmodule.headmodule.view_freq = 4
        self.cfg.avatarmodule.headmodule.noise = 0.0
        self.cfg.avatarmodule.headmodule.bbox = [[-0.15, 0.15], [-0.15, 0.15], [-0.2, 0.1]]
        self.cfg.avatarmodule.headmodule.triplane_res = 256
        self.cfg.avatarmodule.headmodule.triplane_dim = 32
        self.cfg.avatarmodule.headmodule.exp_dim_3d = 128
        self.cfg.avatarmodule.headmodule.triplane_n = 3
        self.cfg.avatarmodule.upsampler_capacity = 32

        self.cfg.srmodule = CN()
        self.cfg.srmodule.identities = ['']
        self.cfg.srmodule.dims = [16, 32, 64, 128, 256]

        self.cfg.neuralcamera = CN()
        self.cfg.neuralcamera.model_bbox = [[-0.15, 0.15], [-0.15, 0.15], [-0.2, 0.1]]
        self.cfg.neuralcamera.image_size = 1024
        self.cfg.neuralcamera.N_samples = 16
        self.cfg.neuralcamera.near_far = [0.0, 1.0]
        self.cfg.neuralcamera.super3d = True

        self.cfg.recorder = CN()
        self.cfg.recorder.name = ''
        self.cfg.recorder.logdir = ''
        self.cfg.recorder.checkpoint_path = ''
        self.cfg.recorder.result_path = ''
        self.cfg.recorder.save_freq = 1
        self.cfg.recorder.show_freq = 1
