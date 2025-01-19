from config_train.config import config_avatar
import argparse
import torch
from torch.nn import DataParallel
import os

from lib.dataset.Dataset import TrainAvatarDataset
from lib.module.AvatarModule import AvatarModule
from lib.module.Tri2Plane_Camera import NeuralCameraModule
from lib.recorder.Recorder import TrainRecorder
from lib.trainer.AvatarTrainer import AvatarTrainer
from lib.utils.util_seed import seed_everything

# runme: CUDA_VISIBLE_DEVICES=2 python train_patch_detach.py --config config_train/YOUR.yaml

if __name__ == '__main__':
    seed_everything(11111)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_train/train_avatar_lizhen.yaml')
    arg = parser.parse_args()

    cfg = config_avatar()
    cfg.load(arg.config)
    cfg = cfg.get_cfg()

    dataset = TrainAvatarDataset(cfg.dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, pin_memory=True, num_workers=8)

    device = torch.device('cuda:%d' % cfg.gpu_ids[0])
    avatarmodule = AvatarModule(cfg.avatarmodule).to(device)
    if os.path.exists(cfg.load_avatarmodule_checkpoint):
        avatarmodule.load_state_dict(torch.load(cfg.load_avatarmodule_checkpoint, map_location=lambda storage, loc: storage), strict=False)   
    avatarmodule = DataParallel(avatarmodule, cfg.gpu_ids)
    
    neural_camera = NeuralCameraModule(avatarmodule, cfg.neuralcamera).to(device)
    optimizer = torch.optim.Adam(avatarmodule.parameters(), lr=cfg.lr)
    recorder = TrainRecorder(cfg.recorder)

    trainer = AvatarTrainer(dataloader, avatarmodule, neural_camera, optimizer, recorder, cfg.gpu_ids[0])
    trainer.train(cfg.start_epoch, 1000)
