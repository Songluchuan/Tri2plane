from config_train.config import config_avatar
import argparse
import torch
from torch.nn import DataParallel
import os

from lib.dataset.Dataset import TrainAvatarDataset
from lib.module.AvatarModule import AvatarModule
from lib.module.Tri2Plane_Camera import NeuralCameraModule
from lib.recorder.Recorder import InferRecorder
from lib.trainer.AvatarReenact import AvatarReenact
from lib.utils.util_seed import seed_everything

# runme: CUDA_VISIBLE_DEVICES=2 python reenactment_patch_detach.py

if __name__ == '__main__':
    seed_everything(11111)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_train/reenactment.yaml')
    arg = parser.parse_args()

    cfg = config_avatar()
    cfg.load(arg.config)
    cfg = cfg.get_cfg()

    dataset = TrainAvatarDataset(cfg.dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, pin_memory=True, num_workers=8)

    device = torch.device('cuda:%d' % cfg.gpu_ids[0])
    avatarmodule = AvatarModule(cfg.avatarmodule).to(device)
    avatarmodule.load_state_dict(torch.load(cfg.load_avatarmodule_checkpoint, map_location=lambda storage, loc: storage))
    
    neural_camera = NeuralCameraModule(avatarmodule, cfg.neuralcamera).to(device)
    optimizer = torch.optim.Adam(avatarmodule.parameters(), lr=cfg.lr)
    recorder = InferRecorder(cfg.recorder)

    Reenacter = AvatarReenact(dataloader, avatarmodule, neural_camera, optimizer, recorder, cfg.gpu_ids[0])
    Reenacter.Reenact()
