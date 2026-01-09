import torch
from models.network import LiteMonoDepth
from models.losses.loss_depth import SilogLoss
from models.registry import MODELS
from models.trainers.mono_depth.BaseModule import MonoDepthBaseModule


@MODELS.register_module(name='LiteMono')
class LiteMono(MonoDepthBaseModule):
    def __init__(self, opt):
        super().__init__(opt)
        self.save_hyperparameters()

        # 讓 BaseModule 自動把 C=1 影像複製成 3 通道後再丟進網路
        self.depth_net = LiteMonoDepth(
            in_chans=3,
            height=opt.model.input_height,
            width=opt.model.input_width,
            min_depth=opt.model.min_depth,
            max_depth=opt.model.max_depth
        )

        self.criterion = SilogLoss()

    def get_optimize_param(self):
        return [{'params': self.depth_net.parameters(),
                 'lr': self.optim_opt.learning_rate}]

    def get_losses(self, pred_depth, gt_depth):
        return self.criterion(pred_depth, gt_depth)
