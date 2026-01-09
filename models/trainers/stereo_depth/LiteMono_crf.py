# models/trainers/stereo_depth/LiteMono_crf.py

import torch
import torch.nn.functional as F

# ← 確認這個名字跟你 models/network/__init__.py 一致
from models.network import LiteMonoCRFDepth
from models.registry import MODELS
from models.trainers.stereo_depth.BaseModule import StereoDepthBaseModule
from models.metrics.eval_metric import compute_depth_errors, compute_disp_errors
from utils.visualization import *


@MODELS.register_module(name='LiteMono_crf')
class LiteMono_crf(StereoDepthBaseModule):
    def __init__(self, opt):
        super().__init__(opt)
        self.save_hyperparameters()

        # ======================
        # 1. 建立網路本體
        # ======================
        # 先只吃 maxdisp，做法跟 GWCNet 一樣：
        #   self.disp_net = GWCNetModel_GC(maxdisp=opt.model.max_disp)
        # 如果 LiteMonoCRFDepth __init__ 需要更多參數
        # （例如 use_concat_volume、encoder_model 等），
        # 在這裡自己加上去。
        self.disp_net = LiteMonoCRFDepth(
            maxdisp=opt.model.max_disp,
            # 如果network __init__ 有這些參數，就把註解打開：
            # use_concat_volume=getattr(opt.model, 'use_concat_volume', True),
            # use_3d_decoder=getattr(opt.model, 'use_3d_decoder', False),
            # encoder_model=opt.model.encoder,
            # pretrained_encoder=getattr(opt.model, 'pretrained_encoder', None),
        )

        # ======================
        # 2. Loss & 參數
        # ======================
        self.criterion = torch.nn.functional.smooth_l1_loss
        self.max_disp = opt.model.max_disp

        # 跟 GWCNet 一樣，預設 4 個尺度的 loss weight
        self.loss_weights = [0.5, 0.5, 0.7, 1.0]

    # ==========================
    # 3. Optimizer 參數
    # ==========================
    def get_optimize_param(self):
        optim_params = [
            {
                'params': self.disp_net.parameters(),
                'lr': self.optim_opt.learning_rate
            },
        ]
        return optim_params

    # ==========================
    # 4. Inference（推論用）
    # ==========================
    def inference_disp(self, left_img, right_img):
        B, C, H, W = left_img.shape
        if C == 1:
            # 和 GWCNet 寫法一致：灰階複製成 3 channel
            left_img = left_img.repeat_interleave(3, axis=1)
            right_img = right_img.repeat_interleave(3, axis=1)

        pred_disp_pyramid = self.disp_net(left_img, right_img)

        # 如果 LiteMonoCRFDepth 跟 GWCNet 一樣回傳 list / pyramid
        # 那就直接拿最後一個尺度
        if isinstance(pred_disp_pyramid, (list, tuple)):
            return pred_disp_pyramid[-1]

        # 如果只回傳單一張 disparity，就直接回傳
        return pred_disp_pyramid

    # ==========================
    # 5. 計算 loss（訓練用）
    # ==========================
    def multiscale_loss(self, pred_disp_pyramid, gt_disp):
        """
        pred_disp_pyramid:
        - 可以是 list/tuple of [B,H,W] (pyramid)
        - 也可以是單一張 [B,H,W] Tensor
        gt_disp: [B,H,W]，px disparity
        """
        mask = (gt_disp > 0) & (gt_disp < self.max_disp)
        all_losses = []

        H_gt, W_gt = gt_disp.size(-2), gt_disp.size(-1)

        # 確保一定是 list
        if isinstance(pred_disp_pyramid, (list, tuple)):
            pred_list = pred_disp_pyramid
        else:
            pred_list = [pred_disp_pyramid]

        for disp_est, weight in zip(pred_list, self.loss_weights):

            # 對齊解析度
            if disp_est.size(-1) != W_gt:
                H_pred, W_pred = disp_est.size(-2), disp_est.size(-1)

                disp_est = disp_est.unsqueeze(1)  # [B,1,H_pred,W_pred]
                disp_est = torch.nn.functional.interpolate(
                    disp_est,
                    size=(H_gt, W_gt),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)  # [B,H_gt,W_gt]

                disp_est = disp_est * (W_gt / W_pred)

            all_losses.append(
                weight * self.criterion(
                    disp_est[mask],
                    gt_disp[mask],
                    size_average=True
                )
            )

        return sum(all_losses)

    def get_losses(self, predictions, gt_disp):

        # predictions: (disp_pyramid_mono, disp_pyramid_stereo)
        disp_pyramid_mono, disp_pyramid_stereo = predictions

        loss_mono = self.multiscale_loss(disp_pyramid_mono, gt_disp)
        loss_stereo = self.multiscale_loss(disp_pyramid_stereo, gt_disp)

        # 跟原本的範本一樣，直接相加就好
        total_loss = loss_mono + loss_stereo
        return total_loss

    def on_validation_epoch_start(self):
        # clear buffer at the beginning of each val epoch
        self._val_outputs = []

    def validation_step(self, batch, batch_idx):
        left_img = batch["tgt_left"]
        right_img = batch["tgt_right"]
        left_vis = batch["tgt_left_eh"]
        depth_gt = batch["tgt_depth_gt"]
        disp_gt = batch["tgt_disp_gt"]
        focal = batch["focal"]
        baseline = batch["baseline"]

        pred_disp_mono, pred_disp_stereo = self.forward(left_img, right_img)

        pred_depth = baseline[0] * focal[0] / (pred_disp_mono + 1e-10)

        errs_depth_m = compute_depth_errors(depth_gt, pred_depth)
        errs_disp_m = compute_disp_errors(disp_gt, pred_disp_mono)

        pred_depth = baseline[0] * focal[0] / (pred_disp_stereo + 1e-10)

        errs_depth = compute_depth_errors(depth_gt, pred_depth)
        errs_disp = compute_disp_errors(disp_gt, pred_disp_stereo)

        errs = {'abs_rel': errs_depth[1], 'sq_rel': errs_depth[2],
                'rmse': errs_depth[4], 'rmse_log': errs_depth[5],
                'a1': errs_depth[6], 'a2': errs_depth[7], 'a3': errs_depth[8],
                'epe': errs_disp[0], 'd1': errs_disp[1], 'thres1': errs_disp[2],
                'thres2': errs_disp[3], 'thres3': errs_disp[4],
                'mono/abs_rel': errs_depth_m[1], 'mono/sq_rel': errs_depth_m[2],
                'mono/rmse': errs_depth_m[4], 'mono/rmse_log': errs_depth_m[5],
                'mono/a1': errs_depth_m[6], 'mono/a2': errs_depth_m[7], 'mono/a3': errs_depth_m[8],
                'mono/epe': errs_disp_m[0], 'mono/d1': errs_disp_m[1], 'mono/thres1': errs_disp_m[2],
                'mono/thres2': errs_disp_m[3], 'mono/thres3': errs_disp_m[4]}

        # plot
        if batch_idx < 2:
            if left_vis[0].size(-1) != pred_depth[0].size(-1):
                C, H, W = left_vis[0].size()
                pred_depth = torch.nn.functional.interpolate(
                    pred_depth, [H, W], mode='nearest')
                pred_disp_stereo = torch.nn.functional.interpolate(
                    pred_disp_stereo, [H, W], mode='nearest')

            vis_img = visualize_image(left_vis[0])  # (3, H, W)
            vis_disp = visualize_depth(
                pred_disp_stereo[0].squeeze())  # (3, H, W)
            vis_depth = visualize_depth(pred_depth[0].squeeze())  # (3, H, W)
            stack = torch.cat([vis_img, vis_disp, vis_depth],
                              dim=1).unsqueeze(0)  # (1, 3, 2*H, W)
            if hasattr(self.logger, "experiment") and hasattr(self.logger.experiment, "add_images"):
                self.logger.experiment.add_images(
                    f'val/img_disp_depth_{batch_idx}', stack, self.current_epoch)

        self._val_outputs.append(errs)

        if batch_idx == 0:
            print("gt_disp range:",
                  float(disp_gt.min()), float(disp_gt.max()))
            print("pred_disp_stereo range:",
                  float(pred_disp_stereo.min()), float(pred_disp_stereo.max()))
        return errs

    def on_validation_epoch_end(self):
        outputs = self._val_outputs

        import numpy as np
        mean_rel = np.array([x['abs_rel'] for x in outputs]).mean()
        mean_sq_rel = np.array([x['sq_rel'] for x in outputs]).mean()
        mean_rmse = np.array([x['rmse'] for x in outputs]).mean()
        mean_rmse_log = np.array([x['rmse_log'] for x in outputs]).mean()

        mean_a1 = np.array([x['a1'] for x in outputs]).mean()
        mean_a2 = np.array([x['a2'] for x in outputs]).mean()
        mean_a3 = np.array([x['a3'] for x in outputs]).mean()

        mean_epe = np.array([x['epe'] for x in outputs]).mean()
        mean_d1 = np.array([x['d1'] for x in outputs]).mean()
        mean_th1 = np.array([x['thres1'] for x in outputs]).mean()
        mean_th2 = np.array([x['thres2'] for x in outputs]).mean()
        mean_th3 = np.array([x['thres3'] for x in outputs]).mean()

        self.log('val_loss', mean_epe, prog_bar=True)

        self.log('val/abs_rel', mean_rel)
        self.log('val/sq_rel', mean_sq_rel)
        self.log('val/rmse', mean_rmse)
        self.log('val/rmse_log', mean_rmse_log)
        self.log('val/a1', mean_a1)
        self.log('val/a2', mean_a2)
        self.log('val/a3', mean_a3)

        self.log('val/epe', mean_epe)
        self.log('val/d1', mean_d1)
        self.log('val/th1', mean_th1)
        self.log('val/th2', mean_th2)
        self.log('val/th3', mean_th3)

        # mono stats
        mean_rel = np.array([x['mono/abs_rel'] for x in outputs]).mean()
        mean_sq_rel = np.array([x['mono/sq_rel'] for x in outputs]).mean()
        mean_rmse = np.array([x['mono/rmse'] for x in outputs]).mean()
        mean_rmse_log = np.array([x['mono/rmse_log'] for x in outputs]).mean()

        mean_a1 = np.array([x['mono/a1'] for x in outputs]).mean()
        mean_a2 = np.array([x['mono/a2'] for x in outputs]).mean()
        mean_a3 = np.array([x['mono/a3'] for x in outputs]).mean()

        mean_epe = np.array([x['mono/epe'] for x in outputs]).mean()
        mean_d1 = np.array([x['mono/d1'] for x in outputs]).mean()
        mean_th1 = np.array([x['mono/thres1'] for x in outputs]).mean()
        mean_th2 = np.array([x['mono/thres2'] for x in outputs]).mean()
        mean_th3 = np.array([x['mono/thres3'] for x in outputs]).mean()

        self.log('val/mono/abs_rel', mean_rel)
        self.log('val/mono/sq_rel', mean_sq_rel)
        self.log('val/mono/rmse', mean_rmse)
        self.log('val/mono/rmse_log', mean_rmse_log)
        self.log('val/mono/a1', mean_a1)
        self.log('val/mono/a2', mean_a2)
        self.log('val/mono/a3', mean_a3)

        self.log('val/mono/epe', mean_epe)
        self.log('val/mono/d1', mean_d1)
        self.log('val/mono/th1', mean_th1)
        self.log('val/mono/th2', mean_th2)
        self.log('val/mono/th3', mean_th3)
