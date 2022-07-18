import json
import os
import sys
from os.path import join as pjoin
from matplotlib import transforms

import tensorboard
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch import optim

import os
import sys
from argparse import ArgumentParser
from statistics import median
from typing import Any, List
import collections

from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torchio as tio

from pytorch_lightning.core.lightning import LightningModule
from torch import nn
from torch.utils.data.dataloader import DataLoader
from Evaluation.evaluate import IOU, Dice, getDice, getIOU, thresholdPred

from Models.DeepSupAttUNet2D import DeepSupAttentionUnet
from Models.unet2D import UNet
from Utils.CTDS import CTDS
from Utils.augmentations import MegaMix, RandomAffine, RandomPixelTranslation
from Utils.seglosses import dice_loss, focal_tversky_loss
from Utils.vessel_utils import log_images
import h5py
class Brain(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        if self.hparams.modelID == 1:
            self.net = UNet(in_channels=1, n_classes=1, depth=4, wf=6, padding=True,
                            batch_norm=True, up_mode='upsample', is_leaky=True)
            self.forward_pass = UNet.forward_pass
        elif self.hparams.modelID == 2:
            self.net = UNetMSS()
        elif self.hparams.modelID == 3:
            self.net = AttUNet()
        elif self.hparams.modelID == 4:
            self.net = DeepSupAttentionUnet(in_channels=1, out_channels=1, is_batchnorm=True, is_leaky=True)
            self.forward_pass = DeepSupAttentionUnet.forward_pass
        elif self.hparams.modelID == 5:
            self.net = ProbUNet()
        else:
            sys.exit("Invalid Model ID")

        if self.hparams.lossID == 1:
            self.criterion = focal_tversky_loss
        elif self.hparams.lossID == 2:
            self.criterion = dice_loss
        else:
            sys.exit("Invalid Loss ID")

        self.get_dice = Dice()
        self.get_iou = IOU()

        if self.hparams.augmentID == 0:
            self.aug_transforms = []
        elif self.hparams.augmentID == 1:
            self.aug_transforms = [RandomPixelTranslation((32, 32))]
        elif self.hparams.augmentID == 2:
            self.aug_transforms = [MegaMix(0.75)]
        elif self.hparams.augmentID == 3:
            self.aug_transforms = [RandomAffine(degrees=45, translate=(0.125,0.125), flip=(True, True))]

        self.example_input_array = torch.empty(2, self.hparams.batch_size, 1, 512, 512).float()

    def configure_optimizers(self):
        optimiser = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
        )
        optim_dict = {
            'optimizer': optimiser,
            'monitor': 'val_loss',
        }
        return optim_dict

    def forward(self, batch):
        img, gt = batch
        return self.forward_pass(self.net, img, gt, self.criterion)
    
    def training_step(self, batch, batch_idx):
        loss, prediction = self(batch[0])
        self.log("running_loss", loss)
        if batch_idx % self.hparams.im_log_freq == 0:
            log_images(self.logger[-1].experiment, batch[0][0].cpu(), prediction, batch[0][1].cpu(), batch_idx, "train")
        return loss
    
    def training_epoch_end(self, outputs: List[Any]) -> None:
        avg_loss = torch.stack([x['loss'] for x in outputs]).median()
        self.log('training_loss', avg_loss)
    
    def validation_step(self, batch, batch_idx):
        loss, prediction = self(batch[0])
        _, dice_b4thres = self.get_dice(prediction.detach().cpu(), batch[0][1].cpu())
        iou_b4thres = self.get_iou(prediction.detach().cpu(), batch[0][1].cpu())
        prediction, gt = thresholdPred(prediction.detach().cpu().numpy()), batch[0][1].cpu().numpy()
        dice = getDice(prediction, gt)
        iou = getIOU(prediction, gt)
        if batch_idx % self.hparams.im_log_freq == 0:
            log_images(self.logger[-1].experiment, batch[0][0].cpu(), prediction, batch[0][1].cpu(), batch_idx, "val")
        return {'val_loss': loss.cpu(), 'val_dice_b4thres': dice_b4thres, 'val_iou_b4thres': iou_b4thres, 'val_dice': dice, 'val_iou': iou}

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).median()
        avg_dice_b4thres = torch.stack([x['val_dice_b4thres'] for x in outputs]).median()
        avgval_iou_b4thres = torch.stack([x['val_iou_b4thres'] for x in outputs]).median()
        avg_dice = np.median(np.stack([x['val_dice'] for x in outputs]))
        avg_iou = np.median(np.stack([x['val_iou'] for x in outputs]))
        self.log('val_loss', avg_loss)
        self.log('val_dice_b4thres', avg_dice_b4thres)
        self.log('val_iou_b4thres', avgval_iou_b4thres)
        self.log('val_dice', avg_dice)
        self.log('val_iou', avg_iou)

    def test_step(self, *args):
        loss, prediction = self(args[0][0])
        _, dice_b4thres = self.get_dice(prediction.detach().cpu(), args[0][0][1].cpu())
        iou_b4thres = self.get_iou(prediction.detach().cpu(), args[0][0][1].cpu())
        subIDs, sliceIDs = args[0][1]
        prediction, gt = thresholdPred(prediction.detach().cpu().numpy()), args[0][0][1].cpu().numpy()
        for i, (subID, sliceID) in enumerate(zip(subIDs, sliceIDs)):
            self.out_aggregators[subID][sliceID.item()] = prediction[i,0]
        dice = getDice(prediction, gt)
        iou = getIOU(prediction, gt)
        if args[1] % self.hparams.im_log_freq == 0:
            log_images(self.logger[-1].experiment, args[0][0][0].cpu(), prediction, args[0][0][1].cpu(), args[1], "test")
        return {'test_loss': loss.cpu(), 'test_dice_b4thres': dice_b4thres, 'test_iou_b4thres': iou_b4thres, 'test_dice': dice, 'test_iou': iou}

    def test_epoch_end(self, outputs: List[Any]) -> None:
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).median()
        avg_dice_b4thres = torch.stack([x['test_dice_b4thres'] for x in outputs]).median()
        avgval_iou_b4thres = torch.stack([x['test_iou_b4thres'] for x in outputs]).median()
        avg_dice = np.median(np.stack([x['test_dice'] for x in outputs]))
        avg_iou = np.median(np.stack([x['test_iou'] for x in outputs]))
        self.log('test_loss', avg_loss)
        self.log('test_dice_b4thres', avg_dice_b4thres)
        self.log('test_iou_b4thres', avgval_iou_b4thres)
        self.log('test_dice', avg_dice)
        self.log('test_iou', avg_iou)
        with h5py.File(self.hparams.res_path + "/output.hdf5", 'w') as h:
            for subID in self.out_aggregators.keys():
                subData = self.out_aggregators[subID]
                sliceIDs = list(subData.keys())
                results = np.zeros((max(sliceIDs)+1, subData[sliceIDs[0]].shape[0], subData[sliceIDs[0]].shape[1]))
                for sliceID in sliceIDs:
                    results[sliceID] = subData[sliceID]
                h.create_dataset(subID, data=results)   

    def train_dataloader(self):
        return DataLoader(CTDS(self.hparams.dataset_path, self.hparams.trainSubs, transforms=self.aug_transforms),
                          shuffle=True,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(CTDS(self.hparams.dataset_path, self.hparams.valSubs),
                          shuffle=False,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        self.out_aggregators = collections.defaultdict(dict)
        return DataLoader(CTDS(self.hparams.dataset_path, self.hparams.testSubs),
                          shuffle=False,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True, num_workers=self.hparams.num_workers)