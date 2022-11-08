#!/usr/bin/env python
"""

"""

import argparse
import random
import os
import numpy as np
import torch.utils.data
from os.path import join as pjoin
from torch.utils.tensorboard import SummaryWriter
from skimage.filters import threshold_otsu

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from Utils.logger import Logger
from Utils.model_manager import getModel
from Utils.vessel_utils import load_model, load_model_with_amp
import torch.nn.functional as F

import pytorch_lightning as pl

from brain import Brain
from overlays import create_diff_mask_binary
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.io as sio

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2022, Faculty of Computer Science, Otto von Guericke University Magdeburg, Germany"
__credits__ = ["Soumick Chatterjee"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Soumick Chatterjee"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Under testing"

pl.seed_everything(1701)
torch.set_num_threads(2)
# torch.autograd.set_detect_anomaly(True)

def save_img(arr, root, filename):
    sio.savemat(f"{root}/{filename}.mat", {"data":arr})
    plt.imshow(arr, cmap="gray")
    plt.tight_layout()
    plt.savefig(f"{root}/{filename}.png", format='png')

def thresholdPredSlc(predicted):    
    thresh = threshold_otsu(predicted)
    return (predicted > thresh).astype(np.float32)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--modelID",
                        type=int,
                        default=4,
                        help="1{U-Net}; \n"
                             "2{U-Net_Deepsup}; \n"
                             "3{Attention-U-Net}; \n"
                             "4{DeepSup-Attention-U-Net}; \n"
                             "5{Probabilistic-U-Net};")
    parser.add_argument("--lossID",
                        type=int,
                        default=1,
                        help="1{focal-tversky}; \n"
                             "2{Dice-Score}; ")
    parser.add_argument("--augmentID",
                        type=int,
                        default=2,
                        help="0{No-augmentation}; \n"
                             "1{Pixel-translation}; \n"
                             "2{Mega-Mix-0.75p}; \n"
                             "3{Random-Affine}; ")
    parser.add_argument("--run_name",
                        default="trial",
                        help="Name of the model")
    parser.add_argument("--dataset_path",
                        # default="/mnt/public/soumick/CTPerf/Data/CHAOS.hdf5",
                        # default="/mnt/public/soumick/CTPerf/Data/AnimalCTv1.hdf5",
                        # default="/mnt/public/soumick/CTPerf/Data/AnimalCTv2.hdf5",
			# default="/mnt/public/soumick/CTPerf/Data/AnimalCArmCTv1.hdf5",
                        default="/mnt/public/soumick/CTPerf/Data/AnimalTSTv1.hdf5",
                        help="Path to folder containing dataset.")
    parser.add_argument("--result_type",
                        default="",
                        help="Will be concatenated with the results. If its the main test (test set of the training), leave it blank.")
    parser.add_argument("--output_path",
                        default="/mnt/public/soumick/CTPerf/Output",
                        help="Folder path to store output ")
    parser.add_argument("--tb_path",
                        default="/mnt/public/soumick/CTPerf/TBLogs",
                        help="Folder path to store tensorboard output. Blank to avoid tboard.")

    parser.add_argument('--trainSubs',
                        # default="2,6,8,10,14,16,18,21,22,23,24,25,26,28,29",
                        default="animal_0312,animal_0808",
                        #default="2",
                        help="List of subIDs for training")
    parser.add_argument('--valSubs',
                        # default="1,5,19,27,30",
                        default="animal_1012,animal_1508",
                        #default="1",
                        help="List of subIDs for validation")
    parser.add_argument('--testSubs',
                        # default="1,5,19,27,30",
                        default="animal_1012",
                        # default="19",
                        help="List of subIDs for testing")

    parser.add_argument('--slice2gen',
                        default=98,
                        type=int,
                        # default="19",
                        help="sliceID for testing")
    parser.add_argument('--output4mss',
                        default="/mnt/public/soumick/CTPerf/Output/Consolidated/Plots/MSS/Fold0TST_TST",
                        # default="19",
                        help="where to store the multi-scale results")

    parser.add_argument('--train',
                        default=False,
                        help="To train the model")
    parser.add_argument('--test',
                        default=True,
                        help="To test the model")
    parser.add_argument('--predict',
                        default=False,
                        help="To predict a segmentation output of the model and to get a diff between label and output")
    parser.add_argument('--predictor_path',
                        default="",
                        help="Path to the input image to predict an output")
    parser.add_argument('--predictor_label_path',
                        default="",
                        help="Path to the label image to find the diff between label an output")

    parser.add_argument('--resume',
                        default=False,
                        action="store_true",
                        help="To use resume old training")
    parser.add_argument('--load_path',
                        default="/mnt/public/soumick/CTPerf/Output/0_Fold/Aug2_DeepSupAttenU_FTL_AnimalTSTv1_ptAnimalCArmCTv1_ptptAnimalCTv1run2_ptptptCHAOSnoAug",
                        # default="/mnt/public/soumick/CTPerf/Output/DeepSupAttenU_FTL_CHAOS",
                        help="Path to checkpoint of existing model to load, ex:/home/model/checkpoint/")
    parser.add_argument('--load_best',
                        default=True,
                        help="Specifiy whether to load the best checkpoiont or the last. Also to be used if Train and Test both are true.")
    parser.add_argument('--deform',
                        default=False,
                        action="store_true",
                        help="To use deformation for training")
    parser.add_argument('--clip_grads',
                        default=True,
                        action="store_true",
                        help="To use deformation for training (not implemented yet)")
    parser.add_argument('--amp',
                        default=True,
                        action="store_true",
                        help="To use half precision on model weights.")

    parser.add_argument("--batch_size",
                        type=int,
                        default=8,
                        help="Batch size for training")
    parser.add_argument("--effective_batch_size",
                        type=int,
                        default=64,
                        help="Batch size for training")
    parser.add_argument("--num_epochs",
                        type=int,
                        default=500,
                        help="Number of epochs for training")
    parser.add_argument("--learning_rate",
                        type=float,
                        default=0.001,
                        help="Learning rate")
    parser.add_argument("--num_workers",
                        type=int,
                        default=4,
                        help="Number of worker threads")



    parser.add_argument("--im_log_freq",
                        type=int,
                        default=100,
                        help="Number of epochs for training")
    parser.add_argument("--log_freq",
                        type=int,
                        default=100,
                        help="Number of epochs for training")

    parser.add_argument("--wnbactive",
                        default=False,
                        action="store_true",
                        help="Name of the WandB project")
    parser.add_argument("--wnbproject",
                        default="CTPrefSeg",
                        help="Name of the WandB project")
    parser.add_argument("--wnbentity",
                        default="mickchimp",
                        help="WandB entity")
    parser.add_argument("--wnbmodellog",
                        default='all',
                        help="WandB: While watching the model, what to save: gradients, parameters, all, None")
    parser.add_argument("--wnbmodelfreq",
                        type=int,
                        default=100,
                        help="WandB: The number of steps between logging gradients")

    args = parser.parse_args()

    if args.deform:
        args.run_name += "_Deform"

    # parser = Brain.add_model_specific_args(parser) #not yet implemented
    parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    hparams.res_path = pjoin(
        hparams.output_path, hparams.run_name, f"Results_{hparams.result_type}")
    os.makedirs(hparams.res_path, exist_ok=True)

    hparams.accumulate_gradbatch = hparams.effective_batch_size//hparams.batch_size

    if hparams.resume:
        path2chk = pjoin(hparams.output_path, hparams.run_name)
        if hparams.load_best:
            checkpoint_dir = pjoin(path2chk, "Checkpoints")
            available_checkpoints = {int(c.split("epoch=")[1].split("-")[0]): c for c in [x for x in os.listdir(checkpoint_dir) if "epoch" in x]}
            chkpoint = pjoin(checkpoint_dir, available_checkpoints[sorted(list(available_checkpoints.keys()))[-1]])
        else:
            chkpoint = pjoin(path2chk, "Checkpoints", "last.ckpt")
    elif bool(hparams.load_path):
        print("Loading existing checkpoint from pre-training")
        if hparams.load_best:
            checkpoint_dir = pjoin(hparams.load_path, "Checkpoints")
            available_checkpoints = {int(c.split("epoch=")[1].split("-")[0]): c for c in [x for x in os.listdir(checkpoint_dir) if "epoch" in x]}
            chkpoint = pjoin(checkpoint_dir, available_checkpoints[sorted(list(available_checkpoints.keys()))[-1]])
        else:
            chkpoint = pjoin(hparams.load_path, "Checkpoints", "last.ckpt")
    else:
        chkpoint = None

    model = Brain(**vars(hparams))
    if bool(chkpoint):
        model.load_state_dict(torch.load(chkpoint)['state_dict'])    

    for datum in tqdm(model.test_dataloader()):
        if hparams.slice2gen in datum[1][1]:
            img, gt = datum[0]
            prediction = model.net(img)

            inp_scale1_batch = F.avg_pool2d(img, 2)
            inp_scale2_batch = F.avg_pool2d(inp_scale1_batch, 2)
            inp_scale3_batch = F.avg_pool2d(inp_scale2_batch, 2)

            img = img.numpy()
            gt = gt.numpy()

            for i in range(img.shape[0]):
                raw_pred_scale3, gt_scale3 = prediction[0][i,0].detach().numpy(), gt[i,0,::8,::8]
                raw_pred_scale2, gt_scale2 = prediction[1][i,0].detach().numpy(), gt[i,0,::4,::4]
                raw_pred_scale1, gt_scale1 = prediction[2][i,0].detach().numpy(), gt[i,0,::2,::2]
                raw_pred_scale0, gt_scale0 = prediction[3][i,0].detach().numpy(), gt[i,0]

                pred_scale0 = thresholdPredSlc(raw_pred_scale0)
                pred_scale1 = thresholdPredSlc(raw_pred_scale1)
                pred_scale2 = thresholdPredSlc(raw_pred_scale2)
                pred_scale3 = thresholdPredSlc(raw_pred_scale3)

                overlay_scale0 = create_diff_mask_binary(pred_scale0, gt_scale0)
                overlay_scale1 = create_diff_mask_binary(pred_scale1, gt_scale1)
                overlay_scale2 = create_diff_mask_binary(pred_scale2, gt_scale2)
                overlay_scale3 = create_diff_mask_binary(pred_scale3, gt_scale3)

                inp_scale0 = img[i,0]
                inp_scale1 = inp_scale1_batch[i,0].numpy()
                inp_scale2 = inp_scale2_batch[i,0].numpy()
                inp_scale3 = inp_scale3_batch[i,0].numpy()

                animalID = datum[1][0][i]
                sliceID = datum[1][1][i].item()
                resID = f"{animalID}_slc{str(sliceID)}"

                outpath_slc = pjoin(hparams.output4mss, resID)
                os.makedirs(outpath_slc, exist_ok=True)

                save_img(raw_pred_scale3, outpath_slc, "predb4thres_scale3")
                save_img(raw_pred_scale2, outpath_slc, "predb4thres_scale2")
                save_img(raw_pred_scale1, outpath_slc, "predb4thres_scale1")
                save_img(raw_pred_scale0, outpath_slc, "predb4thres_scale0")

                save_img(pred_scale3, outpath_slc, "pred_scale3")
                save_img(pred_scale2, outpath_slc, "pred_scale2")
                save_img(pred_scale1, outpath_slc, "pred_scale1")
                save_img(pred_scale0, outpath_slc, "pred_scale0")

                save_img(gt_scale3, outpath_slc, "gt_scale3")
                save_img(gt_scale2, outpath_slc, "gt_scale2")
                save_img(gt_scale1, outpath_slc, "gt_scale1")
                save_img(gt_scale0, outpath_slc, "gt_scale0")

                save_img(overlay_scale0, outpath_slc, "overlay_scale0")
                save_img(overlay_scale1, outpath_slc, "overlay_scale1")
                save_img(overlay_scale2, outpath_slc, "overlay_scale2")
                save_img(overlay_scale3, outpath_slc, "overlay_scale3")

                save_img(inp_scale0, outpath_slc, "inp_scale0")
                save_img(inp_scale1, outpath_slc, "inp_scale1")
                save_img(inp_scale2, outpath_slc, "inp_scale2")
                save_img(inp_scale3, outpath_slc, "inp_scale3")