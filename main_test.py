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

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

import pytorch_lightning as pl

from brain import Brain

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
                        # default="/mnt/public/soumick/CTPerf/Data/AnimalTSTv1.hdf5",
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
                        default="animal_1012,animal_1508",
                        # default="19",
                        help="List of subIDs for testing")

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
                        default=True,
                        action="store_true",
                        help="To use resume old training")
    parser.add_argument('--load_path',
                        default="",
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
                        default=True,
                        action="store_true",
                        help="Name of the WandB project")
    parser.add_argument("--wnbproject",
                        default="CTPrefSeg",
                        help="Name of the WandB project")
    parser.add_argument("--wnbgroup",
                        default=None,
                        help="Name of the WandB group")
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

    loggers = []
    if hparams.wnbactive:
        loggers.append(WandbLogger(name=hparams.run_name, id=hparams.run_name, project=hparams.wnbproject, group=hparams.wnbgroup,
                                    entity=hparams.wnbentity, config=hparams))
        if bool(hparams.wnbmodellog) and hparams.wnbmodellog!= "None":
            loggers[-1].watch(model, log=hparams.wnbmodellog, log_freq=hparams.wnbmodelfreq)
    else:
        os.environ["WANDB_MODE"] = "dryrun"
    if bool(hparams.tb_path):
        # TODO log_graph as True making it crash due to backward hooks
        os.makedirs(hparams.tb_path, exist_ok=True)
        loggers.append(TensorBoardLogger(hparams.tb_path,
                        name=hparams.run_name, log_graph=False))

    checkpoint_callback = ModelCheckpoint(
        dirpath=pjoin(hparams.output_path, hparams.run_name, "Checkpoints"),
        monitor='val_loss',
        save_last=True,
    )

    trainer = Trainer(
        logger=loggers,
        precision=16 if hparams.amp else 32,
        gpus=1,
        callbacks=[checkpoint_callback],
        checkpoint_callback=True,
        max_epochs=hparams.num_epochs,
        terminate_on_nan=True,
        accumulate_grad_batches=hparams.accumulate_gradbatch,
        resume_from_checkpoint=chkpoint if hparams.resume else None,
        check_val_every_n_epoch=1,
        log_every_n_steps=hparams.log_freq,
        flush_logs_every_n_steps=hparams.log_freq*2
    )

    if args.train:
        trainer.fit(model)
        torch.cuda.empty_cache()  # to avoid memory errors

    if args.test and args.load_best:
        trainer.test(model=model, test_dataloaders=model.test_dataloader())
        torch.cuda.empty_cache()  # to avoid memory errors

    if args.predict:
        print("Predict not implemented")
        # pipeline.predict(args.predictor_path, args.predictor_label_path, predict_logger=test_logger)
