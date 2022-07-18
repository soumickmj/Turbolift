#!/usr/bin/env python

# from __future__ import print_function, division
'''

Purpose : 

'''

import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
import torch.utils.data
from skimage.filters import threshold_otsu

__author__ = "Kartik Prabhu, Mahantesh Pattadkal, and Soumick Chatterjee"
__copyright__ = "Copyright 2020, Faculty of Computer Science, Otto von Guericke University Magdeburg, Germany"
__credits__ = ["Kartik Prabhu", "Mahantesh Pattadkal", "Soumick Chatterjee"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Soumick Chatterjee"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Production"

def thresholdPred(predicted):
    thresholded = []
    for i in range(predicted.shape[0]):
        thresh = threshold_otsu(predicted[i])
        thresholded.append((predicted[i] > thresh).astype(np.float32))
    return np.array(thresholded)

def getDice(y_pred, y_true, smooth=1):
    y_pred_f = y_pred.flatten()
    y_true_f = y_true.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f + y_pred_f)
    dice_score = (2. * intersection + smooth) / (union + smooth)
    return dice_score

def getIOU(y_pred, y_true, smooth=1):
    y_pred_f = y_pred.flatten()
    y_true_f = y_true.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f + y_pred_f) - intersection
    score = (intersection + smooth) / (union + smooth)
    return score

def getAdditionalMetrics(y_pred, y_true):
    y_pred_f = y_pred.flatten()
    y_true_f = y_true.flatten()
    precision, _, _, _ = metrics.precision_recall_fscore_support(y_true_f, y_pred_f, average="binary")
    TN, FP, FN, TP = metrics.confusion_matrix(y_true_f, y_pred_f).ravel()
    FPR = FP/(FP+TN)
    sensitivity = TP/(TP+FN) #AKA hit rate, recall, or true positive rate
    specificity = TN/(FP+TN) #AKA true negative rate
    return {
        "Precision": precision,
        "FPR": FPR,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
    }

class Dice(nn.Module):
    """
    Class used to get dice_loss and dice_score
    """

    def __init__(self, smooth=1):
        super(Dice, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred_f = torch.flatten(y_pred)
        y_true_f = torch.flatten(y_true)
        intersection = torch.sum(y_true_f * y_pred_f)
        union = torch.sum(y_true_f + y_pred_f)
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score
        return dice_loss, dice_score


class IOU(nn.Module):
    def __init__(self, smooth=1):
        super(IOU, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred_f = torch.flatten(y_pred)
        y_true_f = torch.flatten(y_true)
        intersection = torch.sum(y_true_f * y_pred_f)
        union = torch.sum(y_true_f + y_pred_f) - intersection
        score = (intersection + self.smooth) / (union + self.smooth)
        return score


class FocalTverskyLoss(nn.Module):
    def __init__(self, smooth=1, gamma=0.75, alpha=0.7):
        super(FocalTverskyLoss, self).__init__()
        self.smooth = smooth
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        y_true_pos = torch.flatten(y_true)
        y_pred_pos = torch.flatten(y_pred)
        true_pos = torch.sum(y_true_pos * y_pred_pos)
        false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)
        pt_1 = (true_pos + self.smooth) / (
                true_pos + self.alpha * false_neg + (1 - self.alpha) * false_pos + self.smooth)
        # return pow((1 - pt_1), self.gamma)
        return pow(abs(1 - pt_1), self.gamma)


class FocalTverskyLoss_detailed(nn.Module):
    def __init__(self, smooth=1, gamma=0.75, alpha=0.7):
        super(FocalTverskyLoss_detailed, self).__init__()
        self.smooth = smooth
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logger, y_pred, y_true):
        y_true_pos = torch.flatten(y_true)
        y_pred_pos = torch.flatten(y_pred)
        true_pos = torch.sum(y_true_pos * y_pred_pos)
        false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)
        logger.info("True Positive:" + str(true_pos) + " False_Negative:" + str(false_neg) + " False_Positive:" + str(false_pos))
        pt_1 = (true_pos + self.smooth) / (
                true_pos + self.alpha * false_neg + (1 - self.alpha) * false_pos + self.smooth)
        return pow((1 - pt_1), self.gamma)


def getMetric(logger, y_pred, y_true):
    y_true_pos = torch.flatten(y_true)
    y_pred_pos = torch.flatten(y_pred)
    true_pos = torch.sum(y_true_pos * y_pred_pos)
    false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)
    intersection = true_pos
    union = torch.sum(y_true_pos + y_pred_pos)
    return true_pos, false_neg, false_pos, intersection, union


def getLosses(logger, true_pos, false_neg, false_pos, intersection, union):
    smooth = 1
    gamma = 0.75
    alpha = 0.7

    dice_score = (2. * intersection + smooth) / (union + smooth)
    dice_loss = 1 - dice_score
    iou = (intersection + smooth) / (union - intersection + smooth)
    pt_1 = (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)
    floss = pow((1 - pt_1), gamma)

    logger.info("True Positive:" + str(true_pos) + " False_Negative:" + str(false_neg) + " False_Positive:" + str(false_pos))
    logger.info("Floss:" + str(floss) + " diceloss:" + str(dice_loss) + " iou:" + str(iou))
    return floss, dice_loss, iou
