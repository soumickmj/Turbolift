import numpy as np
import torch
import h5py
from torch.utils.data import Dataset

from Utils.vessel_utils import minmax

class CTDS(Dataset):
    def __init__(self, filepath, subIDs):
        self.h5path = filepath
        self.subIDs = []
        self.slcIDs = []

        with h5py.File(filepath, 'r') as h5file:
            for k in h5file.keys():
                if k in subIDs:
                    n_slc = h5file[k].shape[0]
                    self.subIDs += [k]*n_slc
                    self.slcIDs += list(np.arange(n_slc))

    def __len__(self):
        return len(self.subIDs)

    def __call__(self, idx):
        subID = self.subIDs[idx]
        sliceID = self.slcIDs[idx]
        with h5py.File(self.h5path, 'r') as h5file:
            data_lbl = h5file[subID][sliceID]
        img, gt = data_lbl[0], data_lbl[1]

        img = minmax(img)

        return (img, gt), (subID, sliceID)