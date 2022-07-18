import numpy as np
import torch
import h5py

from Utils.vessel_utils import minmax
from torchvision.transforms import Compose

class CTDS(torch.utils.data.Dataset):
    def __init__(self, filepath, filterSubIDs, transforms=None):
        self.h5file = h5py.File(filepath, 'r')
        self.subIDs = []
        self.slcIDs = []

        filterSubIDs = filterSubIDs.split(",")
        for k in self.h5file.keys():
            if k in filterSubIDs:
                n_slc = self.h5file[k].shape[0]
                self.subIDs += [k]*n_slc
                self.slcIDs += list(np.arange(n_slc))

        if transforms is not None:
            self.transforms = Compose(transforms)
        else:
            self.transforms = None

    def __len__(self):
        return len(self.subIDs)

    def __getitem__(self, idx):
        subID = self.subIDs[idx]
        sliceID = self.slcIDs[idx]
        data_lbl = self.h5file[subID][sliceID]
        img, gt = data_lbl[0], data_lbl[1]
        
        img = torch.from_numpy(np.expand_dims(img, 0).astype(np.float32)).float()
        gt = torch.from_numpy(np.expand_dims(gt, 0).astype(np.float32)).float()

        if self.transforms is not None:
            img, gt = self.transforms((img, gt))        

        img = minmax(img)

        return (img, gt), (subID, sliceID)