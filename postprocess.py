import argparse
from collections import defaultdict
import h5py
import numpy as np
from tqdm import tqdm
import skimage
import pandas as pd

from Evaluation.evaluate import getDice, getIOU, getAdditionalMetrics
from Utils.create_bbox import seg2bbx

def postprocess(prediction, area_type="actual"): #area_type can be "actual" "bbox_areas" "convex_areas"
    pred_bbox_cord, _, areas = seg2bbx(prediction)
    maxidx = areas[area_type].index(max(areas[area_type]))
    x_min, y_min, x_max, y_max = pred_bbox_cord[maxidx]

    marker = np.zeros(prediction.shape)
    marker[x_min:x_max, y_min:y_max] = 1
    prediction *= marker

    return prediction

parser = argparse.ArgumentParser()
parser.add_argument("--result_file",
                    default="/mnt/public/soumick/CTPerf/Output/Aug2_DeepSupAttenU_FTL_AnimalTSTv1_ptAnimalCArmCTv1_ptptAnimalCTv1run2_ptptptCHAOSnoAug/Results/output.hdf5",
                    #default="/mnt/public/soumick/CTPerf/Output/Fold2_Aug2_DeepSupAttenU_FTL_AnimalCArmCTv1_ptAnimalCTv1run2_ptptCHAOSnoAug/Results_carm_/output.hdf5",
                    help="Output file of testing.")
parser.add_argument("--gt_file",
                    # default="/mnt/public/soumick/CTPerf/Data/CHAOS.hdf5",
                    # default="/mnt/public/soumick/CTPerf/Data/AnimalCTv1.hdf5",
                    default="/mnt/public/soumick/CTPerf/Data/AnimalCTv2.hdf5",
		    # default="/mnt/public/soumick/CTPerf/Data/AnimalCArmCTv1.hdf5",
                    # default="/mnt/public/soumick/CTPerf/Data/AnimalTSTv1.hdf5",
                    help="Ground truth file.")
parser.add_argument("--perform_post",
                    action='store_true',
		    help="Exclude smaller regions.")
parser.add_argument("--save_post",
                    action='store_true',
		    help="Save stat.")
args = parser.parse_args()

result_file = args.result_file
gt_file = args.gt_file
perform_post = args.perform_post
print(perform_post)
save_post = args.save_post
print(save_post)
predicts = h5py.File(f"{result_file}", mode="r")
gts = h5py.File(f"{gt_file}", mode="r")

post_pred = defaultdict(list)
resIDs = []
dices = []
ious = []
newmetrics = []
dices_b4post = []
ious_b4post = []
newmetrics_b4post = []
subwise_dices = defaultdict(list)
subwise_ious = defaultdict(list)
for animal in predicts.keys():
    print(f"\nWorking on: {animal}....\n")
    for i in tqdm(range(predicts[animal].shape[0])):
        prediction = predicts[animal][i,...]
        gt = gts[animal][i,1,...]
        resIDs.append(f"{animal}_{i}")

        if perform_post:
            dices_b4post.append(getDice(prediction, gt))
            ious_b4post.append(getIOU(prediction, gt))
            newmetrics_b4post.append(getAdditionalMetrics(prediction, gt))
            try: #for blank predictions, postprocess will break 
                prediction = postprocess(prediction)
            except:
                pass
            if save_post:
                post_pred[animal].append(prediction)

        dice = getDice(prediction, gt)
        iou = getIOU(prediction, gt)
        metricvals = getAdditionalMetrics(prediction, gt)

        subwise_dices[animal].append(dice)
        subwise_ious[animal].append(iou)

        dices.append(dice)
        ious.append(iou)
        newmetrics.append(metricvals)

d = defaultdict(list)
for adict in newmetrics:
    for key, value in adict.items():
       d[key].append(value)
results = {
    "ResID": resIDs,
    "Dice": dices,
    "IoU": ious,
} | d

if perform_post:
    results['RawDice'] = dices_b4post
    results['RawIoU'] = ious_b4post
    d = defaultdict(list)
    for adict in newmetrics_b4post:
        for key, value in adict.items():
            d["Raw"+key].append(value)
    results |= d

df = pd.DataFrame.from_dict(results)
df.to_csv(result_file.replace('.hdf5', '_scores.csv'))

print("\n-------------\nAnimal-wise Results\n-------------\n")
for sub, value_ in subwise_dices.items():
    print(f"\nAnimal: {sub}------\n")
    print(
        f"Median Dice: {np.round(np.median(value_), 3)}±{np.round(np.var(subwise_dices[sub]), 3)}"
    )

    print(f"Mean Dice: {np.round(np.mean(subwise_dices[sub]),3)}±{np.round(np.std(subwise_dices[sub]),3)}")

    print(f"Median IoU: {np.round(np.median(subwise_ious[sub]),3)}±{np.round(np.var(subwise_ious[sub]),3)}")
    print(f"Mean IoU: {np.round(np.mean(subwise_ious[sub]),3)}±{np.round(np.std(subwise_ious[sub]),3)}")

print("\n-------------\nOverall Results\n-------------\n")
print(f"Median Dice: {np.round(np.median(dices),3)}±{np.round(np.var(dices),3)}")
print(f"Mean Dice: {np.round(np.mean(dices),3)}±{np.round(np.std(dices),3)}")

print(f"Median IoU: {np.round(np.median(ious),3)}±{np.round(np.var(ious),3)}")
print(f"Mean IoU: {np.round(np.mean(ious),3)}±{np.round(np.std(ious),3)}")

if perform_post and save_post:
    with h5py.File(f"{result_file.replace('.hdf5', '_postprocessed.hdf5')}", 'w') as h:
        for sub, value__ in post_pred.items():
            h.create_dataset(sub, data=value__)
