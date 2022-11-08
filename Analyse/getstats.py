import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mannwhitneyu

#script 2 of 3 for analysis

def getStatIndividual(tl, base, modal, tag):
    return {
        "Modality": modal,
        "Tag": tag,
        "Dice": mannwhitneyu(tl.Dice, base.Dice).pvalue,
        "IoU": mannwhitneyu(tl.IoU, base.IoU).pvalue,
        "Precision": mannwhitneyu(tl.Precision, base.Precision).pvalue,
        "Sensitivity": mannwhitneyu(tl.Sensitivity, base.Sensitivity).pvalue,
        "Specificity": mannwhitneyu(tl.Specificity, base.Specificity).pvalue,
    }

def getStatIndividualRaw(res, modal, tag):
    return {
        "Modality": modal,
        "Tag": tag,
        "Dice": mannwhitneyu(res.Dice, res.RawDice).pvalue,
        "IoU": mannwhitneyu(res.IoU, res.RawIoU).pvalue,
        "Precision": mannwhitneyu(res.Precision, res.RawPrecision).pvalue,
        "Sensitivity": mannwhitneyu(
            res.Sensitivity, res.RawSensitivity
        ).pvalue,
        "Specificity": mannwhitneyu(
            res.Specificity, res.RawSpecificity
        ).pvalue,
    }
    

# root = "/mnt/public/soumick/CTPerf/Output/New1508/Consolidated" #6-Fold
root = "/mnt/public/soumick/CTPerf/Output/New1508/4FoldCV/Consolidated" #4-Fold

#Turbolift
TL_ct_ct = pd.read_csv(f"{root}/CT_CT.csv")
TL_carm_carm = pd.read_csv(f"{root}/CArm_CArm.csv")
TL_tst_tst = pd.read_csv(f"{root}/TST_TST.csv")

BASE_ct_ct = [None]
BASE_tst_tst = [pd.read_csv(f"{root}/TSTptCHAOS_TST.csv")]
BASE_carm_carm = [pd.read_csv(f"{root}/CArmptCHAOS_CArm.csv")]
tags = ["DirectWithCHAOS"]
#reverse Turbolift
flag = 0
try:
    BASE_ct_ct.append(None)
    BASE_carm_carm.append(pd.read_csv(f"{root}/rvrsCArm_CArm.csv"))
    flag += 1
    BASE_tst_tst.append(pd.read_csv(f"{root}/rvrsTST_TST.csv"))
    tags.append("TurboFlip")
except:
    if flag==1:
        BASE_ct_ct.pop()
        BASE_carm_carm.pop()

#Turbolift without CHAOS PT
flag = 0
try:
    BASE_ct_ct.append(pd.read_csv(f"{root}/noCHAOSCT_CT.csv"))
    flag += 1
    BASE_carm_carm.append(pd.read_csv(f"{root}/noCHAOSCArm_CArm.csv"))
    flag += 1
    BASE_tst_tst.append(pd.read_csv(f"{root}/noCHAOSTST_TST.csv"))
    tags.append("TLnoCHAOS")
except:
    if flag>=1:
        BASE_ct_ct.pop()
    if flag==2:
        BASE_carm_carm.pop()

#direct (without CHAOS pt)
flag = 0
try:
    BASE_ct_ct.append(None)
    BASE_carm_carm.append(pd.read_csv(f"{root}/dirCArm_CArm.csv")) 
    flag += 1
    BASE_tst_tst.append(pd.read_csv(f"{root}/dirTST_TST.csv"))
    tags.append("DirectWOCHAOS")
except:
    if flag==1:
        BASE_ct_ct.pop()
        BASE_carm_carm.pop()

#complete reverse Turbolift
flag = 0
try:
    BASE_ct_ct.append(pd.read_csv(f"{root}/rvrsTSTCArmCT_CT.csv"))
    flag += 1
    BASE_carm_carm.append(pd.read_csv(f"{root}/rvrsTSTCArm_CArm.csv"))
    flag += 1
    BASE_tst_tst.append(pd.read_csv(f"{root}/TSTptCHAOS_TST.csv"))
    tags.append("revTL")
except:
    if flag>=1:
        BASE_ct_ct.pop()
    if flag==2:
        BASE_carm_carm.pop()

stat_collect = []
stat4raw_collect = []
for i in range(len(tags)):
    tag = tags[i]
    carm_base = BASE_carm_carm[i]
    tst_base = BASE_tst_tst[i]

    if type(BASE_ct_ct[i]) is pd.DataFrame:
        ct_base = BASE_ct_ct[i]
        stat_collect.append(getStatIndividual(TL_ct_ct, ct_base, "CT", tag))
        stat4raw_collect.append(getStatIndividualRaw(ct_base, "CT", tag))

    stat_collect.extend(
        (
            getStatIndividual(TL_carm_carm, carm_base, "CArm", tag),
            getStatIndividual(TL_tst_tst, tst_base, "TST", tag),
        )
    )

    stat4raw_collect.extend(
        (
            getStatIndividualRaw(carm_base, "CArm", tag),
            getStatIndividualRaw(tst_base, "TST", tag),
        )
    )

stat4raw_collect.extend(
    (
        getStatIndividualRaw(TL_ct_ct, "CT", "TL"),
        getStatIndividualRaw(TL_carm_carm, "CArm", "TL"),
        getStatIndividualRaw(TL_tst_tst, "TST", "TL"),
    )
)

pd.DataFrame.from_dict(stat_collect).to_csv(f"{root}/stats.csv")
pd.DataFrame.from_dict(stat4raw_collect).to_csv(f"{root}/stats4raw.csv")

print("Done!!")