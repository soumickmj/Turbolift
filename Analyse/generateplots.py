import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mannwhitneyu

#script 3 of 3 for analysis

sns.set(font_scale=1.1)
sns.set_style('whitegrid', {'font.family':'sans-serif', 'font.sans-serif':'Liberation Serif'})

#To be used for all plots, except for the plot for Turbolift with and without CHAOS
colours = ['#96cac1ff', '#f6f6d6ff', '#ea8e83ff', '#c1bed6ff', '#8aafc9ff']
customPalette = sns.color_palette(colours)

#Only for the plot for Turbolift with and without CHAOS
colours_wnwoCHAOS = ['#96cac1ff', '#52a093ff', '#f6f6d6ff', '#d8d84cff', '#c1bed6ff', '#7872a6ff']
customPalette_wnwoCHAOS = sns.color_palette(colours_wnwoCHAOS)

order = ['CT\n(Turbolift)', 'CT', 'CBCT\n(Turbolift)', 'CBCT', 'CBCT TST\n(Turbolift)', 'CBCT TST']


def gen_plot(df, y, path, chaos_colours=True):
    fig = plt.figure()
    order4plot = [o for o in order if o in df["Experiment Type"].unique()]
    # ax = sns.boxplot(x="Experiment Type", y=y, data=df, order=order4plot, showfliers=False, palette=sns.color_palette("Set3", 10)) #original colour palatte
    if chaos_colours:
        ax = sns.boxplot(x="Experiment Type", y=y, data=df, order=order4plot, showfliers=False, palette=customPalette_wnwoCHAOS)
    else:
        ax = sns.boxplot(x="Experiment Type", y=y, data=df, order=order4plot, showfliers=False, palette=customPalette)
    ax.set_xlabel("\nExperiment Type",fontsize=15)
    ax.set_ylabel(y,fontsize=15)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{path}_{y}.pdf", format='pdf')
    plt.figure().clear()
    plt.close()

def createNsaveFig(tag, savepath, TLct, TLcarm, TLtst, BASEcarm, BASEtst, BASEct=None):    
    BASEcarm['Experiment Type'] = "CBCT"
    BASEtst['Experiment Type'] = "CBCT TST"
    data = [TLct, TLcarm, TLtst, BASEcarm, BASEtst]
    if type(BASEct) is pd.DataFrame:
        BASEct['Experiment Type'] = "CT"
        data.insert(3, BASEct)
    df = pd.concat(data)
    os.makedirs(savepath, exist_ok=True)
    gen_plot(df, "Dice", f"{savepath}/{tag}", chaos_colours=("TLnoCHAOS" in tag) or ("revTL" in tag))
    
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

TL_ct_ct['Experiment Type'] = "CT\n(Turbolift)"
TL_carm_carm['Experiment Type'] = "CBCT\n(Turbolift)"
TL_tst_tst['Experiment Type'] = "CBCT TST\n(Turbolift)"

for i in range(len(tags)):
    tag = tags[i]
    carm_base = BASE_carm_carm[i]
    tst_base = BASE_tst_tst[i]

    if type(BASE_ct_ct[i]) is pd.DataFrame:
        ct_base = BASE_ct_ct[i]
        createNsaveFig(tag, f"{root}/Plots", TL_ct_ct, TL_carm_carm, TL_tst_tst, carm_base, tst_base, BASEct=ct_base)
    else:
        createNsaveFig(tag, f"{root}/Plots", TL_ct_ct, TL_carm_carm, TL_tst_tst, carm_base, tst_base)


print("Done!!")