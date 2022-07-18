from collections import defaultdict
from statistics import variance
import pandas as pd

#script 1 of 3 for analysis (for 6-fold CV)

consolidated_output = "/mnt/public/soumick/CTPerf/Output/New1508/Consolidated"

#Fold, training, testing

fold0 = {
    "CT": {
        #"CHAOS": "/mnt/public/soumick/CTPerf/Output/0_Fold/Aug2_DeepSupAttenU_FTL_AnimalCTv1_ptCHAOSnoAug_run2/Results_chaos/",
        "CT": "/mnt/public/soumick/CTPerf/Output/New1508/Fold0/Aug2_DeepSupAttenU_FTL_AnimalCTv1_ptCHAOSnoAug_run2/Results_CTv2/",
        # "CArm": "/mnt/public/soumick/CTPerf/Output/0_Fold/Aug2_DeepSupAttenU_FTL_AnimalCTv1_ptCHAOSnoAug_run2/Results_carm/",
        # "TST": "/mnt/public/soumick/CTPerf/Output/0_Fold/Aug2_DeepSupAttenU_FTL_AnimalCTv1_ptCHAOSnoAug_run2/Results_tst/",
    },    
    "CArm": {
        #"CHAOS": "/mnt/public/soumick/CTPerf/Output/0_Fold/Aug2_DeepSupAttenU_FTL_AnimalCArmCTv1_ptAnimalCTv1run2_ptptCHAOSnoAug/Results_chaos/",
        # "CT": "/mnt/public/soumick/CTPerf/Output/0_Fold/Aug2_DeepSupAttenU_FTL_AnimalCArmCTv1_ptAnimalCTv1run2_ptptCHAOSnoAug/Results_ct/",
        "CArm": "/mnt/public/soumick/CTPerf/Output/0_Fold/Aug2_DeepSupAttenU_FTL_AnimalCArmCTv1_ptAnimalCTv1run2_ptptCHAOSnoAug/Results_/",
        # "TST": "/mnt/public/soumick/CTPerf/Output/0_Fold/Aug2_DeepSupAttenU_FTL_AnimalCArmCTv1_ptAnimalCTv1run2_ptptCHAOSnoAug/Results_tst/",
    },    
    "TST": {
        #"CHAOS": "/mnt/public/soumick/CTPerf/Output/0_Fold/Aug2_DeepSupAttenU_FTL_AnimalTSTv1_ptAnimalCArmCTv1_ptptAnimalCTv1run2_ptptptCHAOSnoAug/Results_chaos/",
        # "CT": "/mnt/public/soumick/CTPerf/Output/0_Fold/Aug2_DeepSupAttenU_FTL_AnimalTSTv1_ptAnimalCArmCTv1_ptptAnimalCTv1run2_ptptptCHAOSnoAug/Results_ct/",
        # "CArm": "/mnt/public/soumick/CTPerf/Output/0_Fold/Aug2_DeepSupAttenU_FTL_AnimalTSTv1_ptAnimalCArmCTv1_ptptAnimalCTv1run2_ptptptCHAOSnoAug/Results_carm/",
        "TST": "/mnt/public/soumick/CTPerf/Output/0_Fold/Aug2_DeepSupAttenU_FTL_AnimalTSTv1_ptAnimalCArmCTv1_ptptAnimalCTv1run2_ptptptCHAOSnoAug/Results_/",
    },
    "dirTST": {
        "TST": "/mnt/public/soumick/CTPerf/Output/0_Fold/Fold0_Aug2_DeepSupAttenU_FTL_AnimalTSTv1/Results_/"
    },
    "TSTptCHAOS": {
        "TST": "/mnt/public/soumick/CTPerf/Output/0_Fold/Fold0_Aug2_DeepSupAttenU_FTL_AnimalTSTv1_ptCHAOSnoAug/Results_/"
    },
    "dirCArm": {
        "CArm": "/mnt/public/soumick/CTPerf/Output/0_Fold/Fold0_Aug2_DeepSupAttenU_FTL_AnimalCArmCTv1/Results_/"
    },
    "CArmptCHAOS": {
        "CArm": "/mnt/public/soumick/CTPerf/Output/0_Fold/Fold0_Aug2_DeepSupAttenU_FTL_AnimalCArmCTv1_ptCHAOSnoAug/Results_/"
    },
    "CTnoCHAOS": {
        "CT": "/home/soumick/96/mnt/mnt/md1337/hana/nas/mnt/fuenf12er/hana/data/6FOLDS/Fold0_Aug2_DeepSupAttenU_FTL_AnimalCTv1run2/Results_CTv2/"
    },
    "CarmptCTnoCHAOS": {
        "CArm": "/home/soumick/96/mnt/mnt/md1337/hana/nas/mnt/fuenf12er/hana/data/6FOLDS/Fold0_Aug2_DeepSupAttenU_FTL_AnimalCArmCT_ptAnimalCTv1run2/Results_CBCT/"
    }  
}

fold1 = {
    "CT": {
        #"CHAOS": "/mnt/public/soumick/CTPerf/Output/1_Fold/_Fold1_Aug2_DeepSupAttenU_FTL_AnimalCTv1run2_ptCHAOSnoAug/Results_chaos_/",
        "CT": "/mnt/public/soumick/CTPerf/Output/New1508/Fold1/_Fold1_Aug2_DeepSupAttenU_FTL_AnimalCTv1run2_ptCHAOSnoAug/Results_CTv2/",
        # "CArm": "/mnt/public/soumick/CTPerf/Output/1_Fold/_Fold1_Aug2_DeepSupAttenU_FTL_AnimalCTv1run2_ptCHAOSnoAug/Results_carm_/",
        # "TST": "/mnt/public/soumick/CTPerf/Output/1_Fold/_Fold1_Aug2_DeepSupAttenU_FTL_AnimalCTv1run2_ptCHAOSnoAug/Results_tst_/",
    },    
    "CArm": {
        #"CHAOS": "/mnt/public/soumick/CTPerf/Output/1_Fold/_Fold1_Aug2_DeepSupAttenU_FTL_AnimalCArmCTv1_ptAnimalCTv1run2_ptptCHAOSnoAug/Results_chaos_/",
        # "CT": "/mnt/public/soumick/CTPerf/Output/1_Fold/_Fold1_Aug2_DeepSupAttenU_FTL_AnimalCArmCTv1_ptAnimalCTv1run2_ptptCHAOSnoAug/Results_ct_/",
        "CArm": "/mnt/public/soumick/CTPerf/Output/1_Fold/_Fold1_Aug2_DeepSupAttenU_FTL_AnimalCArmCTv1_ptAnimalCTv1run2_ptptCHAOSnoAug/Results_/",
        # "TST": "/mnt/public/soumick/CTPerf/Output/1_Fold/_Fold1_Aug2_DeepSupAttenU_FTL_AnimalCArmCTv1_ptAnimalCTv1run2_ptptCHAOSnoAug/Results_tst_/",
    },    
    "TST": {
        #"CHAOS": "/mnt/public/soumick/CTPerf/Output/1_Fold/Set1_Fold1_Aug2_DeepSupAttenU_FTL_AnimalTSTv1_ptAnimalCArmCTv1_ptptAnimalCTv1run2_ptptptCHAOSnoAug/Results_chaos_/",
        # "CT": "/mnt/public/soumick/CTPerf/Output/1_Fold/Set1_Fold1_Aug2_DeepSupAttenU_FTL_AnimalTSTv1_ptAnimalCArmCTv1_ptptAnimalCTv1run2_ptptptCHAOSnoAug/Results_ct_/",
        # "CArm": "/mnt/public/soumick/CTPerf/Output/1_Fold/Set1_Fold1_Aug2_DeepSupAttenU_FTL_AnimalTSTv1_ptAnimalCArmCTv1_ptptAnimalCTv1run2_ptptptCHAOSnoAug/Results_carm/",
        "TST": "/mnt/public/soumick/CTPerf/Output/1_Fold/Fold1_Aug2_DeepSupAttenU_FTL_AnimalTSTv1_ptAnimalCArmCTv1_ptptAnimalCTv1run2_ptptptCHAOSnoAug/Results_/",
    },
    "dirTST": {
        "TST": "/mnt/public/soumick/CTPerf/Output/1_Fold/Fold1_Aug2_DeepSupAttenU_FTL_AnimalTSTv1/Results_/"
    },
    "TSTptCHAOS": {
        "TST": "/mnt/public/soumick/CTPerf/Output/1_Fold/Fold1_Aug2_DeepSupAttenU_FTL_AnimalTSTv1_ptCHAOSnoAug/Results_/"
    },
    "dirCArm": {
        "CArm": "/mnt/public/soumick/CTPerf/Output/1_Fold/Fold1_Aug2_DeepSupAttenU_FTL_AnimalCArmCTv1/Results_/"
    },
    "CArmptCHAOS": {
        "CArm": "/mnt/public/soumick/CTPerf/Output/1_Fold/Fold1_Aug2_DeepSupAttenU_FTL_AnimalCArmCTv1_ptCHAOSnoAug/Results_/"
    },
    "CTnoCHAOS": {
        "CT": "/home/soumick/96/mnt/mnt/md1337/hana/nas/mnt/fuenf12er/hana/data/6FOLDS/Fold1_Aug2_DeepSupAttenU_FTL_AnimalCTv1run2/Results_CTv2/"
    },
    "CarmptCTnoCHAOS": {
        "CArm": "/home/soumick/96/mnt/mnt/md1337/hana/nas/mnt/fuenf12er/hana/data/6FOLDS/Fold1_Aug2_DeepSupAttenU_FTL_AnimalCArmCT_ptAnimalCTv1run2/Results_CBCT/"
    } 
}

fold2 = {
    "CT": {
        #"CHAOS": "/mnt/public/soumick/CTPerf/Output/2_Fold/Fold2_Aug2_DeepSupAttenU_FTL_AnimalCTv1run2_ptCHAOSnoAug/Results_chaos_/",
        "CT": "/mnt/public/soumick/CTPerf/Output/New1508/Fold2/Fold2_Aug2_DeepSupAttenU_FTL_AnimalCTv2run2_ptCHAOSnoAug/Results_CTv2/",
        # "CArm": "/mnt/public/soumick/CTPerf/Output/2_Fold/Fold2_Aug2_DeepSupAttenU_FTL_AnimalCTv1run2_ptCHAOSnoAug/Results_carm_/",
        # "TST": "/mnt/public/soumick/CTPerf/Output/2_Fold/Fold2_Aug2_DeepSupAttenU_FTL_AnimalCTv1run2_ptCHAOSnoAug/Results_tst_/",
    },    
    "CArm": {
        #"CHAOS": "/mnt/public/soumick/CTPerf/Output/2_Fold/Fold2_Aug2_DeepSupAttenU_FTL_AnimalCArmCTv1_ptAnimalCTv1run2_ptptCHAOSnoAug/Results_chaos_/",
        # "CT": "/mnt/public/soumick/CTPerf/Output/2_Fold/Fold2_Aug2_DeepSupAttenU_FTL_AnimalCArmCTv1_ptAnimalCTv1run2_ptptCHAOSnoAug/Results_ct_/",
        "CArm": "/mnt/public/soumick/CTPerf/Output/New1508/Fold2/Set1_Fold2_Aug2_DeepSupAttenU_FTL_AnimalCArmCTv1_ptAnimalCTv2run2_ptptCHAOSnoAug/Results_/",
        # "TST": "/mnt/public/soumick/CTPerf/Output/2_Fold/Fold2_Aug2_DeepSupAttenU_FTL_AnimalCArmCTv1_ptAnimalCTv1run2_ptptCHAOSnoAug/Results_tst_/",
    },    
    "TST": {
        #"CHAOS": "/mnt/public/soumick/CTPerf/Output/2_Fold/Fold2_Aug2_DeepSupAttenU_FTL_AnimalTSTv1_ptAnimalCArmCTv1_ptptAnimalCTv1run2_ptptptCHAOSnoAug/Results_chaos_/",
        # "CT": "/mnt/public/soumick/CTPerf/Output/2_Fold/Fold2_Aug2_DeepSupAttenU_FTL_AnimalTSTv1_ptAnimalCArmCTv1_ptptAnimalCTv1run2_ptptptCHAOSnoAug/Results_ct_/",
        # "CArm": "/mnt/public/soumick/CTPerf/Output/2_Fold/Fold2_Aug2_DeepSupAttenU_FTL_AnimalTSTv1_ptAnimalCArmCTv1_ptptAnimalCTv1run2_ptptptCHAOSnoAug/Results_carm_/",
        "TST": "/mnt/public/soumick/CTPerf/Output/New1508/Fold2/Set1_Fold2_Aug2_DeepSupAttenU_FTL_AnimalTSTv1_ptAnimalCArmCTv1_ptptAnimalCTv2run2_ptptptCHAOSnoAug/Results_/",
    },
    "dirTST": {
        "TST": "/home/soumick/96/mnt/mnt/md1337/hana/nas/mnt/fuenf12er/hana/data/Fold2_Aug2_DeepSupAttenU_FTL_AnimalTSTv1/Results_/"
    },
    "TSTptCHAOS": {
        "TST": "/home/soumick/96/mnt/mnt/md1337/hana/nas/mnt/fuenf12er/hana/data/Fold2_Aug2_DeepSupAttenU_FTL_AnimalTSTv1_ptCHAOSnoAug/Results_/"
    },
    "dirCArm": {
        "CArm": "/home/soumick/96/mnt/mnt/md1337/hana/nas/mnt/fuenf12er/hana/data/Fold2_Aug2_DeepSupAttenU_FTL_AnimalCArmCTv1/Results_/"
    },
    "CArmptCHAOS": {
        "CArm": "/home/soumick/96/mnt/mnt/md1337/hana/nas/mnt/fuenf12er/hana/data//Fold2_Aug2_DeepSupAttenU_FTL_AnimalCArmCTv1_ptCHAOSnoAug/Results_/"
    },
    "CTnoCHAOS": {
        "CT": "/home/soumick/96/mnt/mnt/md1337/hana/nas/mnt/fuenf12er/hana/data/6FOLDS/Fold2_Aug2_DeepSupAttenU_FTL_AnimalCTv2run2/Results_CTv2/"
    },
    "CarmptCTnoCHAOS": {
        "CArm": "/home/soumick/96/mnt/mnt/md1337/hana/nas/mnt/fuenf12er/hana/data/6FOLDS/Fold2_Aug2_DeepSupAttenU_FTL_AnimalCArmCT_ptAnimalCTv2run2/Results_CBCT/"
    } 
}

fold3 = {
    "CT": {
        #"CHAOS": "/mnt/public/soumick/CTPerf/Output/3_Fold/Fold3_Aug2_DeepSupAttenU_FTL_AnimalCTv1run2_ptCHAOSnoAug/Results_chaos_/",
        "CT": "/mnt/public/soumick/CTPerf/Output/New1508/Fold3/Fold3_Aug2_DeepSupAttenU_FTL_AnimalCTv2run2_ptCHAOSnoAug/Results_CTv2/",
        # "CArm": "/mnt/public/soumick/CTPerf/Output/3_Fold/Fold3_Aug2_DeepSupAttenU_FTL_AnimalCTv1run2_ptCHAOSnoAug/Results_carm_/",
        # "TST": "/mnt/public/soumick/CTPerf/Output/3_Fold/Fold3_Aug2_DeepSupAttenU_FTL_AnimalCTv1run2_ptCHAOSnoAug/Results_tst_/",
    },    
    "CArm": {
        #"CHAOS": "/mnt/public/soumick/CTPerf/Output/3_Fold/Fold3_Aug2_DeepSupAttenU_FTL_AnimalCArmCTv1_ptAnimalCTv1run2_ptptCHAOSnoAug/Results_chaos_/",
        # "CT": "/mnt/public/soumick/CTPerf/Output/3_Fold/Fold3_Aug2_DeepSupAttenU_FTL_AnimalCArmCTv1_ptAnimalCTv1run2_ptptCHAOSnoAug/Results_ct_/",
        "CArm": "/mnt/public/soumick/CTPerf/Output/New1508/Fold3/Set1_Fold3_Aug2_DeepSupAttenU_FTL_AnimalCArmCTv1_ptAnimalCTv2run2_ptptCHAOSnoAug/Results_/",
        # "TST": "/mnt/public/soumick/CTPerf/Output/3_Fold/Fold3_Aug2_DeepSupAttenU_FTL_AnimalCArmCTv1_ptAnimalCTv1run2_ptptCHAOSnoAug/Results_tst_/",
    },    
    "TST": {
        #"CHAOS": "/mnt/public/soumick/CTPerf/Output/3_Fold/Set1_Fold3_Aug2_DeepSupAttenU_FTL_AnimalTSTv1_ptAnimalCArmCTv1_ptptAnimalCTv1run2_ptptptCHAOSnoAug/Results_chaos_/",
        # "CT": "/mnt/public/soumick/CTPerf/Output/3_Fold/Set1_Fold3_Aug2_DeepSupAttenU_FTL_AnimalTSTv1_ptAnimalCArmCTv1_ptptAnimalCTv1run2_ptptptCHAOSnoAug/Results_ct_/",
        # "CArm": "/mnt/public/soumick/CTPerf/Output/3_Fold/Set1_Fold3_Aug2_DeepSupAttenU_FTL_AnimalTSTv1_ptAnimalCArmCTv1_ptptAnimalCTv1run2_ptptptCHAOSnoAug/Results_carm/",
        "TST": "/mnt/public/soumick/CTPerf/Output/New1508/Fold3/Set1_Fold3_Aug2_DeepSupAttenU_FTL_AnimalTSTv1_ptAnimalCArmCTv1_ptptAnimalCTv2run2_ptptptCHAOSnoAug/Results_/",
    },
    "dirTST": {
        "TST": "/home/soumick/96/mnt/mnt/md1337/hana/nas/mnt/fuenf12er/hana/data/Fold3_Aug2_DeepSupAttenU_FTL_AnimalTSTv1/Results_/"
    },
    "TSTptCHAOS": {
        "TST": "/home/soumick/96/mnt/mnt/md1337/hana/nas/mnt/fuenf12er/hana/data//Fold3_Aug2_DeepSupAttenU_FTL_AnimalTSTv1_ptCHAOSnoAug/Results_/"
    },
    "dirCArm": {
        "CArm": "/home/soumick/96/mnt/mnt/md1337/hana/nas/mnt/fuenf12er/hana/data/Fold3_Aug2_DeepSupAttenU_FTL_AnimalCArmCTv1/Results_/"
    },
    "CArmptCHAOS": {
        "CArm": "/home/soumick/96/mnt/mnt/md1337/hana/nas/mnt/fuenf12er/hana/data/Fold3_Aug2_DeepSupAttenU_FTL_AnimalCArmCTv1_ptCHAOSnoAug/Results_/"
    },
    "CTnoCHAOS": {
        "CT": "/home/soumick/96/mnt/mnt/md1337/hana/nas/mnt/fuenf12er/hana/data/6FOLDS/Fold3_Aug2_DeepSupAttenU_FTL_AnimalCTv2run2/Results_CTv2/"
    },
    "CarmptCTnoCHAOS": {
        "CArm": "/home/soumick/96/mnt/mnt/md1337/hana/nas/mnt/fuenf12er/hana/data/6FOLDS/Fold3_Aug2_DeepSupAttenU_FTL_AnimalCArmCT_ptAnimalCTv1run2/Results_CBCT/"
    } 
}

fold4 = {
    "CT": {
        #"CHAOS": "/mnt/public/soumick/CTPerf/Output/4_Fold/Fold4_Aug2_DeepSupAttenU_FTL_AnimalCTv1run2_ptCHAOSnoAug/Results_chaos_/",
        "CT": "/mnt/public/soumick/CTPerf/Output/New1508/Fold4/Fold4_Aug2_DeepSupAttenU_FTL_AnimalCTv2run2_ptCHAOSnoAug/Results_CTv2/",
        # "CArm": "/mnt/public/soumick/CTPerf/Output/4_Fold/Fold4_Aug2_DeepSupAttenU_FTL_AnimalCTv1run2_ptCHAOSnoAug/Results_carm_/",
        # "TST": "/mnt/public/soumick/CTPerf/Output/4_Fold/Fold4_Aug2_DeepSupAttenU_FTL_AnimalCTv1run2_ptCHAOSnoAug/Results_tst_/",
    },    
    "CArm": {
        #"CHAOS": "/mnt/public/soumick/CTPerf/Output/4_Fold/Fold4_Aug2_DeepSupAttenU_FTL_AnimalCArmCTv1_ptAnimalCTv1run2_ptptCHAOSnoAug/Results_chaos_/",
        # "CT": "/mnt/public/soumick/CTPerf/Output/4_Fold/Fold4_Aug2_DeepSupAttenU_FTL_AnimalCArmCTv1_ptAnimalCTv1run2_ptptCHAOSnoAug/Results_ct_/",
        "CArm": "/mnt/public/soumick/CTPerf/Output/New1508/Fold4/Set1_Fold4_Aug2_DeepSupAttenU_FTL_AnimalCArmCTv1_ptAnimalCTv2run2_ptptCHAOSnoAug/Results_/",
        # "TST": "/mnt/public/soumick/CTPerf/Output/4_Fold/Fold4_Aug2_DeepSupAttenU_FTL_AnimalCArmCTv1_ptAnimalCTv1run2_ptptCHAOSnoAug/Results_tst_/",
    },    
    "TST": {
        #"CHAOS": "/mnt/public/soumick/CTPerf/Output/4_Fold/Fold4_Aug2_DeepSupAttenU_FTL_AnimalTSTv1_ptAnimalCArmCTv1_ptptAnimalCTv1run2_ptptptCHAOSnoAug/Results_chaos_/",
        # "CT": "/mnt/public/soumick/CTPerf/Output/4_Fold/Fold4_Aug2_DeepSupAttenU_FTL_AnimalTSTv1_ptAnimalCArmCTv1_ptptAnimalCTv1run2_ptptptCHAOSnoAug/Results_ct_/",
        # "CArm": "/mnt/public/soumick/CTPerf/Output/4_Fold/Fold4_Aug2_DeepSupAttenU_FTL_AnimalTSTv1_ptAnimalCArmCTv1_ptptAnimalCTv1run2_ptptptCHAOSnoAug/Results_carm_/",
        "TST": "/mnt/public/soumick/CTPerf/Output/New1508/Fold4/Set1_Fold4_Aug2_DeepSupAttenU_FTL_AnimalTSTv1_ptAnimalCArmCTv1_ptptAnimalCTv2run2_ptptptCHAOSnoAug/Results_/",
    },
    "dirTST": {
        "TST": "/home/soumick/96/mnt/mnt/md1337/hana/nas/mnt/fuenf12er/hana/data/Fold4_Aug2_DeepSupAttenU_FTL_AnimalTSTv1/Results_/"
    },
    "TSTptCHAOS": {
        "TST": "/home/soumick/96/mnt/mnt/md1337/hana/nas/mnt/fuenf12er/hana/data/Fold4_Aug2_DeepSupAttenU_FTL_AnimalTSTv1_ptCHAOSnoAug/Results_/"
    },
    "dirCArm": {
        "CArm": "/home/soumick/96/mnt/mnt/md1337/hana/nas/mnt/fuenf12er/hana/data/Fold4_Aug2_DeepSupAttenU_FTL_AnimalCArmCTv1/Results_/"
    },
    "CArmptCHAOS": {
        "CArm": "/home/soumick/96/mnt/mnt/md1337/hana/nas/mnt/fuenf12er/hana/data/Fold4_Aug2_DeepSupAttenU_FTL_AnimalCArmCTv1_ptCHAOSnoAug/Results_/"
    },
    "CTnoCHAOS": {
        "CT": "/home/soumick/96/mnt/mnt/md1337/hana/nas/mnt/fuenf12er/hana/data/6FOLDS/Fold4_Aug2_DeepSupAttenU_FTL_AnimalCTv2run2/Results_CTv2/"
    },
    "CarmptCTnoCHAOS": {
        "CArm": "/home/soumick/96/mnt/mnt/md1337/hana/nas/mnt/fuenf12er/hana/data/6FOLDS/Fold4_Aug2_DeepSupAttenU_FTL_AnimalCArmCT_ptAnimalCTv2run2/Results_CBCT/"
    } 
}

fold5 = {
    "CT": {
        #"CHAOS": "/mnt/public/soumick/CTPerf/Output/5_Fold/Fold5_Aug2_DeepSupAttenU_FTL_AnimalCTv1run2_ptCHAOSnoAug/Results_chaos_/",
        "CT": "/mnt/public/soumick/CTPerf/Output/New1508/Fold5/Fold5_Aug2_DeepSupAttenU_FTL_AnimalCTv1run2_ptCHAOSnoAug/Results_CTv2/",
        # "CArm": "/mnt/public/soumick/CTPerf/Output/5_Fold/Fold5_Aug2_DeepSupAttenU_FTL_AnimalCTv1run2_ptCHAOSnoAug/Results_carm_/",
        # "TST": "/mnt/public/soumick/CTPerf/Output/5_Fold/Fold5_Aug2_DeepSupAttenU_FTL_AnimalCTv1run2_ptCHAOSnoAug/Results_tst_/",
    },    
    "CArm": {
        #"CHAOS": "/mnt/public/soumick/CTPerf/Output/5_Fold/Fold5_Aug2_DeepSupAttenU_FTL_AnimalCArmCTv1_ptAnimalCTv1run2_ptptCHAOSnoAug/Results_chaos_/",
        # "CT": "/mnt/public/soumick/CTPerf/Output/5_Fold/Fold5_Aug2_DeepSupAttenU_FTL_AnimalCArmCTv1_ptAnimalCTv1run2_ptptCHAOSnoAug/Results_ct_/",
        "CArm": "/mnt/public/soumick/CTPerf/Output/5_Fold/Fold5_Aug2_DeepSupAttenU_FTL_AnimalCArmCTv1_ptAnimalCTv1run2_ptptCHAOSnoAug/Results_/",
        # "TST": "/mnt/public/soumick/CTPerf/Output/5_Fold/Fold5_Aug2_DeepSupAttenU_FTL_AnimalCArmCTv1_ptAnimalCTv1run2_ptptCHAOSnoAug/Results_tst_/",
    },    
    "TST": {
        #"CHAOS": "/mnt/public/soumick/CTPerf/Output/5_Fold/Fold5_Aug2_DeepSupAttenU_FTL_AnimalTSTv1_ptAnimalCArmCTv1_ptptAnimalCTv1run2_ptptptCHAOSnoAug/Results_chaos_/",
        # "CT": "/mnt/public/soumick/CTPerf/Output/5_Fold/Fold5_Aug2_DeepSupAttenU_FTL_AnimalTSTv1_ptAnimalCArmCTv1_ptptAnimalCTv1run2_ptptptCHAOSnoAug/Results_ct_/",
        # "CArm": "/mnt/public/soumick/CTPerf/Output/5_Fold/Fold5_Aug2_DeepSupAttenU_FTL_AnimalTSTv1_ptAnimalCArmCTv1_ptptAnimalCTv1run2_ptptptCHAOSnoAug/Results_carm_/",
        "TST": "/mnt/public/soumick/CTPerf/Output/5_Fold/Fold5_Aug2_DeepSupAttenU_FTL_AnimalTSTv1_ptAnimalCArmCTv1_ptptAnimalCTv1run2_ptptptCHAOSnoAug/Results_/",
    },
    "dirTST": {
        "TST": "/mnt/public/soumick/CTPerf/Output/5_Fold/Fold5_Aug2_DeepSupAttenU_FTL_AnimalTSTv1/Results_/"
    },
    "TSTptCHAOS": {
        "TST": "/mnt/public/soumick/CTPerf/Output/5_Fold/Fold5_Aug2_DeepSupAttenU_FTL_AnimalTSTv1_ptCHAOSnoAug/Results_/"
    },
    "dirCArm": {
        "CArm": "/mnt/public/soumick/CTPerf/Output/5_Fold/Fold5_Aug2_DeepSupAttenU_FTL_AnimalCArmCTv1/Results_/"
    },
    "CArmptCHAOS": {
        "CArm": "/mnt/public/soumick/CTPerf/Output/5_Fold/Fold5_Aug2_DeepSupAttenU_FTL_AnimalCArmCTv1_ptCHAOSnoAug/Results_/"
    },
    "CTnoCHAOS": {
        "CT": "/home/soumick/96/mnt/mnt/md1337/hana/nas/mnt/fuenf12er/hana/data/6FOLD5/Fold5_Aug2_DeepSupAttenU_FTL_AnimalCTv1run2/Results_CTv2/"
    },
    "CarmptCTnoCHAOS": {
        "CArm": "/home/soumick/96/mnt/mnt/md1337/hana/nas/mnt/fuenf12er/hana/data/6FOLD5/Fold5_Aug2_DeepSupAttenU_FTL_AnimalCArmCT_ptAnimalCTv1run2/Results_CBCT/"
    } 
}

foldlist = [fold0, fold1, fold2, fold3, fold4, fold5]
train_keys = foldlist[0].keys()
test_keys = ["CT", "CArm", "TST"] # foldlist[0][list(train_keys)[0]].keys()

consolidated_res = defaultdict(lambda: defaultdict(list))
for i in range(len(foldlist)):
    for train_k in foldlist[i].keys():
        for test_k in foldlist[i][train_k].keys():
            if bool(foldlist[i][train_k][test_k]):    
                df = pd.read_csv(foldlist[i][train_k][test_k]+"/output_scores.csv", index_col=0)
                df['FoldID'] = i
                df.reset_index(drop=True, inplace=True)
                consolidated_res[train_k][test_k].append(df)

with open(f"{consolidated_output}/consolidations_F6.txt","w") as file_obj:
    for train_k in train_keys:
        for test_k in test_keys:
            if len(consolidated_res[train_k][test_k]) > 0:
                df = pd.concat(consolidated_res[train_k][test_k], ignore_index=True)
                consolidated_res[train_k][test_k] = df
                df.to_csv(f"{consolidated_output}/{train_k}_{test_k}.csv")

                file_obj.write("\n----------------------------\n")
                file_obj.write(f"\nTrain{train_k}_Test{test_k}\n")
                file_obj.write("\n----------------------------\n")
        
                file_obj.write("\nMeanDice: "+ str(df["Dice"].mean().round(3)) + "±" + str(df["Dice"].std().round(3)))
                file_obj.write("\nMedianDice: "+ str(df["Dice"].median().round(3)) + "±" + str(df["Dice"].var().round(3)))
                file_obj.write("\nMeanIoU: "+ str(df["IoU"].mean().round(3)) + "±" + str(df["IoU"].std().round(3)))
                file_obj.write("\nMedianIoU: "+ str(df["IoU"].median().round(3)) + "±" + str(df["IoU"].var().round(3)))

                try:
                    file_obj.write("\nMeanPrecision: "+ str(df["Precision"].mean().round(3)) + "±" + str(df["Precision"].std().round(3)))
                    file_obj.write("\nMedianPrecision: "+ str(df["Precision"].median().round(3)) + "±" + str(df["Precision"].var().round(3)))
                    file_obj.write("\nMeanFPR: "+ str(df["FPR"].mean().round(3)) + "±" + str(df["FPR"].std().round(3)))
                    file_obj.write("\nMedianFPR: "+ str(df["FPR"].median().round(3)) + "±" + str(df["FPR"].var().round(3)))
                    file_obj.write("\nMeanSensitivity: "+ str(df["Sensitivity"].mean().round(3)) + "±" + str(df["Sensitivity"].std().round(3)))
                    file_obj.write("\nMedianSensitivity: "+ str(df["Sensitivity"].median().round(3)) + "±" + str(df["Sensitivity"].var().round(3)))
                    file_obj.write("\nMeanSpecificity: "+ str(df["Specificity"].mean().round(3)) + "±" + str(df["Specificity"].std().round(3)))
                    file_obj.write("\nMedianSpecificity: "+ str(df["Specificity"].median().round(3)) + "±" + str(df["Specificity"].var().round(3)))
                except:
                    pass


                file_obj.write("\nMeanRawDice: "+ str(df["RawDice"].mean().round(3)) + "±" + str(df["RawDice"].std().round(3)))
                file_obj.write("\nMedianRawDice: "+ str(df["RawDice"].median().round(3)) + "±" + str(df["RawDice"].var().round(3)))
                file_obj.write("\nMeanRawIoU: "+ str(df["RawIoU"].mean().round(3)) + "±" + str(df["RawIoU"].std().round(3)))
                file_obj.write("\nMedianRawIoU: "+ str(df["RawIoU"].median().round(3)) + "±" + str(df["RawIoU"].var().round(3)))

                try:
                    file_obj.write("\nMeanRawPrecision: "+ str(df["RawPrecision"].mean().round(3)) + "±" + str(df["RawPrecision"].std().round(3)))
                    file_obj.write("\nMedianRawPrecision: "+ str(df["RawPrecision"].median().round(3)) + "±" + str(df["RawPrecision"].var().round(3)))
                    file_obj.write("\nMeanRawFPR: "+ str(df["RawFPR"].mean().round(3)) + "±" + str(df["RawFPR"].std().round(3)))
                    file_obj.write("\nMedianRawFPR: "+ str(df["RawFPR"].median().round(3)) + "±" + str(df["RawFPR"].var().round(3)))
                    file_obj.write("\nMeanRawSensitivity: "+ str(df["RawSensitivity"].mean().round(3)) + "±" + str(df["RawSensitivity"].std().round(3)))
                    file_obj.write("\nMedianRawSensitivity: "+ str(df["RawSensitivity"].median().round(3)) + "±" + str(df["RawSensitivity"].var().round(3)))
                    file_obj.write("\nMeanRawSpecificity: "+ str(df["RawSpecificity"].mean().round(3)) + "±" + str(df["RawSpecificity"].std().round(3)))
                    file_obj.write("\nMedianRawSpecificity: "+ str(df["RawSpecificity"].median().round(3)) + "±" + str(df["RawSpecificity"].var().round(3)))
                except:
                    pass

                file_obj.write("\n----------------------------\n")
                file_obj.write("\n----------------------------\n")

            else:
                print("nulla!")

print("Done!")
