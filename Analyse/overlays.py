import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt

def create_diff_mask_binary(predicted, label):
    """
    Method, find the difference between the 2 binary images and overlay colours
    predicted, label : slices , 2D tensor
    """

    diff1 = np.subtract(label, predicted) > 0 # under_detected
    diff2 = np.subtract(predicted, label) > 0 # over_detected

    predicted = predicted > 0

    # Define colours
    red = np.array([255, 0, 0], dtype=np.uint8)  # under_detected
    green = np.array([0, 255, 0], dtype=np.uint8)  # over_detected
    black = np.array([0, 0, 0], dtype=np.uint8)  # background
    white = np.array([255, 255, 255], dtype=np.uint8)  # prediction_output
    blue = np.array([0, 0, 255], dtype=np.uint8) # over_detected
    yellow = np.array([255, 255, 0], dtype=np.uint8)  # under_detected
    dead_magenta = np.array([244, 154, 194], dtype=np.uint8) #under_detected
    dead_cyan = np.array([164, 216, 216],  dtype=np.uint8) #over_detected
    pastel_red = np.array([255, 105, 97], dtype=np.uint8) #under
    pastel_blue = np.array([48,206,216], dtype=np.uint8) #over
    # Make RGB array, pre-filled with black(background)
    rgb_image = np.zeros((*predicted.shape, 3), dtype=np.uint8) + black

    # Overwrite with red where threshold exceeded, i.e. where mask is True
    rgb_image[predicted] = white
    rgb_image[diff1] = pastel_red
    rgb_image[diff2] = pastel_blue
    return rgb_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_file",
                        default="/mnt/public/soumick/CTPerf/Output/Aug2_DeepSupAttenU_FTL_AnimalTSTv1_ptAnimalCArmCTv1_ptptAnimalCTv1run2_ptptptCHAOSnoAug/Results/output.hdf5",
                        #default="/mnt/public/soumick/CTPerf/Output/Fold2_Aug2_DeepSupAttenU_FTL_AnimalCArmCTv1_ptAnimalCTv1run2_ptptCHAOSnoAug/Results_carm_/output.hdf5",
                        help="Output file of testing - raw output file or post-processed.")
    parser.add_argument("--gt_file",
                        # default="/mnt/public/soumick/CTPerf/Data/CHAOS.hdf5",
                        # default="/mnt/public/soumick/CTPerf/Data/AnimalCTv1.hdf5",
                        default="/mnt/public/soumick/CTPerf/Data/AnimalCArmCTv1.hdf5",
                        # default="/mnt/public/soumick/CTPerf/Data/AnimalTSTv1.hdf5",
                        help="Ground truth file."),
    parser.add_argument("--out_path",
                        default="/mnt/public/soumick/CTPerf/Output/Consolidated/Plots",
                        help="Path to the folder where output will be saved.")
    parser.add_argument("--out_type",
                        default="Blabla",
                        help="Type of result currently being tested.")
    parser.add_argument("--animalID",
                        default="1508",
                        help="animal ID.")
    parser.add_argument("--sliceID",
                        default=13,
                        type=int,
                        help="slice ID.")
    args = parser.parse_args()

    predicted = h5py.File(f"{args.result_file}", mode="r")[
        f"animal_{args.animalID}"
    ][args.sliceID]

    label = h5py.File(f"{args.gt_file}", mode="r")[f"animal_{args.animalID}"][
        args.sliceID, 1
    ]


    overlayed = create_diff_mask_binary(predicted, label)
    plt.imshow(overlayed)
    plt.tight_layout()
    # plt.show()
    plt.axis('off')
    plt.savefig(f"{args.out_path}/overlay_{args.out_type}_animal{args.animalID}_slice{args.sliceID}.png", format='png', bbox_inches='tight',pad_inches = 0)
