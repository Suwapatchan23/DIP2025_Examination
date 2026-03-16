from glob import glob
import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

from myDIP.evaluation import iou, overallConfusionMatrix


if __name__ == "__main__":

    input_dir = r"../Datasets/Mushroom_Segmentation_2569/output"
    groundtruth_dir = r"../Datasets/Mushroom_Segmentation_2569/gt"

    input_path_list = sorted(glob(input_dir + "\*"))
    groundtruth_path_list = sorted(glob(groundtruth_dir + "\*"))

    output_img_list = []
    groundtruth_img_list = []

    for input_path, groundtruth_path in zip(input_path_list, groundtruth_path_list):

        ### -> Read Input Image
        input_img = cv.imread(input_path, 0)
        groundtruth_img = cv.imread(groundtruth_path, 0)

        input_img = np.where(input_img > 128, 255, 0).astype(np.uint8)
        groundtruth_img = np.where(groundtruth_img > 128, 255, 0).astype(np.uint8)

        ### -> Show IoU
        print(f"{os.path.basename(input_path)}: IoU = {iou(input_img, groundtruth_img):.3f}")

        output_img_list.append(input_img)
        groundtruth_img_list.append(groundtruth_img)

        plt.figure()
        plt.imshow(input_img, cmap="gray")
        plt.figure()
        plt.imshow(groundtruth_img, cmap="gray")
        plt.show()

    ### -> Evaluation
    overall_conf_mat = overallConfusionMatrix(output_img_list, groundtruth_img_list)

    ### -> Show Confusion Matrix
    np.set_printoptions(suppress=True, precision=3)
    print(overall_conf_mat)
    plt.show()