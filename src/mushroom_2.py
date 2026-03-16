import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import skimage.morphology as skmorph
from glob import glob
import skimage.measure as regionprops

from myDIP.segmentations import colorRange, kmeans
from myDIP.morphology import removeFragments, fillHoles


def getLargestConnectedComponent(binary_img):
    '''
        Get the largest component in the Input Image
    '''
    binary_img = binary_img.astype(np.uint8)
    # -> Connected Components
    _, label_img = cv.connectedComponents(binary_img)
    # -> Count Number of pixel of each label
    labels, counts = np.unique(label_img, return_counts=True)

    # -> Remove Background
    counts = counts[labels!=0]
    labels = labels[labels!=0]

    # -> Get largest component
    largest_group_label = labels[np.argmax(counts)]
    output_img = np.zeros_like(binary_img, np.uint8)
    output_img[label_img==largest_group_label] = 1

    return output_img



if __name__ == "__main__":

    input_path = r"../Datasets/Mushroom_Segmentation_2569/images/mushroom_2.png"

    output_path = r"../Datasets/Mushroom_Segmentation_2569/output/"

    input_img = cv.imread(input_path)

    base_filename = os.path.basename(input_path)

    output_path_name = output_path + base_filename

    input_img = cv.imread(input_path)

    # RGB image
    rgb_img = cv.cvtColor(input_img, cv.COLOR_BGR2RGB)

    # HSV image
    hsv_img = cv.cvtColor(input_img, cv.COLOR_BGR2HSV)


    hue_img = hsv_img[:,:,0]
    saturation_img = hsv_img[:,:,1]
    value_img = hsv_img[:,:,2]

    plt.figure()
    plt.imshow(rgb_img)          
    plt.figure()
    plt.imshow(hsv_img)
    plt.figure()
    plt.imshow(hue_img, cmap="hsv")
    plt.figure()
    plt.imshow(saturation_img, cmap="gray")
    plt.figure()
    plt.imshow(value_img, cmap="gray")
    

    lower_stipe = np.array([0, 0, 58])
    upper_stipe = np.array([120, 65, 255])
    mask_thresholding = cv.inRange(hsv_img, lower_stipe, upper_stipe)


    ### => remove small fragments
    mask = removeFragments(mask_thresholding, thresh_percent=0.01)


    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    ### => closing 
    close_img = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)


    ### => fill hole segment
    fillhole_img = fillHoles(close_img)

    output_mask = getLargestConnectedComponent(fillhole_img)

    output_mask = output_mask * 255

    cv.imwrite(output_path_name, output_mask)




    plt.figure()
    plt.imshow(mask_thresholding, cmap="gray")
    plt.figure()
    plt.imshow(mask, cmap="gray")
    plt.figure()
    plt.imshow(close_img, cmap="gray")
    plt.figure()
    plt.imshow(fillhole_img, cmap="gray")
    plt.figure()
    plt.imshow(output_mask, cmap="gray")
    plt.show()
