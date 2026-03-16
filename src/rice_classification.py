import os 
from glob import glob
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops


from myDIP.evaluation import overallConfusionMatrix, confusionMatrix

def segmentation(img):

    segment_img = np.where(img > 127, 1, 0)
    segment_img = segment_img.astype(np.uint8) * 255.0

    return segment_img

def featureExtraction(seg_img):

    ### -> object define by Connected Components
    _, label_img = cv.connectedComponents(seg_img)

    object_list = regionprops(label_img)

    for object in object_list:

        eccentricity = object.eccentricity

        axis_major_length = object.axis_major_length
        

    # -> Feature Vector
    feature_vector = [eccentricity, axis_major_length]
    
    return feature_vector

def classifier(feature_vector):
    '''
        Manual Decision Tree Classifier
    '''
    eccentricity, axis_major_length  = feature_vector

    if eccentricity <= 0.8:
        pred_class = "Karacadag"
    else:
        if 0.8 < eccentricity <= 0.9:
            pred_class = "Arborio"
        else:
            if axis_major_length >= 180:
                pred_class = "Basmati"
            else:
                pred_class = "Jasmine"

    return pred_class


if __name__ == "__main__":

    ### -> Set base directory
    base_dir = r"../Datasets/Rice_Classification_2569"

    ### -> Get class name from folder name
    class_name = os.listdir(base_dir)

    y_true = []
    y_pred = []

    for class_label in class_name:

        ### -> Class Directory
        class_dir = os.path.join(base_dir, class_label)
        img_path_list = sorted(glob(class_dir + "\*")) # - Get Image PATH

        for img_path in img_path_list:

            ### -> Read Input Image
            img = cv.imread(img_path)

            ### -> Color Model Conversion
            rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            ### -> Segmentation
            seg_img = segmentation(gray_img).astype(np.uint8)


            ### -> Feature Extraction
            feature_vector = featureExtraction(seg_img)

            ### -> Classification
            pred_class = classifier(feature_vector)


            y_true.append(class_label)
            y_pred.append(pred_class)

    ### -> Evaluation
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    conf_mat = confusionMatrix(y_pred, y_true)

    ### -> Show Confusion Matrix
    np.set_printoptions(suppress=True, precision=3)
    print(conf_mat)