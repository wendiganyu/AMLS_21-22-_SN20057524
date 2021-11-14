"""
Main Python file which calls other modules in this directory to realise a whole process from loading original dataset
to outputting the classification accuracy.
"""
import PreProcessing
import FeatureExtraction

data_dir = "dataset/image"  # Path of dataset directory
label_path = "dataset/label.csv"  # Path of dataset's label file

# Path of the .npy file which saves the dataset and its binary labels' information as a matrix.
original_mtx_path = "MRI_Matrix.npy"


def pca_SVM():
    """
    
    :return:
    """
