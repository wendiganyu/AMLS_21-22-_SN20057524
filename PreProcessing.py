import os
import cv2
import pandas as pd
import numpy as np


def gen_mri_mtx_binary_label(data_dir, label_path):
    """
    Transform the 3000 raw .jpg images in MRI dataset combined with
    their binary label data indicating the existence of brain tumor or not
    together to a big numpy matrix we could use in program processing.

    Input
        dir：location of the dataset directory
    Return
        mri_mtx: MRI dataset matrix. Each image is represented as
                 a row of 262145 * 1. The first 262144 (512 * 512)
                 elements are data points of image. The last 1 element
                 is its class (0:no_tumor, 1:with_tumor).
    """

    mri_mtx = np.empty((0, 262145), int)
    df = pd.read_csv(label_path)
    df = df.set_index('file_name')

    for fileName in sorted(os.listdir(data_dir)):
        print("This is img " + fileName)
        # Read an image as a matrix.
        full_path = os.path.join(data_dir, fileName)
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

        # Determine this image's class
        if df.loc[fileName, "label"] == "no_tumor":
            img_class = 0  # 0: this img indicates no tumor, vice versa.
        else:
            img_class = 1

        # Convert original image matrix to a row vector appended with its class.
        img_row_vec = img.flatten()
        data_vec = np.append(img_row_vec, img_class)

        # Append this row data vector to mri_matrix.
        mri_mtx = np.append(mri_mtx, [data_vec], axis=0)
    return mri_mtx


def standardization(X):
    """
    Perform Z-score standardization to input data. x' = (x - μ)／σ

    Inputs
        X: Input data as a form of numpy matrix. Each row represents one data element.

    Return
        Data after standardization.
    """
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    return (X - mu) / sigma


if __name__ == "__main__":
    data_dir = "dataset/image"
    label_path = "dataset/label.csv"
    mtx_file_name = "MRI_Matrix.npy"

    # Check if the data matrix already saved as file.
    if os.path.exists(mtx_file_name):
        mri_mtx = np.load(mtx_file_name)
    else:
        mri_mtx = gen_mri_mtx_binary_label(data_dir, label_path)
        np.save(mtx_file_name, mri_mtx)
    print(mri_mtx.shape)
