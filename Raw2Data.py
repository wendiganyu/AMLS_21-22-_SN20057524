"""
Transform the raw MRI dataset files to the formats we use in processing the programs.
For example, matrix, pandas dataframes, etc.
"""
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib as plt


def gen_mri_mtx_binary(data_dir, label_path):
    """
    Convert the 3000 MRI dataset .ipg files into a big matrix
    Input
        dir：location of the dataset directory
    Return
        mri_mtx: MRI dataset matrix. Each image is represented as
                 a row of 262145 * 1. The first 262144 (512 * 512)
                 elements are data points of image. The last 1 element
                 is its class (0:no_tumor, 1:with_tumor).
    """

    # mri_mtx = []
    mri_mtx = np.empty((0,262145), int)
    df = pd.read_csv(label_path)
    df = df.set_index('file_name')

    for fileName in os.listdir(data_dir):
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
        # mri_mtx.append([data_vec])
    return mri_mtx


data_dir = "dataset/image"
label_path = "dataset/label.csv"
mri_mtx = gen_mri_mtx_binary(data_dir, label_path)
print(mri_mtx)
np.savetxt("MRI_Matrix.txt", mri_mtx, fmt="%d", delimiter=",")
