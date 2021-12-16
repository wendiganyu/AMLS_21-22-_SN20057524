import cv2
import numpy
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold

import PreProcessing


def dataset_pixel_distribution(dataset):
    x = np.arange(0, 256)
    distribution = np.zeros(256, int)
    counter = 0
    for image in dataset:
        print(counter)
        for pixel in image:
            distribution[pixel] += 1
        counter += 1
    plt.bar(x, distribution)
    plt.xlabel("pixel value")
    plt.ylabel("number")
    plt.savefig("report_images/pixelDistribution.pdf")

    plt.show()


def dataset_classes_distribution(labelset):
    x = ["no_tumor", "meningioma_tumor", "glioma_tumor", "pituitary_tumor"]
    distribution = np.zeros(4, int)
    counter = 0
    for label in labelset:
        print(counter)
        distribution[label] += 1
        counter += 1

    plt.bar(x, distribution)
    for index, value in enumerate(distribution):
        plt.text(index, value, str(value), ha='center')
    # plt.xlabel("classes of brain tumors")
    plt.ylabel("sample numbers")
    plt.savefig("report_images/classDistribution.pdf")
    plt.show()


def dataset_samples_show(dir_path):
    plt.subplot(1, 4, 1)
    plt.axis('off')
    plt.imshow(Image.open(dir_path + "IMAGE_0000.jpg"))

    plt.subplot(1, 4, 2)
    plt.axis('off')
    plt.imshow(Image.open(dir_path + "IMAGE_0001.jpg"))

    plt.subplot(1, 4, 3)
    plt.axis('off')
    plt.imshow(Image.open(dir_path + "IMAGE_0003.jpg"))

    plt.subplot(1, 4, 4)
    plt.axis('off')
    plt.imshow(Image.open(dir_path + "IMAGE_0009.jpg"))

    plt.savefig("report_images/datasetSamples.pdf")
    plt.show()


if __name__ == '__main__':
    abc = [0.99, 0.99, 0.98, 0.995,  0.995, 0.995, 0.995, 0.995, 0.995,1]
    print(np.array(abc).std())