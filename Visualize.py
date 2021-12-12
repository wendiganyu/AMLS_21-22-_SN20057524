import numpy as np
import matplotlib.pyplot as plt
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
        plt.text( index, value, str(value), ha='center')
    # plt.xlabel("classes of brain tumors")
    plt.ylabel("sample numbers")
    plt.savefig("report_images/classDistribution.pdf")
    plt.show()

if __name__ == '__main__':
    X,Y = PreProcessing.gen_X_Y(is_mul=True)
    dataset_pixel_distribution(X)
    # dataset_classes_distribution(Y)
