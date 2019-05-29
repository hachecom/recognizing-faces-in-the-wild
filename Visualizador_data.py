import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from PIL import Image
from pathlib import Path
import sys, os
import argparse


ap = argparse.ArgumentParser()
ap.add_argument(
    "-d",
    "--dataset",
    required=True,
    help="path to input datasforlder  (i.e., directory of train, test and cvs)",
)
args = vars(ap.parse_args())


# Reading the dataset
train_df = pd.read_csv(args["dataset"] + "/train_relationships.csv")
print(train_df.tail())

TRAIN_BASE = args["dataset"] + "/train/"
families = sorted(os.listdir(TRAIN_BASE))
print("We have {} families in the dataset".format(len(families)))
print(families[:5])

members = {i: sorted(os.listdir(TRAIN_BASE + i)) for i in families}

TEST_BASE = args["dataset"] + "/test/"
test_images_names = os.listdir(TEST_BASE)
print(test_images_names[:5])

# VISUALIZING THE DATASET
def load_img(PATH):
    return np.array(Image.open(PATH))


def plots(ims, figsize=(12, 6), rows=1, titles=None):
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims) // rows, i + 1)
        sp.axis("Off")
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i])
    plt.show()


def plot_relations(df, BASE=args["dataset"] + "/train/", rows=1, titles=None):
    tdf = df[:rows]
    tdf1 = tdf.p1
    tdf2 = tdf.p2
    figsize = (20, 20)
    f = plt.figure(figsize=figsize)
    x = 0
    for i in range(rows):
        sp = f.add_subplot(rows, 2, x + 1)
        sp.axis("Off")
        x += 1
        image_path = os.path.join(BASE, tdf1[i])
        im = os.listdir(image_path)[-1]
        sp.set_title(tdf1[i], fontsize=12)
        plt.imshow(load_img(os.path.join(image_path, im)))

        sp = f.add_subplot(rows, 2, x + 1)
        x += 1
        sp.axis("Off")
        image_path = os.path.join(BASE, tdf2[i])
        im = os.listdir(image_path)[-1]
        sp.set_title(tdf2[i], fontsize=12)
        plt.imshow(load_img(os.path.join(image_path, im)))
    plt.show()


# Plot the relationships
plot_relations(train_df, rows=10)  # uncomment this to plot
print(train_df.shape)


# Plot the the shape of test images and the test images itself
test_images = np.array(
    [load_img(os.path.join(TEST_BASE, image)) for image in test_images_names[:1]]
)  # save test images into np arrays
# print(len(test_images))
print(
    "(n. of test images, pixels, pixels, channels): ", test_images.shape
)  # shape check of test images
plots(test_images[:15], rows=3)  # plot test images
