
import os
from tqdm import tqdm
from PIL import Image, ImageOps
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np


def toPNG(inPath="Data/JPG", outPath="Data/PNG"):

    # iterate all files
    for file in os.listdir(inPath):

        # read image
        img = Image.open(f"{inPath}/{file}")

        # save image to PNG
        img.save(f"{outPath}/{''.join(file.split('.')[:-1])}.png")


def expand(img, size):

    # get target size canvas
    img.thumbnail((size[0], size[1]))

    # calculate padding
    dWidth, dHeight = size[0] - img.size[0], size[1] - img.size[1]
    pWidth, pHeight = dWidth // 2, dHeight // 2
    padding = (pWidth, pHeight, dWidth - pWidth, dHeight - pHeight)

    # expand image and fill with black (fill=0)
    imgResized = ImageOps.expand(img, padding, 0)

    return imgResized


def expandData(size=1024, inPath="Data/PNG", outPath="Data/Resized"):

    # iterate all files
    for index, file in enumerate(os.listdir(inPath)):

        # read image
        img = Image.open(f"{inPath}/{file}")

        # expand image
        img = expand(img, (size, size))

        # save image
        img.save(f"{outPath}/{index + 1}.png")


def loadData(height=128, width=128, channel=3, save=False, inPath="Data/JPG", outPath="Data/Resized"):

    # get size of dataset
    size = len(os.listdir(inPath))

    # initialize return
    result = np.zeros((size, height, width, channel))

    # initialize tqdm bar for visualization
    bar = tqdm(enumerate(os.listdir(inPath)), total=size)
    bar.set_description("Processing images" if save else "Loading images")

    # iterate all files
    for ind, file in bar:

        # read image to ndarray
        img = plt.imread(f"{inPath}/{file}")
        img = img[:, :, :channel]

        # resize ndarray
        img = resize(img, (height, width), mode='constant', preserve_range=False)

        # save image
        if save:
            plt.imshow(img, interpolation='nearest')
            plt.savefig(f"{outPath}/{file}")

        # update return
        result[ind, :, :, :] = img

    return result

