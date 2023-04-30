import os
import random

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torchvision.transforms.functional import perspective
import urllib3
from PIL import Image
from io import BytesIO
from skimage.transform import resize
from tqdm import tqdm

from data import loadData, toPNG

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_CHANNEL = 3
PATH_JPG = "Data/JPG"
PATH_PNG = "Data/PNG"
PATH_RESIZED = "Data/Resized"
PATH_TRANSFORMED = "Data/Transformed"
PATH_MASK = "Data/Mask"
# PIXEL_MM_RATIO = 1.5
SEED = 29


# parent class for all transformation methods
class Method:

    # initialize the name of the method, should be called using super()
    def __init__(self, name):
        self._name = name

    # get method name
    def getName(self):
        return self._name

    # the transform algorithm for the method, expected to be override
    def transform(self, img, shift):
        pass

    # the mask algorithm for the method, expected to be overrided
    def mask(self, img, shift):
        pass


class RectanglePerspectiveMethod(Method):

    def __init__(self):
        super().__init__("Rectangle Perspective")

    def transform(self, img, shift):

        # get image vars
        height, width, channels = img.shape

        # apply transformer to left brain
        imgLeft = [perspective(torch.from_numpy(img[:, :, i])[:, :, None],
                               [[20, 40], [20, width // 2 - 1], [100, 40], [100, width // 2 - 1]],
                               [[20, 40], [20, width // 2 - 1 - shift], [100, 40],
                                [100, width // 2 - 1 - shift]]) for i in range(channels)]
        imgLeft = torch.squeeze(torch.stack(imgLeft, -1))

        # apply transformer to right brain
        imgRight = [perspective(torch.from_numpy(img[:, :, i])[:, :, None],
                                [[20, width // 2], [20, 80], [100, width // 2], [100, 80]],
                                [[20, width // 2 - shift], [20, 80], [100, width // 2 - shift],
                                 [100, 80]]) for i in range(channels)]
        imgRight = torch.squeeze(torch.stack(imgRight, -1))

        # put distorted value into original image
        for i in range(20, 101):
            for j in range(40, width // 2 - shift):
                for k in range(channels):
                    img[i, j, k] = imgLeft[i, j, k]

        for i in range(20, 101):
            for j in range(width // 2 - shift, 81):
                for k in range(channels):
                    img[i, j, k] = imgRight[i, j, k]

        return img

    def mask(self, img, shift):

        # initialize result
        result = np.zeros((*img[:-1], 1))

        # get image vars
        height, width, channels = img.shape

        # put distorted value into original image
        for i in range(height):
            target = width // 2
            if i < 20:
                target -= shift * i // 20
            elif i > 100:
                target -= shift * (height - i) // 28
            else:
                target -= shift
            for j in range(target):
                result[i, j, 0] = img[i, j, 0]

        return result


def transform(data=None, shift=5, method=RectanglePerspectiveMethod, inPath=PATH_JPG, outPath=PATH_TRANSFORMED):

    # set seed for pytorch
    torch.manual_seed(SEED)

    # transform input images
    if data is not None:

        # initialize result
        result = np.zeros(data.shape)

        # get size of dataset
        size = data.shape[0]

        # initialize tqdm bar for visualization
        bar = tqdm(enumerate(data), total=size)
        bar.set_description(f"Transforming images with shift={shift}")

        # iterate all images
        for ind, img in bar:

            # apply transformation method
            result[ind] = method.transform(img, shift)

        return result

    # transform images from file and save
    else:

        # get size of dataset
        size = len(os.listdir(inPath))

        # initialize tqdm bar for visualization
        bar = tqdm(enumerate(os.listdir(inPath)), total=size)
        bar.set_description(f"Transforming images with shift={shift}")

        # iterate all files
        for ind, file in bar:

            # read image to ndarray
            img = plt.imread(f"{inPath}/{file}")
            img = img[:, :, :IMAGE_CHANNEL]

            # resize ndarray
            img = resize(img, (128, 128), mode='constant', preserve_range=False)

            # apply transformation method
            img = method.transform(img, shift)

            # save image
            plt.imshow(img, interpolation='nearest')
            plt.savefig(f"{outPath}/s{shift}-{file}")


def mask(data=None, shift=5, method=RectanglePerspectiveMethod, save=False, outPath=PATH_MASK):

    # initialize result
    result = np.zeros((*data.shape[:-1], 1))

    # get size of dataset
    size = data.shape[0]

    # initialize tqdm bar for visualization
    bar = tqdm(enumerate(data), total=size)
    bar.set_description(f"Creating masks with shift={shift}")

    # iterate all images
    for ind, img in bar:

        result[ind, :, :, :] = method.mask(img, shift)
        print(np.average(result[ind, :, :, :]))

        # save image
        if save:
            plt.imshow(result[ind, :, :, :], interpolation='nearest')
            plt.savefig(f"{outPath}/s{shift}-{ind + 1}.jpg")

    return result


def example():

    # load example image
    url, filename = (
        "https://github.com/mateuszbuda/brain-segmentation-pytorch/raw/master/assets/TCGA_CS_4944.png",
        "example.png")
    http = urllib3.PoolManager()
    resp = http.request("GET", url)
    img = Image.open(BytesIO(resp.data))

    # save image to local
    img.save(filename)

    # load pretrained model
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                           in_channels=3, out_channels=1, init_features=32, pretrained=True)

    # perform transformations
    input_image = Image.open(filename)
    m, s = np.mean(input_image, axis=(0, 1)), np.std(input_image, axis=(0, 1))
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=m, std=s),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    # apply model to example
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model = model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)

    # result
    # print(torch.round(output[0]))
    print(torch.sum(torch.round(output[0])))


if __name__ == '__main__':

    images = loadData(height=IMAGE_HEIGHT,
                      width=IMAGE_WIDTH,
                      channel=IMAGE_CHANNEL,
                      inPath=PATH_JPG,
                      outPath=PATH_RESIZED)

    shiftmm = 5
    method = RectanglePerspectiveMethod()
    transformed = transform(images, shift=shiftmm, method=method)
    mask = mask(images, shift=shiftmm, method=method, save=True)

    print("Done!")
