import numpy as np
import sys
from tensorflow.keras.models import load_model

from data import loadData
from transform import RectanglePerspectiveMethod, transform, mask
from train import getModel, train, plotResult, plotComparison, getDiceScore

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


if __name__ == '__main__':

    # load dataset
    images = loadData(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, channel=IMAGE_CHANNEL)

    # define shift and method
    shiftmm = 5 if len(sys.argv) <= 1 else int(sys.argv[1].split(".")[0])
    method = RectanglePerspectiveMethod()

    # get and save transformed data
    transformed = transform(images, shift=shiftmm, method=method)
    np.save(f'Data/s{shiftmm}trainX', transformed)

    # get and save mask
    mask = mask(images, shift=shiftmm, method=method, save=True)
    np.save(f'Data/s{shiftmm}trainY', mask)

    # set path
    path = f"Model/Unet-s{shiftmm}.h5"

    # load training data
    trainX, trainY = np.load(f'Data/s{shiftmm}trainX.npy'), np.load(f'Data/s{shiftmm}trainY.npy')

    # get model
    model = getModel(trainX.shape[1:])

    # get training result
    result = train(model, trainX, trainY, path=path)

    # plot training result
    plotResult(result, shift=shiftmm)

    # reload trained model
    # model = load_model(path)

    # get model predictions
    prediction = (model.predict(trainX) > 0.5).astype(np.uint8)

    # plot image, mask and predictions
    plotComparison(trainX, trainY, prediction, shift=shiftmm)

    # print numerical results
    print(f"Average dice score for the model is {getDiceScore(trainY, prediction):.4f}")

    print("\n\nDone!")
