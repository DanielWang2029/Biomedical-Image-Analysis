import sys
import random
import warnings #
import pandas as pd
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.morphology import label
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout, Lambda
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def getModel(shape=(128, 128, 3)):

    # define inputs
    inputs = Input(shape)
    s = Lambda(lambda x: x) (inputs)
    # s = Lambda(lambda x: x / 255) (inputs)

    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], run_eagerly=True)

    return model


def train(model, trainX, trainY, path="Model/Unet.h5", validation_split=0.1, batch_size=8, epochs=100):

    # tf.compat.v1.disable_eager_execution()
    model_path = path

    # create checkpoint for model
    checkpoint = ModelCheckpoint(model_path,
                                 monitor="val_loss",
                                 mode="min",
                                 save_best_only=True,
                                 verbose=1)

    # create early stop criteria for model
    earlystop = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=5,
                              verbose=1,
                              restore_best_weights=True)

    # train model
    return model.fit(trainX,
                     trainY,
                     validation_split=validation_split,
                     batch_size=batch_size,
                     epochs=epochs,
                     callbacks=[earlystop, checkpoint])


def plotResult(result, shift=5):

    # plot training accuracy
    plt.plot(result.history['accuracy'])
    plt.plot(result.history['val_accuracy'])
    plt.title(f'model accuracy with shift={shift}')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f'Model/Validation-s{shift}.png')
    plt.clf()

    # plot training loss
    plt.plot(result.history['loss'])
    plt.plot(result.history['val_loss'])
    plt.title(f'model loss with shift={shift}')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f'Model/Loss-s{shift}.png')


def plotComparison(trainX, trainY, prediction, shift=5):

    # initialize tqdm bar for visualization
    bar = tqdm(enumerate(trainX), total=trainX.shape[0])
    bar.set_description(f"Saving result comparison with shift={shift}")

    # iterate all images
    for ix, img in bar:

        # set canvas
        plt.figure(figsize=(20, 20))

        # plot original training image
        plt.subplot(131)
        imshow(img)
        plt.title("Image")

        # plot original mask
        plt.subplot(132)
        imshow(np.squeeze(trainY[ix]))
        plt.title("Mask")

        # plot mask model predicts
        plt.subplot(133)
        imshow(np.squeeze(prediction[ix] > 0.5))
        plt.title("Predictions")

        # set title and save
        plt.suptitle(f"Mask vs Prediction s{shift}-{ix + 1}")
        plt.savefig(f'Data/Comparison/s{shift}-{ix + 1}.png')
        plt.clf()
        plt.close()


def getDiceScore(actual, pred):

    # initialize score container
    scores = []

    # iterate all results
    for ind, img in enumerate(actual):

        # calculate dice score
        scores.append(2 * np.sum(img * pred[ind]) / (np.sum(img) + np.sum(pred[ind])))

    return np.mean(scores)


if __name__ == '__main__':

    # set path to save model and shift
    shiftmm = 5
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
