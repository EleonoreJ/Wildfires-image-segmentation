import os
import numpy as np
import matplotlib.pyplot as plt

def pngBand2rgb(dataPath, dataName, savePath, saveName, bands):
    R = plt.imread(dataPath + "{}.{}.png".format(dataName, bands[0]))[..., 0]  # read data
    G = plt.imread(dataPath + "{}.{}.png".format(dataName,  bands[1]))[..., 0]
    B = plt.imread(dataPath + "{}.{}.png".format(dataName,  bands[2]))[..., 0]

    jpg_data = np.zeros([R.shape[0], R.shape[1], 3])
    jpg_data[..., 0] = R
    jpg_data[..., 1] = G
    jpg_data[..., 2] = B

    # jpg_data = data.transpose(1, 2, 0)

    if not os.path.exists(savePath):
        os.makedirs(savePath)
    plt.imsave(savePath + saveName + ".png", jpg_data)
