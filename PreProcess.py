import cv2
import numpy as np


def PreProc(X,a,b):
    resize_size = (a, b)
    list=[]
    for i in range(len(X)):
        img = cv2.resize(X[i], resize_size)
        img = img.astype('float64')  # visualize the mean image
        list.append(img)
    X = np.array(list)
    print(X.shape)
    X = np.reshape(X, (X.shape[0], -1))
    print('Training data shape: ', X.shape)
    return X

def CalcMean(X,a,b):
    # Preprocessing: subtract the mean image
    import matplotlib.pyplot as plt
    # first: compute the image mean based on the training data
    mean_image = np.mean(X, axis=0)
    print(mean_image[:10])  # print a few of the elements
    plt.figure(figsize=(4, 4))
    plt.imshow(mean_image.reshape((a, b, 3)).astype('uint8'))  # visualize the mean image
    plt.show()
    return mean_image
def Minus_mean(X,mean):
    X -= mean
    return X
def Bias_trick(X):
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    print(X.shape)  # , X_dev.shape
    return X