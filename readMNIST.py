import numpy as np
import matplotlib.pyplot as plt
import copy

def read_data(train_val_split=0.8):
    """
    :param train_val_split: proportion of data being used for training vs. validation
    :return: training, validation and test images and labels
    """

    f = open('train-images-idx3-ubyte', 'r')
    a = np.fromfile(f, dtype='>i4', count=4) # data type is signed integer big-endian
    TRAIN_images = np.fromfile(f, dtype=np.uint8)

    f = open('train-labels-idx1-ubyte', 'r')
    t = np.fromfile(f, count = 2, dtype='>i4') # data type is signed integer big-endian
    TRAIN_labels = np.fromfile(f, dtype=np.uint8)

    N = int(train_val_split*(TRAIN_images.size/784))
    train_images = TRAIN_images[:(N*784)].reshape(N, 784)
    train_labels = TRAIN_labels[:N]
    b = np.zeros((train_labels.size, train_labels.max()+1))
    b[np.arange(train_labels.size),train_labels] = 1
    train_labels = copy.deepcopy(b)

    val_images = TRAIN_images[(N*784):].reshape(TRAIN_labels.size - N, 784)
    val_labels = TRAIN_labels[N:]
    b = np.zeros((val_labels.size, val_labels.max()+1))
    b[np.arange(val_labels.size),val_labels] = 1
    val_labels = copy.deepcopy(b)

    f = open('t10k-labels-idx1-ubyte', 'r')
    t = np.fromfile(f, count = 2, dtype='>i4') # data type is signed integer big-endian
    test_labels = np.fromfile(f, dtype=np.uint8)

    f = open('t10k-images-idx3-ubyte', 'r')
    a = np.fromfile(f, dtype='>i4', count=4) # data type is signed integer big-endian
    TEST_images = np.fromfile(f, dtype=np.uint8)
    test_images = TEST_images.reshape(test_labels.size, 784)

    b = np.zeros((test_labels.size, test_labels.max()+1))
    b[np.arange(test_labels.size),test_labels] = 1
    test_labels = copy.deepcopy(b)

    # centering images
    train_images = np.subtract(train_images, np.mean(train_images))
    val_images = np.subtract(val_images, np.mean(val_images))
    test_images = np.subtract(test_images, np.mean(test_images))
    return train_images, train_labels, val_images, val_labels, test_images, test_labels

if __name__ == '__main__':
    train_images, train_labels, val_images, val_labels, test_images, test_labels = read_data(train_val_split=0.8)

    #%% Show random train, validation and test images
    img = train_images[int(784*4):int(784*5)].reshape((28, 28))
    plt.imshow(img, cmap='gray')
    plt.show()

    img = val_images[int(784*4):int(784*5)].reshape((28, 28))
    plt.imshow(img, cmap='gray')
    plt.show()

    img = test_images[int(784*4):int(784*5)].reshape((28, 28))
    plt.imshow(img, cmap='gray')
    plt.show()

