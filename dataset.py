import gzip
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from PIL import Image

PATH = "E:/ML/mnist"

def dataset(validation=True):

    image_size = 28
    num_train_images = 60000
    num_test_images = 10000

    # training images
    f = gzip.open(
        PATH + '/train-images-idx3-ubyte.gz', 'r')
    f.read(16)
    buf = f.read(image_size * image_size * num_train_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    train_x = data.reshape(num_train_images, image_size, image_size, 1)
    # training labels
    label = []
    f = gzip.open(
        PATH + '/train-labels-idx1-ubyte.gz', 'r')
    f.read(8)
    for i in range(0, num_train_images):
        buf = f.read(1)
        label.append(np.frombuffer(buf, dtype=np.uint8).astype(np.int64)[0])
    train_y = np.asarray(label)

    # test images
    f = gzip.open(
        PATH + '/t10k-images-idx3-ubyte.gz', 'r')
    f.read(16)
    buf = f.read(image_size * image_size * num_test_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    test_x = data.reshape(num_test_images, image_size, image_size, 1)
    # training labels
    label = []
    f = gzip.open(
        PATH + '/t10k-labels-idx1-ubyte.gz', 'r')
    f.read(8)
    for i in range(0, num_test_images):
        buf = f.read(1)
        label.append(np.frombuffer(buf, dtype=np.uint8).astype(np.int64)[0])
    test_y = np.asarray(label)

    if validation:
        # val images and label
        val_x = []
        val_y = []
        for i in range(0, 10):
            for j in range(1, 6):
                img = image.imread(
                    PATH + '/test_numbers\%d_%d.png' % (i, j))[:, :, 0]
                #img[img > 254] = 0
                img = np.abs(
                    ((img - np.min(img)) / (np.max(img) - np.min(img)))-1)
                img = np.array(Image.fromarray(img).resize((28, 28)))
                img[img < 0.1] = 0
                val_x.append(img*255)
                val_y.append(i)
        val_x = np.asanyarray(val_x)
        val_y = np.asanyarray(val_y)
        return (train_x, train_y), (test_x, test_y), (val_x, val_y)

    else:
        return (train_x, train_y), (test_x, test_y)

def plot_image(data, idx):
    image = np.asarray(data[idx]).squeeze()
    plt.imshow(image, cmap='bone')
    plt.show()

'''
_,_,vl = dataset(True)
plot_image(vl[0],1)
'''