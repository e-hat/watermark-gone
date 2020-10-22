from stl10_input import download_and_extract, read_all_images, read_single_image
from watermark import add_watermark
import numpy as np
import sys
from sys import stdout
import os

train_filename = 'watermark_train.npy'
test_filename = 'watermark_test.npy'

def write_watermark_dataset(dataset, filename):
    stdout.flush()
    watermarked_imgs = np.zeros(shape=dataset.shape, dtype='uint8')
    print(dataset.shape[0])
    for i in range(dataset.shape[0]):
        watermarked_imgs[i] = add_watermark(dataset[i])
        stdout.write("\rWriting watermarks to '{0}'. {1:5.2f}% done.".format(filename, 
                                                                        i / dataset.shape[0] * 100))
        stdout.flush()
    
    print(" Done writing.")
    print("Watermarked images shape:", watermarked_imgs.shape)
    write_to_file(filename, watermarked_imgs)


def read_n_images(n, path):
    f = open(path, "r")
    first_img = read_single_image(f)
    result = np.zeros(shape=(n,) + first_img.shape, dtype='uint8')
    for i in range(n):
        if i == 0:
            result[i] = first_img
        else:
            result[i] = read_single_image(f)

    return result

def write_to_file(filename, dataset):
    np.save(os.path.join('./data/', filename), dataset)


def get_data(num_train, num_test):
    y_train = read_n_images(num_train, './data/stl10_binary/train_X.bin')
    y_test = read_n_images(num_test, './data/stl10_binary/test_X.bin')

    x_train = np.load(os.path.join('./data/', train_filename))
    x_test = np.load(os.path.join('./data/', test_filename))

    return (x_train, y_train), (x_test, y_test)

if __name__ == "__main__":
    if len(sys.argv) == 3:
        download_and_extract()
        #y_train = read_all_images('./data/stl10_binary/train_X.bin')
        #y_test = read_all_images('./data/stl10_binary/test_X.bin')
        # print(y_test.shape) = (8000, 96, 96, 3)

        y_test = read_n_images(int(sys.argv[2]), './data/stl10_binary/test_X.bin')
        y_train = read_n_images(int(sys.argv[1]), './data/stl10_binary/train_X.bin')
   
        write_watermark_dataset(y_test, test_filename)
        write_watermark_dataset(y_train, train_filename)
    else:
        print("error: expected 2 arguments denoting num images to open for train and test, respectively")
