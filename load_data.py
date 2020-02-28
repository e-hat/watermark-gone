from stl10_input import download_and_extract, read_all_images
from watermark import add_watermark
import numpy as np
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



def write_to_file(filename, dataset):
    np.save(os.path.join('./data/', filename), dataset)


def get_data():
    y_train = read_all_images('./data/stl10_binary/train_X.bin')
    y_test = read_all_images('./data/stl10_binary/test_X.bin')

    x_train = np.load(os.path.join('./data/', train_filename))
    x_test = np.load(os.path.join('./data/', test_filename))

    return (x_train, y_train), (x_test, y_test)

if __name__ == "__main__":
    download_and_extract()
    y_train = read_all_images('./data/stl10_binary/train_X.bin')
    y_test = read_all_images('./data/stl10_binary/test_X.bin')
    # print(y_test.shape) = (8000, 96, 96, 3)
    write_watermark_dataset(y_test, test_filename)
    write_watermark_dataset(y_train, train_filename)