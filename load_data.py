from stl10_input import download_and_extract, read_all_images
from watermark import add_watermark
import numpy as np
import sys

WIDTH, HEIGHT = 96, 96

def watermark_dataset(dataset):
    watermarked_imgs = np.zeros(shape=dataset.shape, dtype='uint8')
    for i in range(dataset.shape[0]):
        watermarked_imgs[i] = add_watermark(dataset[i])
    
    return watermarked_imgs

def get_data():
    y_train = read_all_images('./data/stl10_binary/train_X.bin')
    x_train = watermark_dataset(y_train)

    y_test = read_all_images('./data/stl10_binary/test_X.bin')
    x_test = watermark_dataset(y_test)

    return (x_train, y_train), (x_test, y_test)

if __name__ == "__main__":
    download_and_extract()