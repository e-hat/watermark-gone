import tensorflow as tf
from tensorflow.keras import Model, layers
from load_data import get_data
import matplotlib.pyplot as plt
import numpy as np
import time

def normalize(data, min, max, new_min, new_max):
    return new_min + (new_max - new_min) / float((max - min)) * (data - min)

BATCH_SIZE = 10
TRAIN_BUFFER_SIZE = 5000
TEST_BUFFER_SIZE = 8000

(x_train, y_train), (x_test, y_test) = get_data(TRAIN_BUFFER_SIZE, TEST_BUFFER_SIZE)

x_train = normalize(x_train.astype('float32'), 0, 255, 0, 1) 
x_test = normalize(x_test.astype('float32'), 0, 255, 0, 1)
y_train = normalize(y_train.astype('float32'), 0, 255, -1, 1)
y_test = normalize(y_test.astype('float32'), 0, 255, -1, 1)

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(TRAIN_BUFFER_SIZE).batch(BATCH_SIZE)
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(TEST_BUFFER_SIZE).batch(BATCH_SIZE)

code_size = 256
img_width, img_height, img_depth = 96, 96, 3

epochs = 100

display_interval = 1
num_display_pics = 2

def autoencoder():
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(64, kernel_size=(5,5), strides=(2,2), 
                            padding='same', 
                            input_shape=(img_height, img_width, img_depth)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(128, kernel_size=(5,5), strides=(2,2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(256, kernel_size=(5,5), strides=(2,2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Flatten())
    model.add(layers.Dense(code_size))

    model.add(layers.Dense(6 * 6 * 256, use_bias=False, input_shape=(code_size,)))
    model.add(layers.Reshape((6, 6, 256)))
    
    # CONVTRANSPOSE 1
    model.add(layers.Conv2DTranspose(128, kernel_size=3, strides=1, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    # CONVTRANSPOSE 2
    model.add(layers.Conv2DTranspose(64, kernel_size=5, strides=4, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2DTranspose(3, kernel_size=5, strides=4, padding='same', 
                                     use_bias=False, activation='tanh'))

    return model


def mse_loss(y_pred, y_true):
    mse = tf.keras.losses.MeanSquaredError()
    return mse(y_pred, y_true)

ae = autoencoder()

optimizer = tf.keras.optimizers.Adam(1e-4)

@tf.function
def train_step(batch):

    x_imgs, y_imgs = batch

    with tf.GradientTape() as tape:
        y_pred = ae(x_imgs, training=True)

        loss = mse_loss(y_pred, y_imgs)

    grad = tape.gradient(loss, ae.trainable_variables)

    optimizer.apply_gradients(zip(grad, ae.trainable_variables))


def train():
    for epoch in range(epochs):
        start_time = time.time()
        for batch in train_data:
            train_step(batch)
        
        print('Time for epoch {} is {} sec.'.format(epoch + 1, time.time() - start_time))

        print('Testing...')
        total_loss = 0
        for batch in test_data:
            total_loss += mse_loss(ae(batch[0], training=False), batch[1])
        print('Avg. loss: {}'.format(total_loss / TEST_BUFFER_SIZE))

        if (epoch + 1) % display_interval == 0:
            generate_and_save_images(epoch)


def generate_and_save_images(epoch):
    predictions = ae(x_test[:num_display_pics], training=False)
    predictions = np.array(normalize(predictions, -1, 1, 0, 255))

    plt.close()

    fig, axes = plt.subplots(nrows=num_display_pics, ncols=2)
    axes[0,0].set_title('Before')
    axes[0,1].set_title('After')

    def remove_axis_labels(axis):
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)

    for i, (left, right) in enumerate(axes):
        left.imshow(normalize(x_test[i], 0, 1, 0, 255).astype(np.uint8))
        right.imshow(predictions[i].astype(np.uint8))

        remove_axis_labels(left)
        remove_axis_labels(right)


    plt.pause(0.01)
    plt.savefig('progress_images/images_at_epoch_{:04d}.png'.format(epoch))
    plt.ion()
    plt.show()


train()
