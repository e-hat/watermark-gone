from load_data import get_data
import matplotlib.pyplot as plt

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = get_data()
    plt.imshow(x_train[0])
    plt.show(block=True)

    plt.imshow(y_train[0])
    plt.show(block=True)