from load_data import get_data
import matplotlib.pyplot as plt
# from model import autoencoder

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = get_data()
    
    print(x_test.shape)
    print(y_test.shape)