import tensorflow as tf
from tensorflow.keras.utils import to_categorical


def load_MNIST(one_hot_encoding=True, num_classes=10, load_validation=True, num_validation=5000):
    train_sample, test_sample = tf.keras.datasets.mnist.load_data()
    (train_x,train_y), (test_x,test_y) = train_sample, test_sample
    if load_validation==False:
        if one_hot_encoding==True:
            train_y = to_categorical(train_y,num_classes=num_classes)
            test_y = to_categorical(test_y,num_classes=num_classes)
        return (train_x,train_y), (test_x,test_y)
    else:
        validation_x = train_x[-num_validation:]
        validation_y = train_y[-num_validation:]
        train_x = train_x[:-num_validation]
        train_y = train_y[:-num_validation]
        if one_hot_encoding==True:
            train_y = to_categorical(train_y,num_classes=num_classes)
            test_y = to_categorical(test_y,num_classes=num_classes)
            validation_y = to_categorical(validation_y,num_classes=num_classes)
        return (train_x,train_y), (test_x,test_y), (validation_x, validation_y)