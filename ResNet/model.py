import keras
from keras.models import Model
from keras.layers import Conv2D,MaxPool2D,Flatten
from  keras.layers import BatchNormalization
from  keras.layers import Activation


def con_bn_relu(model,x):
    model = Conv2D(model)(x)
    model = BatchNormalization()(model)