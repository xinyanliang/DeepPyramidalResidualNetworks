import keras
from keras.models import Model
from keras.layers import Dense,Conv2D, MaxPool2D, Flatten, Input, GlobalAvgPool2D
from  keras.layers import BatchNormalization
from  keras.layers import Activation
from keras.datasets import cifar10
from keras.datasets import cifar100
# refer to https://github.com/raghakot/keras-resnet
def head(inx,width, height):
    x = Input(shape=(width, height, 3))(inx)
    x = conv_bn_relu(x, filters=64, kernel_size=(7, 7),strides=(2,2), padding='same')
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2)(x))

def conv_bn(inx, filters, kernel_size, padding='valid'):
    x = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(inx)
    x = BatchNormalization()(x)
    return x


def conv_bn_relu(inx, filters, kernel_size, padding='valid'):
    x = conv_bn(inx=inx, filters=filters, kernel_size=kernel_size, padding=padding)
    x = Activation(name='relu')(x)
    return x

def resiblocl_botton(inx, filters):
    filters1, filters2, filters3 = filters
    x = conv_bn_relu(inx, filters1, (1, 1))
    x = conv_bn_relu(x, filters2, padding='same')
    x = conv_bn_relu(x, filters3, (1, 1))
    return x

def group(inx, filters, count):
    x = inx
    for i in  range(count):
        x = resiblocl_botton(inx, filters)
    MaxPool2D(pool_size=(2,2))(x)
    return x

def ResNet50(inx,width, height,num_class):
    input = head(inx,width, height)
    filters2 = [64,64,256]
    filters3 = [128,128,512]
    filters4 = [256,256,1024]
    filters5 = [512,512,2048]
    x = group(input, filters2, 3)
    x = group(x, filters3, 4)
    x = group(x, filters4, 6)
    x = group(x, filters5, 3)
    x = GlobalAvgPool2D()(x)
    out = Dense(units=num_class)(x)
    model = Model(inputs=input, outputs=out)
    return model

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print(x_train.shape,y_train.shape)

