from keras.layers import merge
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import Dense, Activation, Flatten
from keras.layers import BatchNormalization
from keras.models import Model
from keras.layers import Input
from keras.preprocessing.image import load_img, img_to_array
import keras.backend as K
import numpy as np

def identity_block(input_tensor, kernel_size, filters, stage, block):
    dim_ordering = K.image_dim_ordering()
    nb_filter1, nb_filter2, nb_filter3 = filters
    if dim_ordering == 'tf':
        axis = 3
    else:
        axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    out = Convolution2D(nb_filter1, 1, 1, dim_ordering=dim_ordering, name=conv_name_base + '2a')(input_tensor)
    out = BatchNormalization(axis=axis, name=bn_name_base + '2a')(out)
    out = Activation('relu')(out)

    out = out = Convolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same',
                              dim_ordering=dim_ordering, name=conv_name_base + '2b')(out)
    out = BatchNormalization(axis=axis, name=bn_name_base + '2b')(out)
    out = Activation('relu')(out)

    out = Convolution2D(nb_filter3, 1, 1, dim_ordering=dim_ordering, name=conv_name_base + '2c')(out)
    out = BatchNormalization(axis=axis, name=bn_name_base + '2c')(out)

    out = merge([out, input_tensor], mode='sum')
    out = Activation('relu')(out)
    return out


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    nb_filter1, nb_filter2, nb_filter3 = filters
    dim_ordering = K.image_dim_ordering()
    if dim_ordering == 'tf':
        axis = 3
    else:
        axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    out = Convolution2D(nb_filter1, 1, 1, subsample=strides,
                        dim_ordering=dim_ordering, name=conv_name_base + '2a')(input_tensor)
    out = BatchNormalization(axis=axis, name=bn_name_base + '2a')(out)
    out = Activation('relu')(out)

    out = Convolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same',
                        dim_ordering=dim_ordering, name=conv_name_base + '2b')(out)
    out = BatchNormalization(axis=axis, name=bn_name_base + '2b')(out)
    out = Activation('relu')(out)

    out = Convolution2D(nb_filter3, 1, 1, dim_ordering=dim_ordering, name=conv_name_base + '2c')(out)
    out = BatchNormalization(axis=axis, name=bn_name_base + '2c')(out)

    shortcut = Convolution2D(nb_filter3, 1, 1, subsample=strides,
                             dim_ordering=dim_ordering, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=axis, name=bn_name_base + '1')(shortcut)

    out = merge([out, shortcut], mode='sum')
    out = Activation('relu')(out)
    return out


def read_img(img_path):
    dim_ordering = K.image_dim_ordering()
    mean = (103.939, 116.779, 123.68)
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img, dim_ordering=dim_ordering)

    # decenterize
    img[0, :, :] -= mean[0]
    img[1, :, :] -= mean[1]
    img[2, :, :] -= mean[2]

    # 'RGB'->'BGR'
    if dim_ordering == 'th':
        img = img[::-1, :, :]
    else:
        img = img[:, :, ::-1]

    # expand dim for test
    img = np.expand_dims(img, axis=0)
    return img


def get_resnet50():
    if K.image_dim_ordering() == 'tf':
        inp = Input(shape=(224, 224, 3))
        axis = 3
    else:
        inp = Input(shape=(3, 224, 224))
        axis = 1

    dim_ordering = K.image_dim_ordering()
    out = ZeroPadding2D((3, 3), dim_ordering=dim_ordering)(inp)
    out = Convolution2D(64, 7, 7, subsample=(2, 2), dim_ordering=dim_ordering, name='conv1')(out)
    out = BatchNormalization(axis=axis, name='bn_conv1')(out)
    out = Activation('relu')(out)
    out = MaxPooling2D((3, 3), strides=(2, 2), dim_ordering=dim_ordering)(out)

    out = conv_block(out, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    out = identity_block(out, 3, [64, 64, 256], stage=2, block='b')
    out = identity_block(out, 3, [64, 64, 256], stage=2, block='c')

    out = conv_block(out, 3, [128, 128, 512], stage=3, block='a')
    out = identity_block(out, 3, [128, 128, 512], stage=3, block='b')
    out = identity_block(out, 3, [128, 128, 512], stage=3, block='c')
    out = identity_block(out, 3, [128, 128, 512], stage=3, block='d')

    out = conv_block(out, 3, [256, 256, 1024], stage=4, block='a')
    out = identity_block(out, 3, [256, 256, 1024], stage=4, block='b')
    out = identity_block(out, 3, [256, 256, 1024], stage=4, block='c')
    out = identity_block(out, 3, [256, 256, 1024], stage=4, block='d')
    out = identity_block(out, 3, [256, 256, 1024], stage=4, block='e')
    out = identity_block(out, 3, [256, 256, 1024], stage=4, block='f')

    out = conv_block(out, 3, [512, 512, 2048], stage=5, block='a')
    out = identity_block(out, 3, [512, 512, 2048], stage=5, block='b')
    out = identity_block(out, 3, [512, 512, 2048], stage=5, block='c')

    out = AveragePooling2D((7, 7), dim_ordering=dim_ordering)(out)
    out = Flatten()(out)
    out = Dense(1000, activation='softmax', name='fc1000')(out)

    model = Model(inp, out)

    return model


if __name__ == '__main__':
    K.set_image_dim_ordering('tf')
    resnet_model = get_resnet50()
    resnet_model.load_weights('resnet50_weights_tf_dim_ordering_tf_kernels.h5')
    class_table = open('synset.txt', 'r')
    lines = class_table.readlines()
    class_table.close()
    for i in range(1,7):
        img = read_img(str(i)+'.jpg')
        print ("prediction for"+str(i)+".jpg is")
        print (lines[np.argmax(resnet_model.predict(img)[0])])