from vgg import *
import skimage
import skimage.io
import skimage.transform
import numpy as np


def load_image_tf(path):
  # load image
  img = skimage.io.imread(path)
  img = img/ 255.0
  assert (0 <= img).all() and (img <= 1.0).all()
  short_edge = min(img.shape[:2])
  yy = int((img.shape[0] - short_edge) / 2)
  xx = int((img.shape[1] - short_edge) / 2)
  crop_img = img[yy : yy + short_edge, xx : xx + short_edge]
  # resize to 224, 224
  resized_img = skimage.transform.resize(crop_img, (224, 224))
  resized_img=resized_img.reshape((1, 224, 224, 3))
  resized_img*=255
  resized_img[0][0]-=103.939
  resized_img[0][1]-=116.779
  resized_img[0][2]-=123.68
  return resized_img


def load_image_th(path):
  # load image
  img = skimage.io.imread(path)
  img = img/ 255.0
  assert (0 <= img).all() and (img <= 1.0).all()
  short_edge = min(img.shape[:2])
  yy = int((img.shape[0] - short_edge) / 2)
  xx = int((img.shape[1] - short_edge) / 2)
  crop_img = img[yy : yy + short_edge, xx : xx + short_edge]
  # resize to 224, 224
  resized_img = skimage.transform.resize(crop_img, (224, 224))
  resized_img*=255
  resized_img[0][0]-=103.939
  resized_img[0][1]-=116.779
  resized_img[0][2]-=123.68
  resized_img=resized_img.reshape((1, 3, 224, 224))
  return resized_img



def load_model_tf():
  model = VGG_16_tf('vgg16_weights_tf_dim_ordering_tf_kernels.h5')
  sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
  model.compile(optimizer=sgd, loss='categorical_crossentropy')
  return model;


def load_model_th():
  model = VGG_16_th('vgg16_weights.h5')
  sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
  model.compile(optimizer=sgd, loss='categorical_crossentropy')
  return model;


def predict_tf(model,path):
  img=load_image_tf(path)
  res=model.predict(img)
  f = open('synset.txt','r')
  lines = f.readlines()
  f.close()
  for i in range(10):
  	pre = np.argmax(res)
  	print(lines[pre-1])
  	res[0][pre]=0


def predict_th(model,path):
  img=load_image_th(path)
  res=model.predict(img)
  f = open('synset.txt','r')
  lines = f.readlines()
  f.close()
  for i in range(10):
  	pre = np.argmax(res)
  	print(lines[pre-1])
  	res[0][pre]=0

