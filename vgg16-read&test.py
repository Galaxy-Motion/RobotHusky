from PIL import Image
#import cv2.cv as cv 

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model, Model
from keras.layers import Dropout, Flatten, Dense
from keras import applications

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions

from keras.preprocessing import image


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

img_width, img_height = 150, 150
top_model_weights_path = 'D:/Corn/program/matser/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'#/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
train_data_dir = 'D:/Corn/program/matser/data/train'#'D:/Corn/corn/corn/double'
validation_data_dir = 'D:/Corn/program/matser/data/validation'#'D:/Corn/corn/corn/single'
nb_train_samples = 136
nb_validation_samples = 40
epochs = 50
batch_size = 4  # 12//4        -.-||


Vmodel = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))#
print('Model loaded.')

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=Vmodel.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base
# model.add(top_model)
model = Model(inputs=Vmodel.input, outputs=top_model(Vmodel.output))


image_path = 'D:/Corn/program/matser/single-1.jpg'
img = image.load_img(image_path,target_size=(150,150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

result1 = model.predict(x)

if result1 == [[1]]:
   print('单株')
else:
   print('双株')


file = r'D:/Corn/program/matser/single-1.jpg'
im = Image.open(file)
im.show()


print(result1)
'''
#----------------------------------
#从Opencv中导入函数





#创建一个窗口，命名为you need tostruggle,

#cv.CV_WINDOW_AUTOSIZE这个参数设定显示窗口虽图片大小自动变化

cv.NamedWindow('You need to struggle', cv.CV_WINDOW_AUTOSIZE)



#加载一张图片，第二个参数指定当图片被加载后的格式，还有另外两个可选参数

#CV_LOAD_IMAGE_GREYSCALE and CV_LOAD_IMAGE_UNCHANGED，分别是灰度格式和不变格式

image=cv.LoadImage('D:/Corn/program/matser/single-1.jpg', cv.CV_LOAD_IMAGE_COLOR)


#创建一个矩形，来让我们在图片上写文字，参数依次定义了文字类型，高，宽，字体厚度等。。

font=cv.InitFont(cv.CV_FONT_HERSHEY_SCRIPT_SIMPLEX, 1, 1, 0, 3, 8)



#将文字框加入到图片中，(5,20)定义了文字框左顶点在窗口中的位置，最后参数定义文字颜色


cv.PutText(image, "Hello World", (30,30), font, (0,255,0))

#在刚才创建的窗口中显示图片

cv.ShowImage('You need to struggle', image)
cv.WaitKey(0)

#保存图片
cv.SaveImage('G://feel.png', image)
'''