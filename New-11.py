'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''
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

# dimensions of our images.
img_width, img_height = 150, 150

top_model_weights_path = 'D:/Corn/program/matser/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'#/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
train_data_dir = 'D:/Corn/program/matser/data/train'#'D:/Corn/corn/corn/double'
validation_data_dir = 'D:/Corn/program/matser/data/validation'#'D:/Corn/corn/corn/single'
nb_train_samples = 32
nb_validation_samples = 12
epochs = 50
batch_size = 4  # 12//4        -.-||



test_datagen = ImageDataGenerator(rescale=1./255)



def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network

    model = applications.VGG16(include_top=False, weights='imagenet')
    
    model.summary()
    
    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    
   

    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    
    np.save(open('bottleneck_features_train.npy', 'wb'),
            bottleneck_features_train)
    
 

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save(open('bottleneck_features_validation.npy', 'wb'),
            bottleneck_features_validation)


def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy','rb'))#  rb  #Dingdong
    train_labels = np.array(
        [0] * (nb_train_samples // 2) + [1] * (nb_train_samples // 2))  #//
  
    validation_data = np.load(open('bottleneck_features_validation.npy','rb'))
    validation_labels = np.array(
        [0] * (nb_validation_samples // 2) + [1] * (nb_validation_samples // 2))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:])) 
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    
    
    model.save_weights(top_model_weights_path, overwrite=True)
    
   

save_bottlebeck_features()
train_top_model()

print('Here!')

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

#%%
image_path = 'D:/Corn/program/matser/double(28).jpg'
img = image.load_img(image_path,target_size=(150,150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

result1 = model.predict(x)

print(result1)
'''
features = vgg16.predict(x)
result = decode_predictions(features,top=5)
print(result)
'''






'''
import h5py

weights_path = top_model_weights_path

f = h5py.File(weights_path)
'''

'''
for k in range(f.attrs['nb_layers']):
    if k >= len(model.layers):
        break
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    model.layers[k].set_weights(weights)
f.close()
print('Model loaded.')
'''





print('I am here')
'''
#--------------------creating the composition of the two------------------
base_model = applications.VGG16(weighs="imagenet", include_top=False,input_shape=(img_width,img_height,3))
layers.trainable = False

save_bottlebeck_features()
train_top_model()

model = Model(inputs=base_model.input, outputs=top_model(base_model.outputs))
model.compile(loss = "binary_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
'''



'''
from tensorflow.python.keras.models import load_model  
model3 = load_model(top_model_weights_path) 
print(predict)
'''


# %%
