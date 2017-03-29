import dataset

import numpy as np
import tensorflow as tf
# np.random.seed(1664)
# tf.set_random_seed(1664)

x,y, files , x_te, y_te, f_te = dataset.load()

scale = 40.0
scale_te = 40.0

y = y/scale
y_te = y_te/scale_te

filename = 'trained.h5'

from keras.utils.np_utils import normalize

from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten, Input, merge, Lambda
from keras.layers.local import LocallyConnected2D

from keras import backend as K
def absTan(args):
    return K.abs(K.tanh(args))
def absVal(args):
    return K.abs(args)

from keras.layers.normalization import BatchNormalization
def createModel(sz=40):

    image = Input(shape=(sz,sz,1), name='image')

    x = LocallyConnected2D(16, 5, 5, input_shape=(sz, sz, 1))(image)
    # x = BatchNormalization()(x)
    x = Lambda(absTan)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = LocallyConnected2D(48, 3, 3)(x)
    # x = (Convolution2D(48, 3, 3))(x)
    # x = BatchNormalization()(x)
    x = Lambda(absTan)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = LocallyConnected2D(64, 3, 3)(x)
    # x = (Convolution2D(64, 3, 3))(x)
    # x = BatchNormalization()(x)
    x = Lambda(absTan)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = LocallyConnected2D(64, 2, 2)(x)
    # x = (Convolution2D(64, 2, 2))(x)
    # x = BatchNormalization()(x)
    x = Lambda(absTan)(x)

    x = Flatten()(x)
    x = Dense(100, name='fc')(x)
    # x = BatchNormalization()(x)
    x = Lambda(absTan)(x)

    pos = Dense(10)(x)
    # x = BatchNormalization()(x)
    pos = Lambda(absTan)(pos)

    model = Model(input=image, output=pos)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


net = createModel()
net.fit(x,y,verbose=1,batch_size=32,nb_epoch=100,
        # validation_split=0.1,
        validation_data=(x_te,y_te)
        )
net.save(filename)

def saveResult(net, X, Y, scale):
    file = open('res.csv', 'w')
    pred = net.predict(X)

    sz = pred.shape
    for i in range (sz[0]):
        for j in range (sz[1]):
            file.write("%f,"%(pred[i][j]*scale))
        file.write("\n")
        for j in range (sz[1]):
            file.write("%f,"%(Y[i][j]*scale))
        file.write("\n")
    file.close()

from keras.models import load_model
net = load_model(filename)

saveResult(net,x,y, scale=scale)

pred = net.predict(x)
pred = pred*scale

y=y*scale



import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array, load_img
DATASET_PATH = '/media/kurnianggoro/data/face/dataset/MTFL/'
idx = 500
print(pred[idx])
print(y[idx])
img = load_img(DATASET_PATH+files[idx])
img = np.asarray(img)

area = np.pi * (3) ** 2


import cv2
haar_folder = '/home/kurnianggoro/opencv/opencv-master/data/haarcascades/'
face_cascade = cv2.CascadeClassifier(haar_folder+'haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2GRAY)
print(gray.shape)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (px,py,w,h) in faces:
    print(px,py,w,h)
    cv2.rectangle(img,(px,py),(px+w,py+h),(255,0,0),2)
    roi_color = img[py:py+h, px:px+w]

plt.imshow(img)
plt.scatter(np.array(y[idx])[0:5], np.array(y[idx])[5:10],s =area,c='b')
plt.scatter(np.array(pred[idx])[0:5], np.array(pred[idx])[5:10],s =area,c='r')
plt.show()
plt.close()



def save_images_results(x,y,files,scale,save_folder):
    pred = net.predict(x)
    pred = pred * scale
    y = y * scale

    import os
    import shutil
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    else:
        shutil.rmtree(save_folder)
        os.makedirs(save_folder)

    for idx in range(len(y)):
        print('saving '+save_folder+ "/%04d.png" % idx)
        plt.clf()
        img = load_img(DATASET_PATH + files[idx])
        plt.imshow(img)
        plt.scatter(np.array(y[idx])[0:5], np.array(y[idx])[5:10], s=area, c='b')
        plt.scatter(np.array(pred[idx])[0:5], np.array(pred[idx])[5:10], s=area, c='r')
        plt.savefig(save_folder+"/%04d.png" % idx)


save_folder = 'results_net'
save_images_results(x_te,y_te,f_te,scale_te,'results_net')