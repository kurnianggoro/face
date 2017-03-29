import numpy as np
np.random.seed(1664)

from os import listdir
import sys
import pandas as pd

from keras.preprocessing.image import img_to_array, load_img

DATASET_PATH = 'dataset/MTFL/'
HAAR_FOLDER = '/home/username/opencv/opencv-master/data/haarcascades/'
HAAR_FILE = 'haarcascade_frontalface_default.xml'

def loadTrain(filename):
    data = pd.read_csv(filename, sep=' ', header=None, low_memory=False)
    files = data[0]  # get the first column (Id)

    feat = np.genfromtxt(filename, delimiter=' ')
    appearances = feat[:, len(feat[0])-4:len(feat[0])]
    feat = feat[:, 1:len(feat[0])-4]  # get the second until last column


    return files, np.asarray(feat), np.asarray(appearances)

def resize_img(img, max_dim=40):
    return img.resize((int(max_dim), int(max_dim)))

def loadImages(path,files, max_dim=40):
    print('loading '+str(len(files))+' images ...')
    import matplotlib.pyplot as plt
    X = np.empty((len(files), max_dim, max_dim, 1))

    for i in range(len(files)):
        filename = path+files[i]
        x = resize_img(load_img(filename, grayscale=True), max_dim=max_dim)
        x = img_to_array(x)

        X[i] = x
    X = np.asarray(X,dtype=np.float)/255.0
    return X

def isInside(px,py,w,h,xp,yp):
    inside = True
    for i in range(len(xp)):
        if xp[i] < px or xp[i] > px+w or yp[i] < py or yp[i] > py+h:
            inside = False
    return inside

import os
import cv2
def face_detect(path, files, feat, labels, feat_file, sz=40):
    haar_folder = HAAR_FOLDER
    face_cascade = cv2.CascadeClassifier(haar_folder + HAAR_FILE)

    save_loc = '/faces/'
    save_dir = path+save_loc
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file = open(save_dir+feat_file, 'w')

    # detecting faces
    for i in range(len(files)):
        print("%s %i/%i" %(files[i], i, len(files)))
        filename = path + files[i]
        img =  cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (px,py,w,h) in faces:
            # print(px,py,w,h)
            roi_color = img[py:py+h, px:px+w]
            # cv2.rectangle(img, (px, py), (px + w, py + h), (255, 0, 0), 2)
            # for j in range(5):
            #     cv2.circle(img,(int(round(feat[i][j])), int(round(feat[i][j+5]))),3,(0,0,255))
            if(isInside(px,py,w,h,feat[i][0:5],feat[i][5:10])):
                for j in range(5):
                    feat[i][j] -= px
                    feat[i][j+5] -= py
                    # cv2.circle(roi_color,(int(round(feat[i][j])), int(round(feat[i][j+5]))),3,(0,0,255))

                roi_color = cv2.resize(roi_color, (sz, sz))
                feat[i] = feat[i] * sz / w
                cv2.imwrite(save_dir+'/'+files[i],roi_color)

                file.write(save_loc+files[i])
                for j in range(10):
                    file.write(" %f" % feat[i][j])
                for j in range(4):
                    file.write(" %i" % labels[i][j])
                file.write("\n")

                # for j in range(5):
                #     cv2.circle(roi_color,(int(round(feat[i][j])), int(round(feat[i][j+5]))),3,(0,0,255))
                # cv2.imwrite("img%i.jpg" % i, roi_color)
    return feat

import shutil
def pre_process(filename, save_folder, path = DATASET_PATH, max_dim = 40):
    if not os.path.exists(path+save_folder):
        os.makedirs(path+save_folder)
    else:
        shutil.rmtree(path+save_folder)
        os.makedirs(path+save_folder)

    file_testIdx = path + filename
    print(file_testIdx)
    files, feat, labels = loadTrain(file_testIdx)
    feat = face_detect(path, files, feat, labels, filename, max_dim)
    return 0


def load(path=DATASET_PATH, max_dim=40):
    if not os.path.exists(path + 'faces'):
        os.makedirs(path + 'faces')

    pre_process('training_lfw.txt', 'faces/lfw_5590', max_dim=max_dim)
    pre_process('training_net.txt', 'faces/net_7876', max_dim=max_dim)
    pre_process('testing.txt', 'faces/AFLW', max_dim=max_dim)

    filenames = [path+'/faces/training_lfw.txt', path+'/faces/training_net.txt']
    with open(path +'/faces/' +'training.txt', 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)

    file_trainIdx = path +'/faces/' +'training.txt'
    file_testIdx = path +'/faces/' +'testing.txt'

    # file_trainIdx = path + 'training_lfw.txt'
    # file_testIdx = path + 'testing.txt'

    files, feat, label = loadTrain(file_trainIdx)
    imgs = loadImages(path, files, max_dim)

    files_test, feat_test, label_test = loadTrain(file_testIdx)
    imgs_test = loadImages(path, files_test, max_dim)


    # import matplotlib.pyplot as plt
    # plt.imshow(np.uint8(imgs[0].reshape(max_dim,max_dim)*255.0),cmap='gray')
    # plt.show()


    return imgs, feat, files , imgs_test, feat_test, files_test
