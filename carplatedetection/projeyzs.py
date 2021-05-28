from PyQt5.QtWidgets import *
from PyQt5.uic import *
from PyQt5 import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QApplication, QFileDialog
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout,QDesktopWidget, QWidget,QTableWidget,QTableView,QTableWidgetItem,QHeaderView,QGraphicsScene,QGraphicsPixmapItem,QFileDialog
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import cv2
import numpy as np
import math
from PyQt5.QtCore import pyqtSlot
import pandas as pd
import urllib
import matplotlib.pyplot as plt
import pathlib
import glob
import os
import time
from PIL import Image
from PyQt5.QtGui import QIcon, QPixmap
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Input, Dropout
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from tensorflow.keras.layers import (Conv2D, MaxPool2D, BatchNormalization, InputLayer, LeakyReLU, Dense, 
Flatten, Dropout, ReLU, SeparableConv2D, AveragePooling2D, GlobalAveragePooling2D)
# from tensorflow_addons.metrics import F1Score
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import tensorflow.keras as keras
# import torch
from PIL import Image
import argparse
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import functools
# import cnnCharacterPre
from keras.models import Model, load_model, Sequential
import pickle 
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
from keras.layers import Dense, Flatten, Input, Conv2D, MaxPooling2D, Dropout
from keras.models import Model, load_model, Sequential
from keras.utils import Sequence, plot_model
from keras.utils import to_categorical
from PIL import Image
import pytesseract


class window(QMainWindow):
    
    def __init__(self):
        super(window, self).__init__()
        loadUi("arayuz.ui", self)
        self.Btn_selectFile.clicked.connect(self.BtnSelectImage)
        self.Btn_modelTrain.clicked.connect(self.cnn)
        self.Btn_modelTrain_2.clicked.connect(self.vgg)
        self.plakatespit.clicked.connect(self.plakatespiti)
        self.printlisance.clicked.connect(self.yaz)
        self.gbsec.setStyleSheet("QGroupBox { border: 1px solid red;}")
        self.gbegit.setStyleSheet("QGroupBox { border: 1px solid red;}")
        self.gbacc.setStyleSheet("QGroupBox { border: 1px solid red;}")
        self.gbtespit.setStyleSheet("QGroupBox { border: 1px solid red;}")
        self.gbyaz.setStyleSheet("QGroupBox { border: 1px solid red;}")
        self.gbcikti.setStyleSheet("QGroupBox { border: 1px solid red;}")
        

        
    def BtnSelectImage(self):
        fi_name = QFileDialog.getOpenFileName(self, 'Open file',
                                            'D:\Windows10\Desktop\\', "Image files (*.jpg *.png)")
        #img = cv2.imread(fi_name[0])
        #img = cv2.resize(img, dsize=(650, 520))
        #height, width = img.shape[:2]
        #img = QtGui.QImage(img, height, width, QtGui.QImage.Format_RGB888)
        #img = QtGui.QPixmap.fromImage(img)      
        img = QPixmap(fi_name[0])
        img = img.scaled(251, 171, Qt.KeepAspectRatio)
        self.label.setPixmap(img)
        self.img=cv2.imread(fi_name[0])
        self.gbsec.setStyleSheet("QGroupBox { border: 1px solid green;}")
        self.gbegit.setStyleSheet("QGroupBox { border: 1px solid yellow;}")
        self.gbtespit.setStyleSheet("QGroupBox { border: 1px solid yellow;}")
    
        
    def resizeannotation(self ,f):
        from lxml import etree
        tree = etree.parse(f)
        IMAGE_SIZE = 200
        for dim in tree.xpath("size"):
            width = int(dim.xpath("width")[0].text)
            height = int(dim.xpath("height")[0].text)
        for dim in tree.xpath("object/bndbox"):
            xmin = int(dim.xpath("xmin")[0].text)/(width/IMAGE_SIZE)
            ymin = int(dim.xpath("ymin")[0].text)/(height/IMAGE_SIZE)
            xmax = int(dim.xpath("xmax")[0].text)/(width/IMAGE_SIZE)
            ymax = int(dim.xpath("ymax")[0].text)/(height/IMAGE_SIZE)
        return [int(xmax), int(ymax), int(xmin), int(ymin)]    
   


    
    def plot_scores(self ,train) :
        accuracy = train.history['accuracy']
        val_accuracy = train.history['val_accuracy']
        epochs = range(len(accuracy))
        plt.plot(epochs, accuracy, 'b', label='Score apprentissage')
        plt.plot(epochs, val_accuracy, 'r', label='Score validation')
        plt.title('Scores')
        plt.legend()
        plt.show()
        
        
    def cnn(self):  
            print("başladı")
            import pandas as pd
            import numpy as np
            from matplotlib import pyplot as plt
            import seaborn as sns
            import cv2
            import os
            import glob
            import os
            IMAGE_SIZE = 224        
            img_dir = "C:/Users/erent/Desktop/carplatedetection/images"  
            data_path = os.path.join(img_dir,'*g')
            files = glob.glob(data_path)
            files.sort() 
            X=[]
            for f1 in files:
                img = cv2.imread(f1)
                img = cv2.resize(img, (IMAGE_SIZE,IMAGE_SIZE))
                X.append(np.array(img))
            path = 'C:/Users/erent/Desktop/carplatedetection/annotations'
            text_files = ['C:/Users/erent/Desktop/carplatedetection/annotations/'+f for f in sorted(os.listdir(path))]
            y=[]
            for i in text_files:
                y.append(self.resizeannotation(i))     
            image = cv2.rectangle(X[0],(y[0][0],y[0][1]),(y[0][2],y[0][3]),(0, 0, 255))
            image = cv2.rectangle(X[1],(y[1][0],y[1][1]),(y[1][2],y[1][3]),(0, 0, 255))
            X=np.array(X)
            y=np.array(y)
            X = X / 255
            y = y / 255
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
            from keras.models import Sequential
            from keras.layers import Dense, Flatten
            from keras.applications.vgg16 import VGG16        
            model = Sequential()
            model.add(VGG16(weights="imagenet", include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
            model.add(Flatten())
            model.add(Dense(128, activation="relu"))
            model.add(Dense(128, activation="relu"))
            model.add(Dense(64, activation="relu"))
            model.add(Dense(4, activation="sigmoid")) 
            model.layers[-6].trainable = False
            model.summary()
            model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
            train = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, verbose=1)
            model.save('my_modelcnn', overwrite=True)
            scores = model.evaluate(X_test, y_test, verbose=0)
            print("Score : %.2f%%" % (scores[1]*100))
            scoreall= "Score : %.2f%%" % (scores[1]*100)
            self.plot_scores(train)
            self.modelacclosstb.setText(str(scores))
            self.modelacctb.setText(str(scoreall))
            y_cnn = model.predict(X_test)
            # plt.figure(figsize=(20,40))
            self.gbegit.setStyleSheet("QGroupBox { border: 1px solid green;}")
            self.gbacc.setStyleSheet("QGroupBox { border: 1px solid green;}")
            self.grafaccloss(train)
            
            
    def vgg(self):   
        print("VGG başladı")
        import pandas as pd
        import numpy as np
        from matplotlib import pyplot as plt
        import seaborn as sns
        import cv2
        import os
        import glob
        import os
        IMAGE_SIZE = 224
        img_dir = "C:/Users/erent/Desktop/carplatedetection/images" 
        data_path = os.path.join(img_dir,'*g')
        files = glob.glob(data_path)
        files.sort()
        X=[]
        for f1 in files:
            img = cv2.imread(f1)
            img = cv2.resize(img, (IMAGE_SIZE,IMAGE_SIZE))
            X.append(np.array(img))    
        from lxml import etree
        path = 'C:/Users/erent/Desktop/carplatedetection/annotations'
        text_files = ['C:/Users/erent/Desktop/carplatedetection/annotations/'+f for f in sorted(os.listdir(path))]
        y=[]
        for i in text_files:
                y.append(self.resizeannotation(i))  
        plt.figure(figsize=(10,20))
        for i in range(0,17) :
            plt.subplot(10,5,i+1)
            plt.axis('off')
            plt.imshow(X[i])        
        image = cv2.rectangle(X[0],(y[0][0],y[0][1]),(y[0][2],y[0][3]),(0, 0, 255))
        image = cv2.rectangle(X[1],(y[1][0],y[1][1]),(y[1][2],y[1][3]),(0, 0, 255))
        X=np.array(X)
        y=np.array(y)
        X = X / 255
        y = y / 255
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1)
        from keras.models import Sequential      
        from keras.layers import Dense, Flatten      
        from keras.applications.vgg16 import VGG16
        model = Sequential()
        model.add(VGG16(weights="imagenet", include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(4, activation="sigmoid")) 
        model.layers[-6].trainable = False
        model.summary()
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        train = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, verbose=1)
        model.save('my_model', overwrite=True) 
        scores = model.evaluate(X_test, y_test, verbose=0)
        print("Score : %.2f%%" % (scores[1]*100))
        self.plot_scores(train)
        test_loss, test_accuracy = model.evaluate(X_test, y_test,steps=int(100))
        scoreall= "Score : %.2f%%" % (scores[1]*100)            
        self.modelacclosstb.setText(str(scores))
        self.modelacctb.setText(str(scoreall))
        print("Test results \n Loss:",test_loss,'\n Accuracy',test_accuracy)
        y_cnn = model.predict(X_test)
        # plt.figure(figsize=(20,40))
        self.gbegit.setStyleSheet("QGroupBox { border: 1px solid green;}")
        self.gbacc.setStyleSheet("QGroupBox { border: 1px solid green;}")
        self.grafaccloss(train)
       
    
    
    
    def grafaccloss(self,train):
        plt.figure(figsize=(10,2.5))
        plt.subplot(1, 2, 1)
        plt.plot(train.history['accuracy'])
        plt.plot(train.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig("acc.png")
        px = QPixmap("acc.png")
        self.accgraf.setPixmap(px)
        plt.show()
        plt.figure(figsize=(10,2.5))
        plt.subplot(1, 2, 1)
        plt.plot(train.history['loss'])
        plt.plot(train.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig("loss.png")
        pxl = QPixmap("loss.png")
        self.lossgraf.setPixmap(pxl)
        plt.show()
    
    
    def imagecrop(self):
        modelsel = self.modelselect.currentText()
        if( modelsel == "VGG16"):
            model = load_model("my_model")
        else:
            model = load_model("my_modelcnn")
            
        img = cv2.resize(self.img / 255.0, dsize=(224, 224))        
        y_hat = model.predict(img.reshape(1, 224, 224, 3)).reshape(-1) * 224    
        xt, yt = y_hat[0], y_hat[1]
        xb, yb = y_hat[2], y_hat[3]
        image1=cv2.rectangle(self.img, (int(xt*self.img.shape[1]/224), 
                                                              int(yt*self.img.shape[0]/224)), 
    (int(xb*self.img.shape[1]/224), int(yb*self.img.shape[0]/224)), (0, 0, 255), 2)       
        cropped = image1[int(yb*image1.shape[0]/224): int(yt*image1.shape[0]/224), int(xb*image1.shape[1]/224) : int(xt*image1.shape[1]/224)]    
        cv2.imwrite("./fullCrop.jpg", cropped)
        degis=cv2.imread("fullCrop.jpg")
        self.pixmapReady = QPixmap("fullCrop.jpg")
        self.label_kp.setPixmap(self.pixmapReady)
        cv2.imwrite("./crop.png", self.img)
        degis=cv2.imread("crop.png")
        imgPx=self.convert_nparray_to_QPixmap(degis)
        self.label_3.setPixmap(imgPx)
     
        

     
    
    
    
    def yaz(self):
        from PIL import Image
        import pytesseract
        im = Image.open("fullCrop.jpg")
        text = pytesseract.image_to_string(im,lang="tur")
        self.plakayazi.setText(str(text))
        self.gbyaz.setStyleSheet("QGroupBox { border: 1px solid green;}")
    
       
    
    
        
    def plakatespiti(self):
        self.imagecrop()
        self.gbtespit.setStyleSheet("QGroupBox { border: 1px solid green;}")
        self.gbcikti.setStyleSheet("QGroupBox { border: 1px solid green;}")
        self.gbyaz.setStyleSheet("QGroupBox { border: 1px solid yellow;}")
        
        
    def convert_nparray_to_QPixmap(self,img):
        w,h,ch = img.shape
        # Convert resulting image to pixmap
        if img.ndim == 1:
            img =  cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        
        qimg = QImage(img.data, h, w, 3*h, QImage.Format_RGB888) 
        qpixmap = QPixmap(qimg)
        
        return qpixmap
    
    
   
        
   
        
        
        
        
        
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = window()
    window.show()
    sys.exit(app.exec())
    