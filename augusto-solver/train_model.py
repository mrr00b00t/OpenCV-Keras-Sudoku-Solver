import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout,Flatten
from tensorflow.keras.layers import Conv2D,MaxPooling2D
import pickle


path = 'digitos'
test_ratio = 0.2
val_ratio = 0.2
img_shape= (32, 32, 3)
batch_size= 50
epochs = 10

images = []
labels = [] 
classes_list = os.listdir(path)
print("Número de classes: ", len(classes_list))

n_classes = len(classes_list)

for x in range (0, n_classes):
    images_names = os.listdir(os.path.join(path, str(x)))

    for y in images_names:
        cv2_img = cv2.imread(os.path.join(path, str(x), y))
        cv2_img = cv2.resize(cv2_img, (32,32))
        images.append(cv2_img)
        labels.append(x)

images = np.array(images)
labels = np.array(labels)

import matplotlib.pyplot as plt
plt.hist(labels)
plt.show()

print(len(images))
print(labels)

X_train, X_test,y_train, y_test = train_test_split(images, labels, test_size=test_ratio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=val_ratio)

def pre_processing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

X_train = np.array(list(map(pre_processing,X_train)))
X_test = np.array(list(map(pre_processing,X_test)))
X_validation = np.array(list(map(pre_processing,X_validation)))

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)

data_gen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
data_gen.fit(X_train)

quit()

y_train = to_categorical(y_train, n_classes)
y_test = to_categorical(y_test, n_classes)
y_validation = to_categorical(y_validation, n_classes)


def my_model():
    n_filters = 60
    filter1 = (5, 5)
    filter2 = (3, 3)
    pool1 = (2, 2)
    n_nodes= 500

    model = Sequential()
    model.add((Conv2D(n_filters,filter1, input_shape=(img_shape[0],
                      img_shape[1],1), activation='relu')))
    model.add((Conv2D(n_filters, filter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=pool1))
    model.add((Conv2D(n_filters//2, filter2, activation='relu')))
    model.add((Conv2D(n_filters//2, filter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=pool1))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(n_nodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = my_model()
print(model.summary())

history = model.fit_generator(data_gen.flow(X_train,y_train,
                                 batch_size=batch_size),
                                 epochs=epochs,
                                 validation_data=(X_validation,y_validation),
                                 shuffle=1)

score = model.evaluate(X_test,y_test,verbose=0)
print('Test Score = ',score[0])
print('Test Accuracy =', score[1])

model.save('augusto-model')