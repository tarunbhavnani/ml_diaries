
#https://www.hackerearth.com/practice/machine-learning/transfer-learning/transfer-learning-intro/tutorial/
import numpy as np
from keras.datasets import cifar10

#Load the dataset:
(X_train, y_train), (X_test, y_test) = cifar10.load_data()


print("There are {} train images and {} test images.".format(X_train.shape[0], X_test.shape[0]))
print('There are {} unique classes to predict.'.format(np.unique(y_train).shape[0]))



#One-hot encoding the labels
num_classes = 10
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 10))

for i in range(1, 9):
    img = X_train[i-1]
    fig.add_subplot(2, 4, i)
    plt.imshow(img)

print('Shape of each image in the training data: ', X_train.shape[1:])

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.layers import Dropout, Flatten, GlobalAveragePooling2D




#build sequential model

model= Sequential()

model.add(Conv2D(32,(3,3), activation='relu', input_shape= (32,32,3)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(Dense(10, activation='softmax'))


model.summary()

#We will be using ‘binary cross-entropy’ as the loss function, ‘adam’ as the optimizer 
#and ‘accuracy’ as the performance metric.


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

"""Finally, we will rescale our data. Rescale is a value by which we will multiply the data 
such that the resultant values lie in the range (0-1). So, in general, scaling ensures that 
just because some features are big in magnitude, it doesn’t mean they act as the main features
 in predicting the label."""

X_train_scratch = X_train/255.
X_test_scratch = X_test/255.


#Creating a checkpointer  to save the weights of the best model!!
from keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint(filepath='scratchmodel.best.hdf5', 
                               verbose=1,save_best_only=True)


#Fitting the model on the train data and labels.

"Dovoding the data in to 32 batches and training on ten epochs"
#model.fit(X_train, y_train, batch_size=32, epochs=10, 
#          verbose=1, callbacks=[checkpointer], validation_split=0.2, shuffle=True)


model.fit(X_train, y_train, batch_size=32, epochs=3, verbose=1, callbacks=[checkpointer],
          validation_split=.2, shuffle=True)




#Evaluate the model on the test data
score = model.evaluate(X_test, y_test)

#Accuracy on test data
print('Accuracy on the Test Images: ', score[1])
#Accuracy on the Test Images:  0.8200000054359436


#so we have some accuracy
#we will try to use transfer learning to improve it
#we will download the pretrained model Resnet50 model,
# pre-trained on the ‘Imagenet weights’ to implement transfer learning. 


#Importing the ResNet50 model
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.resnet50 import ResNet50, preprocess_input


#Loading the ResNet50 model with pre-trained ImageNet weights
#model = ResNet50(weights='imagenet', include_top=False, input_shape=(200, 200, 3))
model_tl= ResNet50(weights='imagenet', include_top=False, input_shape=(200, 200, 3))
model_tl.summary()

for layer in model.layers:
    print(layer)

"""The Cifar-10 dataset is small and similar to the ‘ImageNet’ dataset. 
So, we will remove the fully connected layers of the pre-trained network near the end. 
To implement this, we set ‘include_top = False’, while loading the ResNet50 model.
"""

#Reshaping the training data
X_train_new = np.array([imresize(X_train[i], (200, 200, 3)) for i in range(0, len(X_train))]).astype('float32')

#Preprocessing the data, so that it can be fed to the pre-trained ResNet50 model. 
resnet_train_input = preprocess_input(X_train_new)

#Creating bottleneck features for the training data
train_features = model.predict(resnet_train_input)

#Saving the bottleneck features
np.savez('resnet_features_train', features=train_features)



X_train_new = np.array([imresize(X_train[i], (200, 200, 3)) for i in range(0, len(X_train))]).astype('float32')




