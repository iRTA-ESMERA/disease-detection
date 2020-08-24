#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import mlflow.keras

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import CenterCrop
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

from sklearn.metrics import  confusion_matrix


# In[2]:


train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator(validation_split=0.2) # set validation split

train_labels = pd.read_csv('/home/chris/Desktop/disease-detection/data/train_big.csv') 
test_labels = pd.read_csv('/home/chris/Desktop/disease-detection/data/test.csv') 





train_it = train_datagen.flow_from_dataframe(dataframe=train_labels, directory='/home/chris/Desktop/disease-detection/data/images_data/images/',x_col='image_id',y_col=["healthy","multiple_diseases","rust","scab"], class_mode='raw', batch_size=32)

test_it = test_datagen.flow_from_dataframe(dataframe=test_labels, directory='/home/chris/Desktop/disease-detection/data/images_data/test/',x_col='image_id',y_col=["healthy","multiple_diseases","rust","scab"], class_mode='raw',subset='training', batch_size=32)
val_it = test_datagen.flow_from_dataframe(dataframe=test_labels, directory='/home/chris/Desktop/disease-detection/data/images_data/test/',x_col='image_id',y_col=["healthy","multiple_diseases","rust","scab"], class_mode='raw',subset='validation', batch_size=32)

# train_generator = train_datagen.flow_from_directory(
#     train_data_dir,
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='binary',
#     subset='training') # set as training data

# validation_generator = train_datagen.flow_from_directory(
#     train_data_dir, # same directory as training data
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='binary',
#     subset='validation') # set as validation data


# featurewise_center=True,
# featurewise_std_normalization=True,


# In[3]:


# train_labels = pd.read_csv('/home/chris/Desktop/plant_disease/plant-pathology-2020-fgvc7/train.csv') 
# test_labels = pd.read_csv('/home/chris/Desktop/plant_disease/plant-pathology-2020-fgvc7/test.csv') 


# datagen = ImageDataGenerator()

# # # load and iterate training dataset
# train_it = datagen.flow_from_dataframe(dataframe=train_labels, directory='/home/chris/Desktop/plant_disease/plant-pathology-2020-fgvc7/data/train/',x_col='image_id',y_col=["healthy","multiple_diseases","rust","scab"], class_mode='raw', batch_size=32)
# # # load and iterate test dataset
# test_it = datagen.flow_from_dataframe(dataframe=test_labels, directory='/home/chris/Desktop/plant_disease/plant-pathology-2020-fgvc7/data/test/',x_col='image_id',y_col=["healthy","multiple_diseases","rust","scab"], class_mode='raw', batch_size=32)


# In[4]:


# #%%
# # Create a dataset.
# dataset = keras.preprocessing.image_dataset_from_directory(
#   '/home/chris/Desktop/grapes_detection/wgisd-1.0.0/thsant-wgisd-ab223e5/data', batch_size=10, image_size=(256, 256))

# # For demonstration, iterate over the batches yielded by the dataset.
# print(len(dataset))
# data_batches = []
# labels_batches = []
# for data, labels in dataset:
#     data_batches.append(data)
#     labels_batches.append(labels)
# #     print(data.shape)  # (64, 200, 200, 3)
# #     print(data.dtype)  # float32
# #     print(labels.shape)  # (64,)
# #     print(labels.dtype)  # int32
# print(len(data_batches))
# print(len(labels_batches))


# In[5]:


# print((train_it.next()))
data,labels = train_it.next()
print(data.shape)
print(type(data[0][0][0][0]))
print((data[0][0][0][0]))
plt.figure()
plt.imshow(data[0]/255)
plt.show()


# In[6]:


# #%%
# from tensorflow.keras.layers.experimental.preprocessing import CenterCrop
# from tensorflow.keras.layers.experimental.preprocessing import Rescaling

# # Example image data, with values in the [0, 255] range
# training_data = np.random.randint(0, 256, size=(64, 200, 200, 3)).astype("float32")

# cropper = CenterCrop(height=150, width=150)
# scaler = Rescaling(scale=1.0 / 255)

# output_data = scaler(cropper(training_data))
# print("shape:", output_data.shape)
# print("min:", np.min(output_data))
# print("max:", np.max(output_data))


# In[7]:


def residual_block(y, nb_channels, _strides=(1, 1), _project_shortcut=False):
    shortcut = y

    # down-sampling is performed with a stride of 2
    y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, kernel_regularizer=tf.keras.regularizers.l1_l2(), padding='same')(y)
    y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU()(y)

    y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=(1, 1), kernel_regularizer=tf.keras.regularizers.l1_l2(), padding='same')(y)
    y = layers.BatchNormalization()(y)

    # identity shortcuts used directly when the input and output are of the same dimensions
    if _project_shortcut or _strides != (1, 1):
        # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
        # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
        shortcut = layers.Conv2D(nb_channels, kernel_size=(1, 1), strides=_strides, kernel_regularizer=tf.keras.regularizers.l1_l2(), padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    y = layers.add([shortcut, y])
    y = layers.LeakyReLU()(y)

    return y


# In[8]:


#%% RESIDUAL NEURAL NETWORK
inputs = keras.Input(shape=(256, 256, 3))


# Center-crop images to 150x150
# x = CenterCrop(height=150, width=150)(inputs)
# Rescale images to [0, 1]
x = Rescaling(scale=1.0 / 255)(inputs)

x = layers.Conv2D(filters=8, kernel_size=(3, 3), kernel_regularizer=tf.keras.regularizers.l1_l2(), padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

x = residual_block(x,8)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

x = residual_block(x,16,_project_shortcut=True)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

x = residual_block(x,32,_project_shortcut=True)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

x = residual_block(x,64,_project_shortcut=True)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

x = residual_block(x,128,_project_shortcut=True)


# Apply global average pooling to get flat feature vectors
x = layers.GlobalMaxPool2D()(x)
# x = layers.Flatten()(x)
# x = layers.Dense(80,input_shape=(128,), activation='relu')(x)
# x = layers.Dense(16, activation='relu')(x)
# x = layers.Dense(4)(x)
# Add a dense classifier on top
num_classes = 4
outputs = layers.Dense(num_classes, activation="softmax")(x)


# In[9]:


# #%% SIMPLE CONVOLUTIONAL NEURAL NETWORK
# inputs = keras.Input(shape=(256, 256, 3))

# # Center-crop images to 150x150
# # x = CenterCrop(height=150, width=150)(inputs)
# # Rescale images to [0, 1]
# x = Rescaling(scale=1.0 / 255)(inputs)

# x = layers.Conv2D(filters=8, kernel_size=(3, 3), padding='same')(x)
# x = layers.LeakyReLU()(x)
# x = layers.MaxPooling2D(pool_size=(2, 2))(x)

# x = layers.Conv2D(filters=16, kernel_size=(3, 3))(x)
# x = layers.LeakyReLU()(x)
# x = layers.MaxPooling2D(pool_size=(2, 2))(x)

# x = layers.Conv2D(filters=32, kernel_size=(3, 3))(x)
# x = layers.LeakyReLU()(x)
# x = layers.MaxPooling2D(pool_size=(2, 2))(x)

# x = layers.Conv2D(filters=64, kernel_size=(3, 3))(x)
# x = layers.LeakyReLU()(x)
# x = layers.MaxPooling2D(pool_size=(2, 2))(x)


# # Apply global average pooling to get flat feature vectors
# # x = layers.GlobalMaxPool2D()(x)
# x = layers.Flatten()(x)
# x = layers.Dense(256, activation='relu')(x)
# # x = layers.Dense(80,input_shape=(128,), activation='relu')(x)
# x = layers.Dense(32, activation='relu')(x)
# # x = layers.Dense(4)(x)
# # Add a dense classifier on top
# num_classes = 4
# outputs = layers.Dense(num_classes, activation="softmax")(x)


# In[10]:


#%%
epochs=50
loss=keras.losses.CategoricalCrossentropy()
optimizer=keras.optimizers.Adam(learning_rate=0.001)
model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])


earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, epsilon=1e-4, mode='min')


# In[ ]:


# mlflow.set_experiment('test_project')
# mlflow.set_tag('test', 'test_project' )
# with mlflow.start_run():
mlflow.keras.autolog()
print(mlflow.get_tracking_uri())
mlflow.log_param("output", outputs)
mlflow.log_param("epochs", epochs)
mlflow.log_param("loss_function", loss)
mlflow.log_param("optimizer", optimizer)
mlflow.log_param("early_stopping", earlyStopping)
mlflow.log_param("model_checkpoint", mcp_save)
mlflow.log_param("reduce_lr_loss", reduce_lr_loss)

            
#     mlflow.keras.log_model(keras_model, model_dir)


history = model.fit(train_it, epochs=epochs, callbacks=[earlyStopping, mcp_save, reduce_lr_loss], validation_data=val_it)

loss, acc = model.evaluate(test_it)  # returns loss and metrics
print("loss: %.4f" % loss)
print("acc: %.4f" % acc)
mlflow.log_metric("final_acc", acc)


mlflow.end_run()


# In[ ]:


# from keras.utils import plot_model
# plot_model(model, to_file='model.png')


# In[84]:


# # print((labels))
# x,y = test_it.next()
# labels = []
# labels.append(y)
# for i in range(len(test_it)-1):
#     x,y = test_it.next()
#     labels.append(y)


# true_labels = []
# for i in labels:
#     for j in i:
#         true_labels.append(np.argmax(j))
# true_labels = np.asarray(true_labels)

# print('Confusion Matrix')
# print(confusion_matrix(true_labels, y_pred))


# In[ ]:





# In[ ]:




