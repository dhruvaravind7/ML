import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras.callbacks import EarlyStopping
import tensorflow as tf
from keras import layers, models, optimizers, losses, regularizers
import numpy as np
import matplotlib.pyplot as plt


def getModels():
    model_list = []
    model_list.append(models.Sequential(layers.Dense(25, activation = "relu"),
                                        layers.Dense(15, activation="relu"),
                                        layers.Dense(10, activation="linear"))
    )
    model_list.append(models.Sequential(layers.Dense(25, activation="relu"),
                                        layers.Dense(10, activation="linear"))
    )
    model_list.append(models.Sequential(layers.Dense(10, activation="linear")))
    model_list.append(models.Sequential(layers.Dense(256, activation="relu"),
                                        layers.Dense(128, activation="relu"),
                                        layers.Dense(10, activation="linear"))
    )
    model_list.append(models.Sequential(layers.Dense(128, activation="relu"),
                                        layers.Dense(128, activation="relu"),
                                        layers.Dense(64, activation="relu"),
                                        layers.Dense(64, activation="relu"),
                                        layers.Dense(10, activation="linear"))
    )
    model_list.append(models.Sequential(layers.Dense(512, activation="relu"),
                                        layers.Dropout(0.3),
                                        layers.Dense(256, activation="relu"),
                                        layers.Dropout(0.3),
                                        layers.Dense(128, activation="relu"),
                                        layers.Dense(10, activation="linear"))
    )
    return(model_list)
    
    
    


#Loading the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28*28)
x_test = x_test.reshape(x_test.shape[0], 28*28)

num_samples= len(x_train)

cv = 0.2
num_cv = int(num_samples*cv)

indices = np.random.permutation(num_samples)

cv_index = indices[:num_cv]
train_index = indices[num_cv:]

# Creating the cv set and the training set
x_cv, y_cv = x_train[cv_index], y_train[cv_index]
x_train, y_train = x_train[train_index], y_train[train_index]

# Creating the model

model_list = getModels()

early_stop = EarlyStopping(monitor="val_loss",
                           patience=5,
                           restore_best_weights=True
)


model_training_losses = []
model_cv_losses = []

for model in model_list:
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
              loss=losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=[x_cv, y_cv], callbacks = [early_stop])

    train_loss, train_acc = model.evaluate(x_train, y_train)
    cv_loss, cv_acc = model.evaluate(x_cv, y_cv)

    model_training_losses.append(train_loss)
    model_cv_losses.append(cv_loss)
    
print(model_training_losses + "\n" + model_cv_losses)
print(model_cv_losses.index(min(model_cv_losses)))