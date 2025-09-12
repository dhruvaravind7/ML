import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras.callbacks import EarlyStopping
import tensorflow as tf
from keras import layers, models, optimizers, losses, regularizers
import numpy as np
import matplotlib.pyplot as plt


# def getModels():
#     model_list = []
#     # model_list.append(models.Sequential([layers.Dense(25, activation = "relu"),
#     #                                     layers.Dense(15, activation="relu"),
#     #                                     layers.Dense(10, activation="linear")])
#     # )
#     # model_list.append(models.Sequential([layers.Dense(25, activation="relu"),
#     #                                     layers.Dense(10, activation="linear")])
#     # )
#     # model_list.append(models.Sequential([layers.Dense(10, activation="linear")]))
#     # model_list.append(models.Sequential([layers.Dense(256, activation="relu"),
#     #                                     layers.Dense(128, activation="relu"),
#     #                                     layers.Dense(10, activation="linear")])
#     # )
    
#     # model_list.append(models.Sequential([layers.Dense(256, activation="relu"),
#     #                                      layers.Dropout(0.1),
#     #                                      layers.Dense(128, activation="relu"),
#     #                                      layers.Dropout(0.1),
#     #                                      layers.Dense(10, activation="linear")]))
    
#     # model_list.append(models.Sequential([layers.Dense(256, activation="relu"),
#     #                                      layers.Dropout(0.1),
#     #                                      layers.Dense(128, activation="relu"),
#     #                                      layers.Dropout(0.1),
#     #                                      layers.Dense(64, activation = "relu"),
#     #                                      layers.Dropout(0.1),
#     #                                      layers.Dense(10, activation="linear")]))
    
#     # model_list.append(models.Sequential([layers.Dense(128, activation="relu"),
#     #                                     layers.Dense(128, activation="relu"),
#     #                                     layers.Dense(64, activation="relu"),
#     #                                     layers.Dense(64, activation="relu"),
#     #                                     layers.Dense(10, activation="linear")])
#     # )
#     model_list.append(models.Sequential([layers.Dense(512, activation="relu"),
#                                         layers.Dropout(0.1),
#                                         layers.Dense(256, activation="relu"),
#                                         layers.Dropout(0.1),
#                                         layers.Dense(128, activation="relu"),
#                                         layers.Dropout(0.1),
#                                         layers.Dense(10, activation="linear")])
#     )
#     return(model_list)
    
    
    

model = models.Sequential([layers.Dense(512, activation="relu"),
                                        layers.Dropout(0.1),
                                        layers.Dense(256, activation="relu"),
                                        layers.Dropout(0.1),
                                        layers.Dense(128, activation="relu"),
                                        layers.Dropout(0.1),
                                        layers.Dense(10, activation="linear")])
#Loading the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_new_train = []
y_new_train = []
counter = 0
for data in x_train:
    new_shift_up = data.copy()
    new_shift_down = data.copy()
    for row in range(len(data)):
        if row < len(data)-5:
            new_shift_up[row] = data[row+5]
        else:
            new_shift_up[row] = np.zeros(28)
        
        if (row >= 5):
            new_shift_down[row] = data[row-5]
        else:
            new_shift_down[row] = np.zeros(28)
    x_new_train.append(new_shift_down)
    x_new_train.append(new_shift_up)
    y_new_train.append(y_train[counter])
    y_new_train.append(y_train[counter])    
    counter += 1

x_train = np.append(x_train, x_new_train, axis= 0)
y_train = np.append(y_train, y_new_train)

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
early_stop = EarlyStopping(monitor="val_loss",
                           patience=3,
                           restore_best_weights=True
)


model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
            loss=losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"])

model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=[x_cv, y_cv], callbacks = [early_stop])

train_loss, train_acc = model.evaluate(x_train, y_train)
cv_loss, cv_acc = model.evaluate(x_cv, y_cv)


print(f"Training Loss: {train_loss}, Training Accuracy: {train_acc}\n")
print(f"CV Loss: {cv_loss}, CV Accuracy: {cv_acc}")

pred_logits = model.predict(x_test)
pred_classes = tf.argmax(pred_logits, axis=1)
test_accuracy = np.mean(pred_classes == y_test)
print(f"Test Accuracy: {test_accuracy}")