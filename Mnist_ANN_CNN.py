'''Handwritten digits classification using neural network'''
'''We will use ANN AND CNN for learning and training'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
print(1)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print(len(x_train))
print(plt.matshow(x_train[2]))
plt.show()
print(y_train[:5])
print(x_train.shape)

'''SCALING PROCESS'''

x_train = x_train/255
x_test = x_test / 255
# print(x_train[0])

x_train_flattened = x_train.reshape(len(x_train), 28*28)
x_test_flattened = x_test.reshape(len(x_test), 28*28)
# print(x_train_flattened.shape)
# print(x_test_flattened[0])

'''MODEL FITTING'''
model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train_flattened, y_train, epochs=5)

'''Accuracy : 92.62%'''

model.evaluate(x_test_flattened, y_test)

# coustom prediction
y_pred = model.predict(x_test_flattened)
np.argmax(y_pred[1])

y_pred_labels = [np.argmax(i) for i in y_pred]
print(y_pred_labels[:5])

cm = tf.math.confusion_matrix(labels=y_test, predictions=y_pred_labels)

import seaborn as sn
plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True, fmt='d')  # creates map using cm
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

''' NOW WE WILL BUILD MODEL USING HIDDEN LAYER '''
'''Using relu and sigmoid'''

model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train_flattened, y_train, epochs=5)
model.evaluate(x_test_flattened, y_test)

# prediction and plotting

y_predicted = model.predict(x_test_flattened)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
cm = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)

plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

print("This ends our ANN MODEL")
print("MODEL Accuracy normal =", 92.62, "%")
print("MODEL Accuracy with 1 hidden layer =", 97.56, "%")

'''CNN for classification'''

cnn_model = keras.Sequential([

    layers.Conv2D(30, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

# compile and fit the model
cnn_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

cnn_model.fit(x_train, y_train, epochs=5)

cnn_model.evaluate(x_test,y_test)


