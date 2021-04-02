import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#Normalizing Dataset
x_train = keras.utils.normalize(x_train, axis=1)
x_test = keras.utils.normalize(x_test, axis =1)
#Building Model
model=Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(300, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation="softmax"))
#Compiling Model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
#Fitting the Model
model.fit(x_train, y_train, epochs=10)
model.save('model.hdf5')
#Evaluate the Model
print(model.evaluate(x_test, y_test))
#Predicting First 10 test images
pred = model.predict(x_test[:10])
# print(pred)
p=np.argmax(pred, axis=1)
print(p)
print(y_test[:10])
#Visualizing prediction
for i in range(10):
  plt.imshow(x_test[i], cmap='binary')
  plt.title("Original: {}, Predicted: {}".format(y_test[i], p[i]))
  plt.axis("Off")
  plt.figure()