import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt

import random

x_train = []  # Price of Item
y_train = []  # Classification of item as expensive or cheap.

x_test = []
y_test = []

for i in range(10):
  randNum = random.randint(0, 10)
  response = input(
      f"Is ${randNum} for a Loaf of Bread, considered Expensive or Cheap? (Exp, Cheap)"
  )
  x_train.append([randNum])

  if response == "Exp":
    y_train.append([1, 0])  # Classification is Expensive
  elif response == "Cheap":
    y_train.append([0, 1])  # Classification is Cheap

# First Layer is used to process input data (price of item).
# Second Layer is 2 units, used for classification of whether price is Expensive or Cheap.
model = keras.Sequential([
    keras.layers.Dense(1, input_shape=[1]),
    keras.layers.Dense(2, activation="softmax")
])

#Compile Model
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Train Model
model.fit(x_train, y_train, epochs=1024, verbose=1)

for i in range(128):
  randNum = random.randint(0, 10)
  prediction = model.predict([randNum], verbose=0)
  x_test.append(randNum)

  if prediction[0][0] > prediction[0][1]:
    y_test.append(1)
    print(f"{randNum} is expensive")
  else:
    y_test.append(0)
    print(f"{randNum} is cheap")

# Graph Predictions of Neural Network
plt.scatter(x_test, y_test, c=y_test)
plt.xlabel("Price for Loaf of Bread")
plt.ylabel("Cheap (0) or Expensive (1)")
plt.show()

del x_train
del y_train
