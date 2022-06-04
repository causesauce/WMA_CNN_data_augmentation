import numpy as np
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# %%

dataset = keras.utils.image_dataset_from_directory(
    "data",
    #   validation_split=0.2,
    # labels='inferred'
    # subset="training",
    seed=111,
    image_size=(128, 128),
    batch_size=None
)
# print(dataset)
# for element in dataset:
#     print(element)

# %% plot it
import matplotlib.pyplot as plt

class_names = dataset.class_names
print(class_names)

plt.figure(figsize=(10, 10))
for images, labels in dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
        plt.show()

# %%
X, y = [], []
# dataset.
for image, label in dataset:
    X.append(image)
    y.append(label)
#   plt.imshow(image.numpy().astype("uint8"))
#  plt.show()
# print( label)
y = np.array(y)
X = np.array(X) / 255.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

# %% show first image
plt.imshow(X_train[0].numpy().astype("uint8"))
plt.show()

# %%

from keras import models, layers

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
model.build(input_shape=(128, 128, 3))
model.summary()

# %%
model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# %% model evaluation

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(test_acc)
