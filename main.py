from os import listdir
from os.path import isfile, join
from scipy.fftpack import fft2, fftshift
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import gc

########################################DNN BINARY###############################################"
# Load and preprocess images
mypath_ernest = 'C:/Users/aacer/PycharmProjects/TP1/TA/TP5/dataset/ernest_celestine'
onlyfiles_ernest = [f for f in listdir(mypath_ernest) if isfile(join(mypath_ernest, f))]
images_ernest = []

for filename in onlyfiles_ernest:
    image = cv2.imread(join(mypath_ernest, filename))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (0, 0), fx=0.4, fy=0.4, interpolation=cv2.INTER_NEAREST)
    images_ernest.append(image_resized)

# Apply FFT and normalize the images
fft_images_ernest = []

for image in images_ernest:
    fft_result = fft2(image)
    fft_shifted = fftshift(fft_result)
    magnitude_spectrum = np.abs(fft_shifted)
    normalized_spectrum = magnitude_spectrum / (np.max(magnitude_spectrum) + 1e-10)
    fft_images_ernest.append(normalized_spectrum)

# Load and preprocess images for 'Toy Story'
mypath_toy_story = 'C:/Users/aacer/PycharmProjects/TP1/TA/TP5/dataset/toy_story_3'
onlyfiles_toy_story = [f for f in listdir(mypath_toy_story) if isfile(join(mypath_toy_story, f))]
images_toy_story = []

for filename in onlyfiles_toy_story:
    image = cv2.imread(join(mypath_toy_story, filename))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (0, 0), fx=0.4, fy=0.4, interpolation=cv2.INTER_NEAREST)
    images_toy_story.append(image_resized)

# Apply FFT and normalize the images for 'Toy Story'
fft_images_toy_story = []

for image in images_toy_story:
    fft_result = fft2(image)
    fft_shifted = fftshift(fft_result)
    magnitude_spectrum = np.abs(fft_shifted)
    normalized_spectrum = magnitude_spectrum / (np.max(magnitude_spectrum) + 1e-10)
    fft_images_toy_story.append(normalized_spectrum)

# Concatenate FFT data
X_train_fft = np.concatenate((fft_images_ernest, fft_images_toy_story), axis=0)

# Handle NaN or infinite values
X_train_fft = np.nan_to_num(X_train_fft)

# Create labels and concatenate data
Y_ernest = np.ones(len(fft_images_ernest))
Y_toy_story = np.zeros(len(fft_images_toy_story))
Y_train = np.concatenate((Y_ernest, Y_toy_story), axis=0)

# Reshape data to (None, 216) for compatibility with binary cross-entropy
X_train_fft = X_train_fft.reshape(X_train_fft.shape[0], -1)

# Split the data into training, validation, and test sets
X_train, X_temp, Y_train, Y_temp = train_test_split(X_train_fft, Y_train, test_size=0.2, random_state=42)
X_valid, X_test, Y_valid, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

# Create the neural network model
model = Sequential([
    Dense(16, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])

# Compile the model
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(X_train, Y_train, epochs=64, batch_size=32, validation_data=(X_valid, Y_valid))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, Y_test)

# Print training and validation results
print("Training Results:")
print(f'Training Loss: {history.history["loss"][-1]:.4f}')
print(f'Training Accuracy: {history.history["accuracy"][-1]:.4f}')

print("\nValidation Results:")
print(f'Validation Loss: {history.history["val_loss"][-1]:.4f}')
print(f'Validation Accuracy: {history.history["val_accuracy"][-1]:.4f}')

# Print test results
print("\nTest Results:")
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# Plot training, validation, and test accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.axhline(y=test_accuracy, color='r', linestyle='--', label='Test Accuracy')
plt.title('DNN Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training, validation, and test loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.axhline(y=test_loss, color='r', linestyle='--', label='Test Loss')
plt.title('DNN Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.close('all')

#free memory
del onlyfiles_ernest, images_ernest, fft_images_ernest, onlyfiles_toy_story, images_toy_story
del fft_images_toy_story, X_train_fft, Y_ernest, Y_toy_story, Y_train, X_train, X_valid
del X_test, history, test_loss, test_accuracy

tf.keras.backend.clear_session()

cv2.destroyAllWindows()

gc.collect()

#####################################CNN BINARY####################################################

# Load and preprocess images
mypath_ernest = 'C:/Users/aacer/PycharmProjects/TP1/TA/TP5/dataset/ernest_celestine'
images_ernest = []
labels_ernest = []

for file in os.listdir(mypath_ernest):
    image_path = os.path.join(mypath_ernest, file)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (64, 64), interpolation=cv2.INTER_NEAREST)
    image_resized = np.array(image_resized).reshape(64, 64, 3)
    image_resized = image_resized.astype('float32') / 255
    images_ernest.append(image_resized)
    labels_ernest.append('ernest-celestine')

mypath_toy_story = 'C:/Users/aacer/PycharmProjects/TP1/TA/TP5/dataset/toy_story_3'
images_toy_story = []
labels_toy_story = []

for file in os.listdir(mypath_toy_story):
    image_path = os.path.join(mypath_toy_story, file)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (64, 64), interpolation=cv2.INTER_NEAREST)
    image_resized = np.array(image_resized).reshape(64, 64, 3)
    image_resized = image_resized.astype('float32') / 255
    images_toy_story.append(image_resized)
    labels_toy_story.append('toy_story_3')

# Concatenate images
X_train_images = np.concatenate((images_ernest, images_toy_story), axis=0)

# Create labels and concatenate data
Y_ernest = np.full(len(images_ernest), 'ernest-celestine', dtype=object)
Y_toy_story = np.full(len(images_toy_story), 'toy_story_3', dtype=object)
Y_train = np.concatenate((Y_ernest, Y_toy_story), axis=0)

# Encode labels
label_encoder = LabelEncoder()
Y_train_encoded = label_encoder.fit_transform(Y_train)

# Convert to one-hot encoding
Y_train_one_hot = to_categorical(Y_train_encoded)


# Split the data into training, validation, and test sets
X_train, X_temp, Y_train_one_hot, Y_temp = train_test_split(X_train_images, Y_train_one_hot, test_size=0.2, random_state=42)
X_valid, X_test, Y_valid_one_hot, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)


# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(X_train)

# Create the CNN model
model = Sequential([
    Conv2D(8, (3, 3), activation="relu", input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(16, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(100, activation="relu"),
    Dense(len(label_encoder.classes_), activation="softmax")
])

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model with data augmentation
history = model.fit(datagen.flow(X_train, Y_train_one_hot, batch_size=32), epochs=64, validation_data=(X_valid, Y_valid_one_hot))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, Y_test)
train_loss, train_accuracy = model.evaluate(X_train, Y_train_one_hot)
val_loss, val_accuracy = model.evaluate(X_valid, Y_valid_one_hot)

# Print training, validation, and test results
print(f'Training Loss: {train_loss:.4f}')
print(f'Training Accuracy: {train_accuracy:.4f}')
print(f'Validation Loss: {val_loss:.4f}')
print(f'Validation Accuracy: {val_accuracy:.4f}')
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# Plot training, validation, and test accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.axhline(y=test_accuracy, color='r', linestyle='--', label='Test Accuracy')
plt.title('CNN Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training, validation, and test loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.axhline(y=test_loss, color='r', linestyle='--', label='Test Loss')
plt.title('CNN Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.close('all')


# free memory
del images_ernest, labels_ernest, images_toy_story, labels_toy_story, X_train_images
del Y_ernest, Y_toy_story, Y_train, Y_train_encoded, Y_train_one_hot
del X_train, X_temp, Y_temp, X_valid, X_test, Y_valid_one_hot, Y_test

plt.close('all')

tf.keras.backend.clear_session()

cv2.destroyAllWindows()

gc.collect()

#####################################################CNN 4 CLASSES################################################

# Load and preprocess images
target_size = (128, 128)

# Load and preprocess images from different folders
images_ernest = []
labels_ernest = []
for i, file in enumerate(os.listdir('C:/Users/aacer/PycharmProjects/TP1/TA/TP5/dataset/ernest_celestine')):
    if i >= 500:
        break
    image_path = os.path.join('C:/Users/aacer/PycharmProjects/TP1/TA/TP5/dataset/ernest_celestine', file)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST)
    image_resized = np.array(image_resized).reshape(target_size[0], target_size[1], 3)
    image_resized = image_resized.astype('float32') / 255
    images_ernest.append(image_resized)
    labels_ernest.append('ernest_celestine')

images_toy_story_1 = []
labels_toy_story_1 = []
for i, file in enumerate(os.listdir('C:/Users/aacer/PycharmProjects/TP1/TA/TP5/dataset/toy_story_1')):
    if i >= 500:
        break
    image_path = os.path.join('C:/Users/aacer/PycharmProjects/TP1/TA/TP5/dataset/toy_story_1', file)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST)
    image_resized = np.array(image_resized).reshape(target_size[0], target_size[1], 3)
    image_resized = image_resized.astype('float32') / 255
    images_toy_story_1.append(image_resized)
    labels_toy_story_1.append('toy_story_1')

images_toy_story_2 = []
labels_toy_story_2 = []
for i, file in enumerate(os.listdir('C:/Users/aacer/PycharmProjects/TP1/TA/TP5/dataset/toy_story_2')):
    if i >= 500:
        break
    image_path = os.path.join('C:/Users/aacer/PycharmProjects/TP1/TA/TP5/dataset/toy_story_2', file)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST)
    image_resized = np.array(image_resized).reshape(target_size[0], target_size[1], 3)
    image_resized = image_resized.astype('float32') / 255
    images_toy_story_2.append(image_resized)
    labels_toy_story_2.append('toy_story_2')

images_toy_story_3 = []
labels_toy_story_3 = []
for i, file in enumerate(os.listdir('C:/Users/aacer/PycharmProjects/TP1/TA/TP5/dataset/toy_story_3')):
    if i >= 500:
        break
    image_path = os.path.join('C:/Users/aacer/PycharmProjects/TP1/TA/TP5/dataset/toy_story_3', file)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST)
    image_resized = np.array(image_resized).reshape(target_size[0], target_size[1], 3)
    image_resized = image_resized.astype('float32') / 255
    images_toy_story_3.append(image_resized)
    labels_toy_story_3.append('toy_story_3')

# Concatenate images
X_train_images = np.concatenate((images_ernest, images_toy_story_1, images_toy_story_2, images_toy_story_3), axis=0)

# Create labels and concatenate data
Y_ernest = np.full(len(images_ernest), 'ernest_celestine', dtype=object)
Y_toy_story_1 = np.full(len(images_toy_story_1), 'toy_story_1', dtype=object)
Y_toy_story_2 = np.full(len(images_toy_story_2), 'toy_story_2', dtype=object)
Y_toy_story_3 = np.full(len(images_toy_story_3), 'toy_story_3', dtype=object)
Y_train = np.concatenate((Y_ernest, Y_toy_story_1, Y_toy_story_2, Y_toy_story_3), axis=0)

# Encode labels
label_encoder = LabelEncoder()
Y_train_encoded = label_encoder.fit_transform(Y_train)

# Convert to one-hot encoding
Y_train_one_hot = to_categorical(Y_train_encoded)

# Split the data into training, validation, and test sets
X_train, X_temp, Y_train_one_hot, Y_temp = train_test_split(X_train_images, Y_train_one_hot, test_size=0.2, random_state=42)
X_valid, X_test, Y_valid_one_hot, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(X_train)

# Create the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation="relu"),
    Dropout(0.6),  # Adjust dropout rate
    Dense(256, activation="relu"),
    Dense(128, activation="relu"),
    Dropout(0.3),  # Adjust dropout rate
    Dense(64, activation="relu"),
    Dense(len(label_encoder.classes_), activation="softmax")
])

# Learning rate scheduling
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=10000, decay_rate=0.9)
custom_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Compile the model with the custom optimizer
model.compile(optimizer=custom_optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# Early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with data augmentation and early stopping
history = model.fit(datagen.flow(X_train, Y_train_one_hot, batch_size=32), epochs=64,
                    validation_data=(X_valid, Y_valid_one_hot), callbacks=[early_stopping])

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, Y_test)
train_loss, train_accuracy = model.evaluate(X_train, Y_train_one_hot)
val_loss, val_accuracy = model.evaluate(X_valid, Y_valid_one_hot)

# Print training, validation, and test results
print(f'Training Loss: {train_loss:.4f}')
print(f'Training Accuracy: {train_accuracy:.4f}')
print(f'Validation Loss: {val_loss:.4f}')
print(f'Validation Accuracy: {val_accuracy:.4f}')
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# Plot training, validation, and test accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.axhline(y=test_accuracy, color='r', linestyle='--', label='Test Accuracy')
plt.title('CNN-4 v2 classes Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training, validation, and test loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.axhline(y=test_loss, color='r', linestyle='--', label='Test Loss')
plt.title('CNN-4 v2 classes Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.close('all')

#free memory
del images_ernest, labels_ernest
del images_toy_story_1, labels_toy_story_1
del images_toy_story_2, labels_toy_story_2
del images_toy_story_3, labels_toy_story_3
del X_train_images
del Y_ernest, Y_toy_story_1, Y_toy_story_2, Y_toy_story_3, Y_train

tf.keras.backend.clear_session()

cv2.destroyAllWindows()

gc.collect()



