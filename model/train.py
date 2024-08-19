import tensorflow as tf
from tensorflow.keras import datasets, layers, models, callbacks
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.visualization import TrainingVisualizer


# Check if GPU is available
gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
if gpus:
    print("GPU is available and will be used for training.")
else:
    print("No GPU available, training on CPU.")

# Load the dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Preprocess the data (Reshape and normalization)
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Build the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='RMSProp',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

learning_rate_reduction = callbacks.ReduceLROnPlateau(monitor='val_accuracy', 
                                                      patience=3, 
                                                      verbose=1, 
                                                      factor=0.5, 
                                                      min_lr=0.00001)
epochs = 30 
batch_size = 86

# With data augmentation to prevent overfitting.
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.1,  # Randomly zoom image 
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False)  # randomly flip images

datagen.fit(train_images)

# Train the model
history = model.fit(datagen.flow(train_images, train_labels, batch_size=batch_size),
                    epochs=epochs, validation_data=(test_images, test_labels), 
                    callbacks=[learning_rate_reduction], verbose=2)

# Save the model
model.save('digit_recognition_model.keras')
# Visualize the training process
visualizer = TrainingVisualizer(history)
visualizer.plot_accuracy()
visualizer.plot_loss()
visualizer.plot_learning_rate()