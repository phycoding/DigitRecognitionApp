import tensorflow as tf
import numpy as np
import sys
import os
from tensorflow.keras import datasets, layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2, VGG16, ResNet50, InceptionV3
from keras.preprocessing.image import img_to_array, array_to_img


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


train_images = np.asarray([img_to_array(array_to_img(im, scale=False).resize((32,32))) for im in train_images])
test_images = np.asarray([img_to_array(array_to_img(im, scale=False).resize((32,32))) for im in test_images])

#dict to store models
models_dict = {}

# Build the custom model
custom_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
custom_model.compile(optimizer='RMSProp',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

learning_rate_reduction = callbacks.ReduceLROnPlateau(monitor='val_accuracy',                                                   patience=3, 
                                                      verbose=1, 
                                                      factor=0.5, 
                                                      min_lr=0.00001)
models_dict['CustomModel'] = custom_model

# Define pre-trained models with custom final layers
def add_final_layer(base_model):
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(10, activation='softmax')
    ])
    return model


# MobileNetV2
mobile_model = add_final_layer(MobileNetV2(input_shape=(32, 32, 1), include_top=False, weights=None))
mobile_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
models_dict['MobileNetV2'] = mobile_model

# VGG16
vgg_model = add_final_layer(VGG16(input_shape=(32, 32, 1), include_top=False, weights=None))
vgg_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
models_dict['VGG16'] = vgg_model

# ResNet50
resnet_model = add_final_layer(ResNet50(input_shape=(32, 32, 1), include_top=False, weights=None))
resnet_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
models_dict['ResNet50'] = resnet_model


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
# Train all models
for name, model in models_dict.items():
    print(f"Training {name}...")
    if name == 'CustomModel':
        history = model.fit(datagen.flow(train_images, train_labels, batch_size=batch_size),
                        epochs=epochs, validation_data=(test_images, test_labels), 
                        callbacks=[learning_rate_reduction], verbose=2)
    else:
        history = model.fit(datagen.flow(train_images, train_labels, batch_size=batch_size),
                        epochs=epochs, validation_data=(test_images, test_labels), 
                         verbose=2)

    # Save the model
    model.save(f'{name}.keras')
    # Visualize the training process
    visualizer = TrainingVisualizer(history)
    # visualizer.plot_accuracy()
    # visualizer.plot_loss()
    # visualizer.plot_learning_rate()
    print(f"{name} training completed and model saved.")