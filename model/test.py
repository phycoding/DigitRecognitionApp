import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Load the saved model
model = tf.keras.models.load_model('digit_recognition_model.h5')

# Load the MNIST dataset
(_, _), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Preprocess the test data
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}")

# Get predictions
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Generate a classification report
print("\nClassification Report:\n")
print(classification_report(test_labels, predicted_labels))

# Generate a confusion matrix
conf_matrix = confusion_matrix(test_labels, predicted_labels)
print("\nConfusion Matrix:\n")
print(conf_matrix)

# Visualize some of the test images along with their predicted and true labels
def plot_images(images, labels_true, labels_pred, num_rows=3, num_cols=3):
    plt.figure(figsize=(10, 10))
    for i in range(num_rows * num_cols):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        plt.title(f"True: {labels_true[i]}, Pred: {labels_pred[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Plot some sample test images along with predictions
plot_images(test_images, test_labels, predicted_labels)
