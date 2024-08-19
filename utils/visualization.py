import matplotlib.pyplot as plt

class TrainingVisualizer:
    def __init__(self, history):
        self.history = history

    def plot_accuracy(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_loss(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_learning_rate(self):
        if 'lr' in self.history.history:
            plt.figure(figsize=(8, 6))
            plt.plot(self.history.history['lr'], label='Learning Rate')
            plt.title('Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.legend()
            plt.grid(True)
            plt.show()
