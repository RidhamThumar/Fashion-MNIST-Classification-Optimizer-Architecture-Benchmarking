import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import time

# Load Fashion MNIST Dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape data for CNN
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# One-hot encode labels
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# Class names for Fashion MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Number of images to display
num_images = 10

# Plot images with labels
plt.figure(figsize=(12, 4))
for i in range(num_images):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(class_names[y_train[i]])
    plt.axis('off')
plt.show()

def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(10, activation='softmax')
    ])
    return model

optimizers = {
    'SGD': tf.keras.optimizers.SGD(),
    'Adam': tf.keras.optimizers.Adam(),
    'RMSprop': tf.keras.optimizers.RMSprop(),
    'Adagrad': tf.keras.optimizers.Adagrad()
}

results = {}

for name, optimizer in optimizers.items():
    print(f"Training with {name} optimizer:")
    model = create_model()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    start_time = time.time()
    history = model.fit(x_train, y_train_cat, epochs=10, batch_size=128, validation_split=0.2, verbose=1)
    elapsed_time = time.time() - start_time
    
    test_loss, test_accuracy = model.evaluate(x_test, y_test_cat, verbose=0)
    y_pred = np.argmax(model.predict(x_test), axis=1)
    
    results[name] = {
        'model': model,
        'history': history,
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'y_pred': y_pred,
        'elapsed_time': elapsed_time
    }
    print(f"{name} - Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}, Time: {elapsed_time:.2f}s\n")

plt.figure(figsize=(14, 6))

# Accuracy Plot
plt.subplot(1, 2, 1)
for name in results:
    plt.plot(results[name]['history'].history['val_accuracy'], label=f'{name}')
plt.title('Validation Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
for name in results:
    plt.plot(results[name]['history'].history['val_loss'], label=f'{name}')
plt.title('Validation Loss Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

for name in results:
    cm = confusion_matrix(y_test, results[name]['y_pred'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(i) for i in range(10)])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {name}')
    plt.show()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

num_images = 5  # Number of test images to display

for name in results:
    print(f"Sample Predictions using {name}:")
    plt.figure(figsize=(15, 3))
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        pred_label = class_names[results[name]['y_pred'][i]]
        true_label = class_names[y_test[i]]
        color = 'green' if pred_label == true_label else 'red'
        plt.title(f"P: {pred_label}\nT: {true_label}", color=color)
        plt.axis('off')
    plt.show()

import pandas as pd

summary_data = {
    'Optimizer': [],
    'Test Accuracy': [],
    'Test Loss': [],
    'Training Time (s)': []
}

for name in results:
    summary_data['Optimizer'].append(name)
    summary_data['Test Accuracy'].append(round(results[name]['test_accuracy'], 4))
    summary_data['Test Loss'].append(round(results[name]['test_loss'], 4))
    summary_data['Training Time (s)'].append(round(results[name]['elapsed_time'], 2))

df_summary = pd.DataFrame(summary_data)
print("\nPerformance Summary:\n")
print(df_summary)