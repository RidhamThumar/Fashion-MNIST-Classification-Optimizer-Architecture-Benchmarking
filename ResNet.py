import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from timm import create_model
import warnings
import pandas as pd

warnings.filterwarnings("ignore")  # Suppress warnings

# --- Configuration ---
BATCH_SIZE = 64
NUM_WORKERS = 4  # Adjust based on your system
NUM_EPOCHS = 5
MODEL_NAME = "resnet18" # Or "resnet34", "resnet50", etc.
IMAGE_SIZE = 224 # ResNets pretrained on ImageNet often expect 224x224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Data Loading and Preprocessing ---

# Define transformations
# Using standard ImageNet normalization as ResNet is pretrained on it
transform = transforms.Compose([
    transforms.Grayscale(3),  # Convert FashionMNIST grayscale to 3 channels
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), # Resize to match ResNet input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
])

# Load FashionMNIST dataset
try:
    dataset_train = datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
    dataset_test = datasets.FashionMNIST(root="./data", train=False, transform=transform, download=True)
except Exception as e:
    print(f"Error downloading or loading dataset: {e}")
    print("Please check your internet connection or file permissions.")
    exit()

# Create DataLoaders
dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

print(f"Training dataset size: {len(dataset_train)}")
print(f"Test dataset size: {len(dataset_test)}")
print(f"Number of training batches: {len(dataloader_train)}")
print(f"Number of test batches: {len(dataloader_test)}")


# --- ResNet Model Definition ---
class ResNetModel(nn.Module):
    def __init__(self, model_name=MODEL_NAME, num_classes=10, pretrained=True):
        super(ResNetModel, self).__init__()
        self.model = create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

# --- Training and Evaluation Function ---
def train_and_evaluate(optimizer_name):
    print(f"\n--- Training with {optimizer_name} ---")
    model = ResNetModel(num_classes=10).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    # Define optimizer based on name
    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    elif optimizer_name == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=0.01)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Enable mixed precision if using CUDA
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # --- Training Loop ---
    train_losses, train_acc = [], []
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, (images, labels) in enumerate(dataloader_train):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            # Mixed precision context
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Scales loss. Calls backward() on scaled loss to create scaled gradients.
            scaler.scale(loss).backward()

            # Unscales gradients and calls optimizer.step()
            scaler.step(optimizer)

            # Updates the scale for next iteration.
            scaler.update()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # Print progress
            if (i + 1) % 100 == 0:
                 print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(dataloader_train)}], Loss: {loss.item():.4f}')


        epoch_loss = running_loss / len(dataloader_train)
        epoch_acc = correct_train / total_train
        train_losses.append(epoch_loss)
        train_acc.append(epoch_acc)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} Summary:")
        print(f"  Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.4f}")

    training_time = time.time() - start_time
    print(f"Training finished in {training_time:.2f} seconds.")

    # --- Testing Loop ---
    model.eval()
    correct_test, total_test, test_loss = 0, 0, 0.0
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in dataloader_test:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                 outputs = model(images)
                 loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    final_test_accuracy = correct_test / total_test
    final_test_loss = test_loss / len(dataloader_test)
    print(f"Test Loss: {final_test_loss:.4f}, Test Accuracy: {final_test_accuracy:.4f}")

    # --- Plotting ---
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, marker='o', label="Training Loss")
    # Optionally plot test loss if calculated per epoch
    # plt.plot(range(1, NUM_EPOCHS + 1), test_losses, marker='o', label="Test Loss") # Needs test loss calculation within epoch loop
    plt.title(f"{optimizer_name} - Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, NUM_EPOCHS + 1), train_acc, marker='o', label="Training Accuracy")
    # Optionally plot test accuracy if calculated per epoch
    # plt.plot(range(1, NUM_EPOCHS + 1), test_acc, marker='o', label="Test Accuracy") # Needs test acc calculation within epoch loop
    plt.title(f"{optimizer_name} - Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05) # Set y-axis limits for accuracy
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    class_names = dataset_train.classes # Get class names from the dataset
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{optimizer_name} - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    return optimizer_name, final_test_accuracy, final_test_loss, training_time

# --- Main Execution Block ---
results = []
optimizers_to_test = ["SGD", "Adam", "RMSprop", "Adagrad"]

for opt_name in optimizers_to_test:
    try:
        result = train_and_evaluate(opt_name)
        results.append(result)
    except Exception as e:
        print(f"Error occurred during training/evaluation with {opt_name}: {e}")
        # Optionally add placeholder results or skip
        # results.append((opt_name, float('nan'), float('nan'), float('nan')))
        continue # Continue to the next optimizer


# --- Print Summary Table ---
if results:
    results_df = pd.DataFrame(results, columns=["Optimizer", "Test Accuracy", "Test Loss", "Training Time (s)"])
    print("\n--- Summary Results ---")
    print(results_df)
else:
    print("\nNo results generated.")