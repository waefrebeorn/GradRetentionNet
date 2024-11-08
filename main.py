# main.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from EnhancedSGD import EnhancedSGD  # Ensure this module is in your project directory
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
import time
import psutil  # For RAM tracking
import gc  # Garbage collector to free memory when necessary
import numpy as np
import cv2  # For image processing in Grad-CAM
import json  # For saving results
import random  # For random sample selection
from PIL import Image

# ----------------------- Device Configuration -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Using device: {device}")

# ----------------------- Directory Setup -----------------------
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)
gradcam_dir = os.path.join(output_dir, "gradcam_images")
os.makedirs(gradcam_dir, exist_ok=True)

# ----------------------- VOCSegmentationWithTransform -----------------------
class VOCSegmentationWithTransform(datasets.VOCSegmentation):
    def __init__(self, root, year="2012", image_set="train", transform_image=None, transform_mask=None, download=False):
        super().__init__(root=root, year=year, image_set=image_set, download=download)
        self.transform_image = transform_image
        self.transform_mask = transform_mask

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        if self.transform_image:
            image = self.transform_image(image)
        if self.transform_mask:
            # For segmentation masks, use nearest interpolation to preserve class indices
            target = self.transform_mask(target)
            # Convert mask to LongTensor without normalization
            target = torch.as_tensor(np.array(target), dtype=torch.long)
        return image, target

# ----------------------- Memory Tracking -----------------------
def track_memory():
    if torch.cuda.is_available():
        vram = torch.cuda.memory_allocated(device) / (1024 ** 2)  # Convert to MB
    else:
        vram = 0
    ram = psutil.virtual_memory().used / (1024 ** 2)  # Convert to MB
    return vram, ram

# ----------------------- Time Tracking -----------------------
def time_epoch(start_time):
    elapsed = time.time() - start_time
    print(f"Epoch duration: {elapsed:.2f} seconds")
    return elapsed

# ----------------------- Models Definition -----------------------
class SimpleCNN_MNIST(nn.Module):
    def __init__(self):
        super(SimpleCNN_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # [batch, 32, 14, 14]
        x = self.pool(torch.relu(self.conv2(x)))  # [batch, 64, 7, 7]
        x = x.view(-1, 64 * 7 * 7)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SimpleCNN_CIFAR10(nn.Module):
    def __init__(self):
        super(SimpleCNN_CIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # [batch, 32, 16, 16]
        x = self.pool(torch.relu(self.conv2(x)))  # [batch, 64, 8, 8]
        x = x.view(-1, 64 * 8 * 8)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TextClassifier(nn.Module):
    def __init__(self, num_class):
        super(TextClassifier, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_class)

    def forward(self, input_ids, attention_mask):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask).logits

class SimpleUNet(nn.Module):
    """
    Simple U-Net model for image segmentation tasks.
    """
    def __init__(self, num_classes=21):  # Pascal VOC has 21 classes
        super(SimpleUNet, self).__init__()
        # Using FCN ResNet50 for segmentation
        self.base_model = models.segmentation.fcn_resnet50(weights=None, num_classes=num_classes)

    def forward(self, x):
        output = self.base_model(x)
        if isinstance(output, dict) and 'out' in output:
            return output['out']
        return output

# ----------------------- Dataset Utilities -----------------------
def dataset_exists(dataset_name):
    data_path = "./data"
    if dataset_name == "MNIST":
        return os.path.exists(os.path.join(data_path, "MNIST"))
    elif dataset_name == "CIFAR10":
        return os.path.exists(os.path.join(data_path, "cifar-10-batches-py"))
    elif dataset_name in ["IMDB", "AG_NEWS"]:
        return os.path.exists(os.path.join(data_path, f"{dataset_name.lower()}_train.csv")) and \
               os.path.exists(os.path.join(data_path, f"{dataset_name.lower()}_test.csv"))
    elif dataset_name == "VOC":
        return os.path.exists(os.path.join(data_path, "VOCdevkit", "VOC2012"))
    return False

from datasets import load_dataset

import pandas as pd

def load_dataset(dataset_name):
    data_path = "./data"
    if dataset_name.lower() == "mnist":
        if not dataset_exists("MNIST"):
            print("MNIST dataset not found. Downloading...")
            datasets.MNIST(root=data_path, train=True, download=True)
            datasets.MNIST(root=data_path, train=False, download=True)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(root=data_path, train=True, transform=transform, download=False)
        test_dataset = datasets.MNIST(root=data_path, train=False, transform=transform, download=False)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True if torch.cuda.is_available() else False)
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2, pin_memory=True if torch.cuda.is_available() else False)
        model = SimpleCNN_MNIST().to(device)

    elif dataset_name.lower() == "cifar10":
        if not dataset_exists("CIFAR10"):
            print("CIFAR10 dataset not found. Downloading...")
            datasets.CIFAR10(root=data_path, train=True, download=True)
            datasets.CIFAR10(root=data_path, train=False, download=True)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = datasets.CIFAR10(root=data_path, train=True, transform=transform, download=False)
        test_dataset = datasets.CIFAR10(root=data_path, train=False, transform=transform, download=False)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True if torch.cuda.is_available() else False)
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2, pin_memory=True if torch.cuda.is_available() else False)
        model = SimpleCNN_CIFAR10().to(device)

    elif dataset_name.lower() in ["imdb", "ag_news"]:
        csv_train_path = os.path.join(data_path, f"{dataset_name.lower()}_train.csv")
        csv_test_path = os.path.join(data_path, f"{dataset_name.lower()}_test.csv")

        if os.path.exists(csv_train_path) and os.path.exists(csv_test_path):
            print(f"Loading {dataset_name.upper()} dataset from CSV files.")
            train_df = pd.read_csv(csv_train_path)
            test_df = pd.read_csv(csv_test_path)
        else:
            print(f"{dataset_name.upper()} dataset not found in CSV. Downloading from Hugging Face.")
            dataset = load_dataset(dataset_name.lower())
            train_df, test_df = dataset["train"].to_pandas(), dataset["test"].to_pandas()

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        def tokenize_function(texts):
            return tokenizer(texts, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

        train_texts = train_df["text"].tolist()
        test_texts = test_df["text"].tolist()
        train_labels = train_df["label"].tolist()
        test_labels = test_df["label"].tolist()

        train_encodings = tokenize_function(train_texts)
        test_encodings = tokenize_function(test_texts)

        train_encodings["labels"] = torch.tensor(train_labels)
        test_encodings["labels"] = torch.tensor(test_labels)

        train_dataset = torch.utils.data.TensorDataset(train_encodings["input_ids"], train_encodings["attention_mask"], train_encodings["labels"])
        test_dataset = torch.utils.data.TensorDataset(test_encodings["input_ids"], test_encodings["attention_mask"], test_encodings["labels"])

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True if torch.cuda.is_available() else False)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True if torch.cuda.is_available() else False)

        num_class = len(set(train_labels))
        model = TextClassifier(num_class).to(device)

    elif dataset_name.lower() == "voc":
        if not dataset_exists("VOC"):
            print("Pascal VOC dataset not found. Downloading...")
            datasets.VOCSegmentation(root=data_path, year="2012", image_set="train", download=True)
            datasets.VOCSegmentation(root=data_path, year="2012", image_set="val", download=True)

        transform_image = transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        transform_mask = transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

        train_dataset = VOCSegmentationWithTransform(root=data_path, year="2012", image_set="train", download=False, transform_image=transform_image, transform_mask=transform_mask)
        test_dataset = VOCSegmentationWithTransform(root=data_path, year="2012", image_set="val", download=False, transform_image=transform_image, transform_mask=transform_mask)

        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True if torch.cuda.is_available() else False)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True if torch.cuda.is_available() else False)
        model = SimpleUNet(num_classes=21).to(device)
    else:
        raise ValueError("Unsupported dataset. Choose from 'MNIST', 'CIFAR10', 'IMDB', 'AG_NEWS', 'VOC'.")

    return model, train_loader, test_loader, test_dataset

# ----------------------- Grad-CAM Implementation -----------------------
class GradCAM:
    """
    Grad-CAM implementation for visualizing model focus on image-based tasks.
    """
    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer or self._find_target_layer()
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _find_target_layer(self):
        # Automatically choose the last Conv layer if not specified
        for name, module in reversed(list(self.model.named_modules())):
            if isinstance(module, nn.Conv2d):
                return name
        raise ValueError("No suitable Conv2d layer found for Grad-CAM.")

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        layer = dict([*self.model.named_modules()]).get(self.target_layer, None)
        if layer is None:
            raise ValueError(f"Layer '{self.target_layer}' not found in the model.")
        self.hook_handles.append(layer.register_forward_hook(forward_hook))
        self.hook_handles.append(layer.register_full_backward_hook(backward_hook))  # Updated to register_full_backward_hook

    def generate_heatmap(self, input_image, target_class):
        self.model.zero_grad()
        output = self.model(input_image)
        if isinstance(output, dict):  # For segmentation models
            output = output['out']
            # Aggregate output for multi-class segmentation
            # For simplicity, take mean over spatial dimensions
            output = output.mean(dim=(2, 3))  # [batch, num_classes]
        loss = output[:, target_class].sum()
        loss.backward()

        if self.gradients is None or self.activations is None:
            raise ValueError("Gradients or activations not captured. Check hooks.")

        gradients = self.gradients.cpu().numpy()[0]  # [C, H, W]
        activations = self.activations.cpu().numpy()[0]  # [C, H, W]

        weights = np.mean(gradients, axis=(1, 2))  # [C]
        heatmap = np.dot(weights, activations.reshape((activations.shape[0], -1)))  # [H*W]
        heatmap = heatmap.reshape(activations.shape[1], activations.shape[2])  # [H, W]
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1

        return heatmap

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

def apply_heatmap_on_image(image, heatmap):
    """
    Overlays the heatmap on the original image.

    Args:
        image (torch.Tensor): Original image tensor.
        heatmap (numpy.ndarray): Generated heatmap.

    Returns:
        numpy.ndarray: Image with heatmap overlay.
    """
    image = image.cpu().numpy().transpose(1, 2, 0)
    image = np.clip(image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlayed = heatmap * 0.4 + np.uint8(255 * image) * 0.6
    return overlayed

# ----------------------- Training and Evaluation -----------------------
def train_and_evaluate(model, optimizer, train_loader, test_loader, num_epochs=5, log_interval=50, optimizer_name="", dataset_name=""):
    model.to(device)
    model.train()
    train_losses, test_accuracies, epoch_times, memory_usage = [], [], [], []

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        vram_start, ram_start = track_memory()
        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # Adjust target for segmentation
            if dataset_name.lower() == "voc" and isinstance(model, SimpleUNet):
                target = target.squeeze(1).long()  # Remove channel dimension for segmentation

            optimizer.zero_grad()
            output = model(data)

            # Compute loss
            try:
                if dataset_name.lower() == "voc" and isinstance(model, SimpleUNet):
                    if isinstance(output, torch.Tensor):
                        output_tensor = output
                    else:
                        raise TypeError(f"Expected tensor output for VOC, got {type(output)}")

                    assert isinstance(output_tensor, torch.Tensor), f"Expected tensor, got {type(output_tensor)}"
                    assert output_tensor.dim() == 4, f"Expected 4D output tensor, got {output_tensor.dim()}D"
                    assert output_tensor.shape[1] == 21, f"Expected 21 classes for VOC, got {output_tensor.shape[1]}"
                    assert output_tensor.shape[2:] == target.shape[1:], (
                        f"Output spatial dimensions {output_tensor.shape[2:]} do not match target {target.shape[1:]}"
                    )
                    assert target.dtype == torch.long, f"Expected target dtype torch.long, got {target.dtype}"

                    loss = nn.CrossEntropyLoss()(output_tensor, target)
                else:
                    if not isinstance(output, torch.Tensor):
                        raise TypeError(f"Expected tensor output, got {type(output)}")
                    loss = nn.CrossEntropyLoss()(output, target)

                loss_value = loss.item()
                running_loss += loss_value

                # Backpropagation and optimization step
                loss.backward()
                optimizer.step()

            except Exception as e:
                print(f"Error during loss computation or backpropagation: {e}")
                if 'str' in str(e).lower():
                    print("[Error Trace] A 'str' type was detected in tensor operations.")
                continue

            # Print batch details every 'log_interval' batches
            if batch_idx % log_interval == 0 or batch_idx == len(train_loader) - 1:
                print(f"Epoch [{epoch}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], "
                      f"Batch Loss: {loss_value:.4f}")

        # Calculate average loss and log test accuracy
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        test_accuracy = test_epoch(model, test_loader, dataset_name)
        test_accuracies.append(test_accuracy)

        # Track memory and time
        vram_end, ram_end = track_memory()
        epoch_time = time_epoch(start_time)
        epoch_times.append(epoch_time)
        memory_usage.append((vram_end - vram_start, ram_end - ram_start))

        # Print epoch summary
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        print(f"Memory Usage - VRAM Change: {vram_end - vram_start:.2f} MB, RAM Change: {ram_end - ram_start:.2f} MB")

    return train_losses, test_accuracies, epoch_times, memory_usage

def test_epoch(model, test_loader, dataset_name="VOC", log_interval=50):
    model.eval()
    total_correct = 0
    total_elements = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            try:
                if isinstance(model, TextClassifier):
                    # Text classification handling
                    inputs = {
                        "input_ids": batch["input_ids"].to(device),
                        "attention_mask": batch["attention_mask"].to(device)
                    }
                    targets = batch["label"].to(device)
                    output = model(**inputs)
                    preds = output.argmax(dim=1)
                    total_correct += (preds == targets).sum().item()
                    total_elements += targets.size(0)
                else:
                    # For image/segmentation datasets
                    data, target = batch
                    data, target = data.to(device), target.to(device)

                    if dataset_name.lower() == "voc" and isinstance(model, SimpleUNet):
                        # Handle per-pixel accuracy for segmentation tasks
                        target = target.squeeze(1).long()  # Ensure target has correct shape and dtype
                        output = model(data)

                        # Ensure output is a tensor and has correct shape
                        if not isinstance(output, torch.Tensor):
                            raise TypeError(f"Expected tensor output, got {type(output)}")
                        assert output.shape[1] == 21, f"Expected 21 classes, got {output.shape[1]}"
                        assert output.shape[2:] == target.shape[1:], (
                            f"Output spatial dimensions {output.shape[2:]} do not match target {target.shape[1:]}"
                        )
                        assert target.dtype == torch.long, f"Expected target dtype torch.long, got {target.dtype}"

                        preds = output.argmax(dim=1)  # Get per-pixel class predictions

                        # Calculate per-pixel accuracy
                        total_correct += (preds == target).sum().item()
                        total_elements += target.numel()  # Count total pixels
                    else:
                        # For other tasks, assume standard classification
                        output = model(data)
                        preds = output.argmax(dim=1)
                        total_correct += (preds == target).sum().item()
                        total_elements += target.size(0)

                # Print batch summary every 'log_interval' batches
                if batch_idx % log_interval == 0 or batch_idx == len(test_loader) - 1:
                    print(f"Batch [{batch_idx}/{len(test_loader)}] processed, Total Correct: {total_correct}, Total Elements: {total_elements}")

            except Exception as e:
                print(f"Error during batch processing in test_epoch: {e}")
                if 'str' in str(e).lower():
                    print("[Error Trace] A 'str' type was detected in tensor operations during testing.")
                continue

    model.train()
    accuracy = total_correct / total_elements if total_elements > 0 else 0
    print(f"Testing accuracy for {dataset_name}: {accuracy:.4f}")
    return accuracy

# ----------------------- Plotting Results -----------------------
def plot_results(num_epochs, results, dataset_name):
    """
    Plots the training loss, test accuracy, learning rate, gradient variance,
    epoch times, and memory usage for each optimizer.

    Parameters:
        num_epochs (int): Number of epochs.
        results (dict): Dictionary containing lists of metrics for each optimizer.
        dataset_name (str): Name of the dataset being plotted.
    """
    plt.figure(figsize=(20, 15))

    # Training Loss Comparison
    plt.subplot(3, 2, 1)
    for optimizer_name, data in results.items():
        if 'loss' in data:
            plt.plot(range(1, num_epochs + 1), data['loss'], label=optimizer_name)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title(f"{dataset_name} - Training Loss Comparison")
    plt.legend()

    # Test Accuracy Comparison
    plt.subplot(3, 2, 2)
    for optimizer_name, data in results.items():
        if 'accuracy' in data:
            plt.plot(range(1, num_epochs + 1), data['accuracy'], label=optimizer_name)
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title(f"{dataset_name} - Test Accuracy Comparison")
    plt.legend()

    # Learning Rate over Batches (EnhancedSGD only)
    plt.subplot(3, 2, 3)
    for optimizer_name, data in results.items():
        if 'learning_rate' in data and optimizer_name == "EnhancedSGD":
            plt.plot(range(1, len(data['learning_rate']) + 1), data['learning_rate'], label=optimizer_name)
    plt.xlabel("Batch")
    plt.ylabel("Learning Rate")
    plt.title(f"{dataset_name} - Learning Rate over Batches (EnhancedSGD)")
    plt.legend()

    # Gradient Variance over Batches (EnhancedSGD only)
    plt.subplot(3, 2, 4)
    for optimizer_name, data in results.items():
        if 'gradient_variance' in data and optimizer_name == "EnhancedSGD":
            plt.plot(range(1, len(data['gradient_variance']) + 1), data['gradient_variance'], label=optimizer_name)
    plt.xlabel("Batch")
    plt.ylabel("Gradient Variance")
    plt.title(f"{dataset_name} - Gradient Variance over Batches (EnhancedSGD)")
    plt.legend()

    # Training Time per Epoch
    plt.subplot(3, 2, 5)
    for optimizer_name, data in results.items():
        if 'epoch_time' in data:
            plt.plot(range(1, num_epochs + 1), data['epoch_time'], label=optimizer_name)
    plt.xlabel("Epoch")
    plt.ylabel("Epoch Time (s)")
    plt.title(f"{dataset_name} - Training Time per Epoch")
    plt.legend()

    # Memory Usage per Epoch
    plt.subplot(3, 2, 6)
    for optimizer_name, data in results.items():
        if 'memory_usage' in data:
            vram = [m[0] for m in data['memory_usage']]
            ram = [m[1] for m in data['memory_usage']]
            plt.plot(range(1, num_epochs + 1), vram, label=f"{optimizer_name} - VRAM (MB)")
            plt.plot(range(1, num_epochs + 1), ram, label=f"{optimizer_name} - RAM (MB)")
    plt.xlabel("Epoch")
    plt.ylabel("Memory Usage (MB)")
    plt.title(f"{dataset_name} - Memory Usage per Epoch")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_metrics.png"))
    plt.close()
    print(f"Saved metrics plot to {os.path.join(output_dir, f'{dataset_name}_metrics.png')}")

# ----------------------- Main Function -----------------------
def main():
    # Set seeds for reproducibility
    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    set_seed()

    # Choose number of epochs
    try:
        num_epochs = int(input("Enter the number of epochs for testing each optimizer: "))
    except ValueError:
        print("Invalid input. Using default of 10 epochs.")
        num_epochs = 10

    # Choose dataset or 'all' for multiple datasets
    dataset_input = input("Choose a dataset (MNIST, CIFAR10, IMDB, AG_NEWS, VOC) or type 'all' for all datasets: ").strip()
    run_all = dataset_input.lower() == 'all'
    if run_all:
        dataset_list = ["MNIST", "CIFAR10", "IMDB", "AG_NEWS", "VOC"]
    else:
        valid_datasets = ["MNIST", "CIFAR10", "IMDB", "AG_NEWS", "VOC"]
        if dataset_input not in valid_datasets:
            print("Invalid dataset choice. Exiting.")
            return
        dataset_list = [dataset_input]

    usage_case = "GenAI"  # Adjust based on your use case

    # Define optimizers to test
    optimizers = {
        "EnhancedSGD": EnhancedSGD,
        "SGD": optim.SGD,
        "AdamW": optim.AdamW,
        "RMSprop": optim.RMSprop,
        "Adam": optim.Adam
    }

    # Initialize results dictionary
    results = {}

    # Iterate through each dataset and optimizer combination
    for dataset_name in dataset_list:
        print(f"\nPreparing dataset: {dataset_name}")

        for opt_name, opt_class in optimizers.items():
            print(f"\nTraining with {opt_name} optimizer on {dataset_name} dataset...")

            # Reload the dataset and model for each optimizer to avoid uninitialized variables
            try:
                model, train_loader, test_loader, test_dataset = load_dataset(dataset_name)
            except Exception as e:
                print(f"Error loading dataset {dataset_name}: {e}")
                continue

            # Initialize optimizer
            try:
                if opt_name == "EnhancedSGD":
                    optimizer = opt_class(
                        model.parameters(),
                        lr=0.01,
                        model=model,
                        usage_case=usage_case,
                        use_amp=True,
                        lookahead_k=5,
                        lookahead_alpha=0.5,
                        apply_noise=True,
                        adaptive_momentum=True,
                        gradient_centering=True
                    )
                else:
                    if dataset_name in ["IMDB", "AG_NEWS"]:
                        optimizer = opt_class(model.parameters(), lr=0.01)
                    elif dataset_name in ["MNIST", "CIFAR10"]:
                        if opt_name == "SGD":
                            optimizer = opt_class(model.parameters(), lr=0.01, momentum=0.9)
                        else:
                            optimizer = opt_class(model.parameters(), lr=0.001)
                    elif dataset_name == "VOC":
                        optimizer = opt_class(model.parameters(), lr=0.001)
                    else:
                        optimizer = opt_class(model.parameters(), lr=0.01)
            except Exception as e:
                print(f"Error initializing optimizer {opt_name} for {dataset_name}: {e}")
                # Clean up before continuing
                del model, train_loader, test_loader, optimizer
                gc.collect()
                torch.cuda.empty_cache()
                continue

            # Initialize Grad-CAM if applicable
            grad_cam = None
            if dataset_name.lower() in ["cifar10", "voc"]:
                # Determine target layer based on model type
                if dataset_name.lower() == "voc":
                    # For FCN ResNet50, let's assume 'base_model.classifier.4' is the last conv layer
                    target_layer = 'base_model.classifier.4'
                else:
                    # For SimpleCNN_CIFAR10, assuming 'conv2' is the target layer
                    target_layer = 'conv2'
                try:
                    grad_cam = GradCAM(model, target_layer)
                except ValueError as e:
                    print(f"Grad-CAM initialization error: {e}")
                    grad_cam = None

            # Train and evaluate
            try:
                train_losses, test_accuracies, epoch_times, memory_usage = train_and_evaluate(
                    model, optimizer, train_loader, test_loader, num_epochs=num_epochs,
                    log_interval=50, optimizer_name=opt_name, dataset_name=dataset_name
                )
            except Exception as e:
                print(f"Error during training with {opt_name} on {dataset_name}: {e}")
                # Clean up before continuing
                del model, train_loader, test_loader, optimizer, grad_cam
                gc.collect()
                torch.cuda.empty_cache()
                continue

            # Store results
            results_key = f"{opt_name}_{dataset_name}"
            results[results_key] = {
                "loss": train_losses,
                "accuracy": test_accuracies,
                "epoch_time": epoch_times,
                "memory_usage": memory_usage
            }

            # Remove Grad-CAM hooks if initialized
            if grad_cam:
                grad_cam.remove_hooks()

            # Clean up memory
            del model, train_loader, test_loader, optimizer, grad_cam
            gc.collect()
            torch.cuda.empty_cache()

    # Save all results to a JSON file
    results_path = os.path.join(output_dir, "all_results.json")
    try:
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\nSaved all results to {results_path}")
    except Exception as e:
        print(f"Error saving results to JSON: {e}")

    # Plotting results for each dataset
    for dataset_name in dataset_list:
        # Extract relevant keys
        relevant_keys = [key for key in results.keys() if key.endswith(f"_{dataset_name}")]
        if not relevant_keys:
            print(f"No results to plot for dataset: {dataset_name}")
            continue
        dataset_results = {key.split('_')[0]: results[key] for key in relevant_keys}
        plot_results(num_epochs, dataset_results, dataset_name)

    print("\nAll training and evaluations completed.")
    print("Program finished. Press any key to exit.")

if __name__ == '__main__':
    main()
