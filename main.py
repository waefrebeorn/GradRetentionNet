import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from EnhancedSGD import EnhancedSGD
import time

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define models for MNIST and CIFAR-10
class SimpleCNN_MNIST(nn.Module):
    def __init__(self):
        super(SimpleCNN_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
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
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_dataset(dataset_name):
    if dataset_name.lower() == "mnist":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True if torch.cuda.is_available() else False)
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=0, pin_memory=True if torch.cuda.is_available() else False)
        model = SimpleCNN_MNIST().to(device)
    elif dataset_name.lower() == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # mean and std for CIFAR-10
        ])
        train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True if torch.cuda.is_available() else False)
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=0, pin_memory=True if torch.cuda.is_available() else False)
        model = SimpleCNN_CIFAR10().to(device)
    else:
        raise ValueError("Unsupported dataset. Choose 'MNIST' or 'CIFAR10'.")
    
    return model, train_loader, test_loader

def train_and_evaluate(model, optimizer, train_loader, test_loader, num_epochs=5, optimizer_name=""):
    model.to(device)
    model.train()
    train_losses, test_accuracies, learning_rates, gradient_variances = [], [], [], []

    for epoch in range(num_epochs):
        running_loss = 0.0
        print(f"\nStarting Epoch {epoch + 1}/{num_epochs} for {optimizer_name} optimizer")

        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to GPU
            data, target = data.to(device), target.to(device)
            
            # Zero gradients, forward pass, loss computation, backpropagation, and optimization step
            optimizer.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()
            
            # Track training loss
            running_loss += loss.item()
            batch_loss = loss.item()

            # Collect gradient variance and learning rate if available
            grad_var = getattr(optimizer, 'grad_var', None)  # Only available in EnhancedSGD
            lr = optimizer.param_groups[0]["lr"]
            if grad_var is not None:  # Only track for EnhancedSGD
                gradient_variances.append(grad_var)
                learning_rates.append(lr)

            # Print batch data for tracking
            if batch_idx % 50 == 0 or batch_idx == len(train_loader) - 1:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], "
                      f"Batch Loss: {batch_loss:.4f}, Learning Rate: {lr:.4e}, Gradient Variance: {grad_var}")

        # Calculate average loss and log test accuracy
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        test_accuracy = test_epoch(model, test_loader)
        test_accuracies.append(test_accuracy)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    return train_losses, test_accuracies, learning_rates, gradient_variances

def test_epoch(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return correct / len(test_loader.dataset)

def main():
    # Choose dataset
    dataset_name = input("Choose a dataset (MNIST or CIFAR10): ")
    print(f"Using {dataset_name} dataset.")
    
    # Train and evaluate with EnhancedSGD
    model, train_loader, test_loader = load_dataset(dataset_name)
    optimizer_enhanced_sgd = EnhancedSGD(model.parameters(), lr=0.01, model=model, usage_case="GenAI", use_amp=True, lookahead_k=5, lookahead_alpha=0.5)
    print("\nTraining with EnhancedSGD optimizer...")
    loss_enhanced_sgd, acc_enhanced_sgd, lr_enhanced_sgd, var_enhanced_sgd = train_and_evaluate(model, optimizer_enhanced_sgd, train_loader, test_loader, num_epochs=25, optimizer_name="EnhancedSGD")
    
    # Train and evaluate with standard SGD
    model, train_loader, test_loader = load_dataset(dataset_name)
    optimizer_sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    print("\nTraining with SGD optimizer...")
    loss_sgd, acc_sgd, _, _ = train_and_evaluate(model, optimizer_sgd, train_loader, test_loader, num_epochs=25, optimizer_name="SGD")
    
    # Train and evaluate with AdamW optimizer
    model, train_loader, test_loader = load_dataset(dataset_name)
    optimizer_adamw = optim.AdamW(model.parameters(), lr=0.001)
    print("\nTraining with AdamW optimizer...")
    loss_adamw, acc_adamw, _, _ = train_and_evaluate(model, optimizer_adamw, train_loader, test_loader, num_epochs=25, optimizer_name="AdamW")

    # Plot results
    plt.figure(figsize=(12, 10))

    # Training Loss Comparison
    plt.subplot(2, 2, 1)
    plt.plot(range(1, 26), loss_enhanced_sgd, label="EnhancedSGD")
    plt.plot(range(1, 26), loss_sgd, label="SGD")
    plt.plot(range(1, 26), loss_adamw, label="AdamW")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Comparison")
    plt.legend()

    # Test Accuracy Comparison
    plt.subplot(2, 2, 2)
    plt.plot(range(1, 26), acc_enhanced_sgd, label="EnhancedSGD")
    plt.plot(range(1, 26), acc_sgd, label="SGD")
    plt.plot(range(1, 26), acc_adamw, label="AdamW")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy Comparison")
    plt.legend()

    # Learning Rate over Epochs (EnhancedSGD only)
    plt.subplot(2, 2, 3)
    plt.plot(range(1, len(lr_enhanced_sgd) + 1), lr_enhanced_sgd, label="Learning Rate")
    plt.xlabel("Batch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate over Epochs (EnhancedSGD)")

    # Gradient Variance over Epochs (EnhancedSGD only)
    plt.subplot(2, 2, 4)
    plt.plot(range(1, len(var_enhanced_sgd) + 1), var_enhanced_sgd, label="Gradient Variance")
    plt.xlabel("Batch")
    plt.ylabel("Gradient Variance")
    plt.title("Gradient Variance over Epochs (EnhancedSGD)")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
