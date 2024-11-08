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
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True if torch.cuda.is_available() else False)
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=0, pin_memory=True if torch.cuda.is_available() else False)
        model = SimpleCNN_CIFAR10().to(device)
    else:
        raise ValueError("Unsupported dataset. Choose 'MNIST' or 'CIFAR10'.")
    
    return model, train_loader, test_loader

def main():
    # Choose dataset
    dataset_name = input("Choose a dataset (MNIST or CIFAR10): ")
    model, train_loader, test_loader = load_dataset(dataset_name)
    print(f"Using {dataset_name} dataset.")

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Initialize optimizers
    optimizer_enhanced_sgd = EnhancedSGD(model.parameters(), lr=0.01, model=model, usage_case="GenAI", use_amp=True, lookahead_k=5, lookahead_alpha=0.5)
    optimizer_sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer_adamw = optim.AdamW(model.parameters(), lr=0.001)

    def train(model, optimizer, num_epochs=5):
        model.train()
        train_losses = []
        test_accuracies = []
        learning_rates = []
        gradient_variances = []

        # To store all batch data for LLM ingestion and tracking
        batch_data_log = []

        for epoch in range(num_epochs):
            running_loss = 0.0
            epoch_start_time = time.time()
            batch_epoch_log = []

            print(f"\nStarting Epoch {epoch + 1}/{num_epochs}")
            for batch_idx, (data, target) in enumerate(train_loader):
                # Move data to GPU
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                # Track loss and optimizer-specific metrics
                running_loss += loss.item()
                batch_loss = loss.item()

                # Log gradient variance and learning rate if available
                grad_var = optimizer.grad_var if hasattr(optimizer, "grad_var") else None
                lr = optimizer.param_groups[0]["lr"]
                gradient_variances.append(grad_var)
                learning_rates.append(lr)

                # Print batch data and store for tracking
                if batch_idx % 50 == 0 or batch_idx == len(train_loader) - 1:
                    batch_info = {
                        "epoch": epoch + 1,
                        "batch_idx": batch_idx,
                        "batch_loss": batch_loss,
                        "learning_rate": lr,
                        "gradient_variance": grad_var
                    }
                    batch_epoch_log.append(batch_info)
                    print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], "
                          f"Batch Loss: {batch_loss:.4f}, Learning Rate: {lr:.4e}, Gradient Variance: {grad_var}")

            epoch_loss = running_loss / len(train_loader)
            train_losses.append(epoch_loss)

            # Calculate and log test accuracy at each epoch
            test_accuracy = test_epoch(model)
            test_accuracies.append(test_accuracy)
            epoch_duration = time.time() - epoch_start_time
            print(f"Epoch [{epoch + 1}/{num_epochs}] completed in {epoch_duration:.2f}s, Average Loss: {epoch_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
            
            # Append batch log data for this epoch
            batch_data_log.extend(batch_epoch_log)

        return train_losses, test_accuracies, learning_rates, gradient_variances, batch_data_log

    def test_epoch(model):
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        accuracy = correct / len(test_loader.dataset)
        return accuracy

    # Run training
    num_epochs = 25
    print(f"\nTraining with {dataset_name} dataset using EnhancedSGD optimizer...")
    loss_enhanced_sgd, acc_enhanced_sgd, lr_enhanced_sgd, var_enhanced_sgd, batch_data_enhanced_sgd = train(model, optimizer_enhanced_sgd, num_epochs=num_epochs)
    
    # Visualize results
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), loss_enhanced_sgd, label="EnhancedSGD - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss over Epochs")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
