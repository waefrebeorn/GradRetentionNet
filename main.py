import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# Custom optimizer with persistent gradient accumulation based on Kalomaze's idea
class PersistentSGD(optim.Optimizer):
    def __init__(self, params, lr=0.01, decay=0.99):
        defaults = dict(lr=lr, decay=decay)
        super(PersistentSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            decay = group['decay']
            for p in group['params']:
                if p.grad is None:
                    continue

                # Create state for persistent gradient if it doesn't exist
                state = self.state[p]
                if 'persistent_grad' not in state:
                    state['persistent_grad'] = torch.zeros_like(p.data)

                # Accumulate gradients using decay factor instead of zeroing out
                persistent_grad = state['persistent_grad']
                persistent_grad.mul_(decay).add_(p.grad.data)

                # Update weights using the accumulated persistent gradient
                p.data.add_(-group['lr'], persistent_grad)

        return loss


# Simple neural network model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input tensor
        x = torch.relu(self.fc1(x))  # Apply ReLU activation
        x = self.fc2(x)  # Output layer
        return x


# Training function
def train(model, device, train_loader, optimizer, epoch, log_interval=100):
    model.train()
    criterion = nn.CrossEntropyLoss()
    epoch_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # Clear existing gradients
        output = model(data)  # Forward pass
        loss = criterion(output, target)  # Calculate loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Optimize
        epoch_loss += loss.item()
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    return epoch_loss / len(train_loader)  # Return average epoch loss


# Validation function
def validate(model, device, val_loader):
    model.eval()
    val_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    accuracy = 100. * correct / len(val_loader.dataset)
    print(f'\nValidation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(val_loader.dataset)} '
          f'({accuracy:.0f}%)\n')
    return val_loss, accuracy  # Return validation loss and accuracy


# Testing function
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.0f}%)\n')
    return test_loss, accuracy  # Return test loss and accuracy


# Plotting function
def plot_losses(epochs, train_losses, val_losses, labels, title):
    """Plot the training and validation loss for different optimizers."""
    for i, loss in enumerate(train_losses):
        plt.plot(range(1, epochs + 1), loss, label=f'Train {labels[i]}')
    for i, loss in enumerate(val_losses):
        plt.plot(range(1, epochs + 1), loss, linestyle='--', label=f'Val {labels[i]}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


# Main script
def main():
    # Hyperparameters and setup
    batch_size = 64
    epochs = 10
    learning_rate = 0.01
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Data transformation and loaders
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    full_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # Split the training set into training and validation sets
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform),
                             batch_size=1000, shuffle=False)

    # Initialize models and optimizers
    model_persistent = SimpleNet().to(device)
    model_adamw = SimpleNet().to(device)
    optimizer_persistent = PersistentSGD(model_persistent.parameters(), lr=learning_rate)
    optimizer_adamw = optim.AdamW(model_adamw.parameters(), lr=learning_rate)

    # Track losses for plotting
    persistent_train_losses, persistent_val_losses = [], []
    adamw_train_losses, adamw_val_losses = [], []

    # Train models for all epochs and validate
    for epoch in range(1, epochs + 1):
        train_loss_persistent = train(model_persistent, device, train_loader, optimizer_persistent, epoch)
        val_loss_persistent, _ = validate(model_persistent, device, val_loader)

        train_loss_adamw = train(model_adamw, device, train_loader, optimizer_adamw, epoch)
        val_loss_adamw, _ = validate(model_adamw, device, val_loader)

        persistent_train_losses.append(train_loss_persistent)
        persistent_val_losses.append(val_loss_persistent)

        adamw_train_losses.append(train_loss_adamw)
        adamw_val_losses.append(val_loss_adamw)

    # Plot the training and validation loss curves for comparison
    plot_losses(epochs, [persistent_train_losses, adamw_train_losses],
                [persistent_val_losses, adamw_val_losses], ['PersistentSGD', 'AdamW'], 'Training & Validation Loss Comparison')

    # Test the models and output results
    print("Testing PersistentSGD Model")
    test(model_persistent, device, test_loader)
    print("Testing AdamW Model")
    test(model_adamw, device, test_loader)


if __name__ == '__main__':
    main()
