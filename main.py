import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# Custom optimizer with ephemeral gradient application based on Kalomaze's idea
class EphemeralSGD(optim.Optimizer):
    def __init__(self, params, lr=0.01, decay=0.99):
        defaults = dict(lr=lr, decay=decay)
        super(EphemeralSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single ephemeral optimization step without permanently updating weights."""
        loss = None
        if closure is not None:
            loss = closure()

        # Save the original state of the parameters
        original_params = [p.data.clone() for group in self.param_groups for p in group['params'] if p.grad is not None]

        # Apply temporary updates using ephemeral gradients
        for group in self.param_groups:
            decay = group['decay']
            for p, orig in zip(group['params'], original_params):
                if p.grad is None:
                    continue
                # Create state for persistent gradient if it doesn't exist
                state = self.state[p]
                if 'persistent_grad' not in state:
                    state['persistent_grad'] = torch.zeros_like(p.data)

                # Accumulate gradients using decay factor instead of zeroing out
                persistent_grad = state['persistent_grad']
                persistent_grad.mul_(decay).add_(p.grad.data)

                # Apply temporary update using persistent gradients
                p.data.add_(persistent_grad, alpha=-group['lr'])

        # Calculate the temporary loss and revert to the original parameter state
        loss = closure() if closure is not None else None
        for p, orig in zip([p for group in self.param_groups for p in group['params']], original_params):
            p.data.copy_(orig)  # Revert the weights to the original state

        return loss


# Simple neural network model for MNIST
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


# Training function for ephemeral updates
def train_ephemeral(model, device, train_loader, optimizer, epoch, log_interval=100):
    model.train()
    criterion = nn.CrossEntropyLoss()
    epoch_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # Clear existing gradients

        def closure():
            """Closure function to calculate the loss."""
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            return loss

        # Use ephemeral step for virtual loss calculation
        loss = optimizer.step(closure)
        epoch_loss += loss.item()

        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    return epoch_loss / len(train_loader)


# Standard training function for other optimizers
def train_standard(model, device, train_loader, optimizer, epoch, log_interval=100):
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
    return epoch_loss / len(train_loader)


# Validation function to assess model performance
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
    return val_loss, accuracy


# Testing function to measure model accuracy on the test set
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
    return test_loss, accuracy


# Plotting function to visualize training and validation loss
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


# Main script for comprehensive testing and comparison
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
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1000, shuffle=False)

    # Initialize models and optimizers
    model_ephemeral = SimpleNet().to(device)
    model_adamw = SimpleNet().to(device)
    model_sgd = SimpleNet().to(device)

    optimizer_ephemeral = EphemeralSGD(model_ephemeral.parameters(), lr=learning_rate)
    optimizer_adamw = optim.AdamW(model_adamw.parameters(), lr=learning_rate)
    optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=learning_rate)

    # Track losses for plotting
    ephemeral_train_losses, ephemeral_val_losses = [], []
    adamw_train_losses, adamw_val_losses = [], []
    sgd_train_losses, sgd_val_losses = [], []

    # Train and validate models for all epochs
    for epoch in range(1, epochs + 1):
        ephemeral_train_loss = train_ephemeral(model_ephemeral, device, train_loader, optimizer_ephemeral, epoch)
        val_loss_ephemeral, _ = validate(model_ephemeral, device, val_loader)

        adamw_train_loss = train_standard(model_adamw, device, train_loader, optimizer_adamw, epoch)
        val_loss_adamw, _ = validate(model_adamw, device, val_loader)

        sgd_train_loss = train_standard(model_sgd, device, train_loader, optimizer_sgd, epoch)
        val_loss_sgd, _ = validate(model_sgd, device, val_loader)

        ephemeral_train_losses.append(ephemeral_train_loss)
        ephemeral_val_losses.append(val_loss_ephemeral)

        adamw_train_losses.append(adamw_train_loss)
        adamw_val_losses.append(val_loss_adamw)

        sgd_train_losses.append(sgd_train_loss)
        sgd_val_losses.append(val_loss_sgd)

    # Plot the training and validation loss curves for all optimizers
    plot_losses(epochs, [ephemeral_train_losses, adamw_train_losses, sgd_train_losses],
                [ephemeral_val_losses, adamw_val_losses, sgd_val_losses],
                ['EphemeralSGD', 'AdamW', 'SGD'], 'Training & Validation Loss Comparison for Multiple Optimizers')

    # Test the models and output results
    print("Testing EphemeralSGD Model")
    test(model_ephemeral, device, test_loader)
    print("Testing AdamW Model")
    test(model_adamw, device, test_loader)
    print("Testing SGD Model")
    test(model_sgd, device, test_loader)


if __name__ == '__main__':
    main()
