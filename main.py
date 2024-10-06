import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
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


# Custom AdEMA optimizer for comparison
class AdEMA(optim.Optimizer):
    def __init__(self, params, lr=0.001, beta=0.999, decay=0.999):
        defaults = dict(lr=lr, beta=beta, decay=decay)
        super(AdEMA, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta = group['beta']
            decay = group['decay']
            for p in group['params']:
                if p.grad is None:
                    continue

                # Create state for EMA gradients if it doesn't exist
                state = self.state[p]
                if 'exp_avg' not in state:
                    state['exp_avg'] = torch.zeros_like(p.data)

                # Calculate exponential moving average of gradients
                exp_avg = state['exp_avg']
                exp_avg.mul_(beta).add_(1 - beta, p.grad.data)

                # Apply decay and update weights
                p.data.add_(-group['lr'] * (1 - decay), exp_avg)

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
    return test_loss, accuracy  # Return loss and accuracy


# Plotting function
def plot_losses(epochs, losses, labels, title):
    """Plot the training loss for different optimizers."""
    for i, loss in enumerate(losses):
        plt.plot(range(1, epochs + 1), loss, label=labels[i])
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
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform),
                              batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform),
                             batch_size=1000, shuffle=False)

    # Initialize models and optimizers
    model_persistent = SimpleNet().to(device)
    model_adamw = SimpleNet().to(device)
    optimizer_persistent = PersistentSGD(model_persistent.parameters(), lr=learning_rate)
    optimizer_adamw = optim.AdamW(model_adamw.parameters(), lr=learning_rate)

    # Track losses for plotting
    persistent_losses, adamw_losses = [], []

    # Train models for all epochs
    for epoch in range(1, epochs + 1):
        train_loss_persistent = train(model_persistent, device, train_loader, optimizer_persistent, epoch)
        train_loss_adamw = train(model_adamw, device, train_loader, optimizer_adamw, epoch)
        persistent_losses.append(train_loss_persistent)
        adamw_losses.append(train_loss_adamw)

    # Plot the training loss curves for comparison
    plot_losses(epochs, [persistent_losses, adamw_losses], ['PersistentSGD', 'AdamW'], 'Training Loss Comparison')

    # Test the models and output results
    print("Testing PersistentSGD Model")
    test(model_persistent, device, test_loader)
    print("Testing AdamW Model")
    test(model_adamw, device, test_loader)


if __name__ == '__main__':
    main()
