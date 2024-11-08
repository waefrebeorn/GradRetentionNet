import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from EnhancedSGD import EnhancedSGD  # Import the EnhancedSGD optimizer

# Define a simple CNN model for MNIST
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
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

# Set up data loaders for the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Loss function
criterion = nn.CrossEntropyLoss()

# Set up models and optimizers
model_enhanced_sgd = SimpleCNN()
model_sgd = SimpleCNN()
model_adamw = SimpleCNN()

# Initialize optimizers
optimizer_enhanced_sgd = EnhancedSGD(model_enhanced_sgd.parameters(), lr=0.01, model=model_enhanced_sgd, 
                                     usage_case="GenAI", use_amp=True, lookahead_k=5, lookahead_alpha=0.5)
optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=0.01, momentum=0.9)
optimizer_adamw = optim.AdamW(model_adamw.parameters(), lr=0.001)

# Training function
def train(model, optimizer, num_epochs=5):
    model.train()
    train_losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")
    return train_losses

# Test function
def test(model):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = correct / len(test_loader.dataset)
    return accuracy

# Train and test each optimizer
num_epochs = 25
print("Training with EnhancedSGD...")
loss_enhanced_sgd = train(model_enhanced_sgd, optimizer_enhanced_sgd, num_epochs=num_epochs)
print("\nTraining with SGD...")
loss_sgd = train(model_sgd, optimizer_sgd, num_epochs=num_epochs)
print("\nTraining with AdamW...")
loss_adamw = train(model_adamw, optimizer_adamw, num_epochs=num_epochs)

# Test accuracies
accuracy_enhanced_sgd = test(model_enhanced_sgd)
accuracy_sgd = test(model_sgd)
accuracy_adamw = test(model_adamw)

# Plotting training losses
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), loss_enhanced_sgd, label="EnhancedSGD")
plt.plot(range(1, num_epochs + 1), loss_sgd, label="SGD")
plt.plot(range(1, num_epochs + 1), loss_adamw, label="AdamW")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss Comparison of Optimizers on MNIST")
plt.legend()
plt.show()

# Print test accuracies
print("\nTest Accuracies:")
print(f"EnhancedSGD Test Accuracy: {accuracy_enhanced_sgd:.4f}")
print(f"SGD Test Accuracy: {accuracy_sgd:.4f}")
print(f"AdamW Test Accuracy: {accuracy_adamw:.4f}")
