import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import queue
import time

# Constants
NUM_PARAMS = 10  # Number of parameters to visualize
VALUE_RANGE = 0.05  # Range of parameter values (-VALUE_RANGE to VALUE_RANGE)
Y_RANGE_MULTIPLIER = 1.0  # Default multiplier for y-axis range

# Queue for communication between training and GUI
data_queue = queue.Queue()

class EphemeralSGD(optim.Optimizer):
    def __init__(self, params, lr=0.01, decay=0.99, new_min=0.1, new_max=10.0):
        """
        Initializes the EphemeralSGD optimizer.

        Args:
            params (iterable): Iterable of parameters to optimize.
            lr (float): Learning rate.
            decay (float): Decay factor for persistent gradients.
            new_min (float): New minimum gradient magnitude after rescaling.
            new_max (float): New maximum gradient magnitude after rescaling.
        """
        defaults = dict(lr=lr, decay=decay, new_min=new_min, new_max=new_max)
        super(EphemeralSGD, self).__init__(params, defaults)

    def rescale_gradients(self, grad, new_min, new_max):
        """
        Rescales gradients to compress the range of magnitudes.

        Args:
            grad (torch.Tensor): Gradient tensor.
            new_min (float): Desired minimum gradient magnitude.
            new_max (float): Desired maximum gradient magnitude.

        Returns:
            torch.Tensor: Rescaled gradient tensor.
        """
        # Compute absolute gradients
        abs_grad = grad.abs()
        min_grad = abs_grad.min()
        max_grad = abs_grad.max()

        # Avoid division by zero
        if max_grad == min_grad:
            return grad

        # Rescale using min-max normalization to the desired range
        scaled_grad = (abs_grad - min_grad) / (max_grad - min_grad)  # Normalize to [0, 1]
        scaled_grad = scaled_grad * (new_max - new_min) + new_min      # Scale to [new_min, new_max]

        # Preserve the sign of the original gradient
        rescaled_grad = scaled_grad * grad.sign()

        return rescaled_grad

    def step(self, closure=None):
        """
        Performs a single optimization step with rescaled gradients.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.

        Returns:
            loss: The loss computed by the closure, if provided.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            decay = group['decay']
            lr = group['lr']
            new_min = group['new_min']
            new_max = group['new_max']

            for p in group['params']:
                if p.grad is None:
                    continue

                # Rescale the gradient
                rescaled_grad = self.rescale_gradients(p.grad.data, new_min, new_max)

                # Initialize persistent gradient if not present
                state = self.state[p]
                if 'persistent_grad' not in state:
                    state['persistent_grad'] = torch.zeros_like(p.data)

                # Update persistent gradient with decay
                persistent_grad = state['persistent_grad']
                persistent_grad.mul_(decay).add_(rescaled_grad)

                # Update parameters permanently
                p.data.add_(persistent_grad, alpha=-lr)

        return loss

# Simple neural network model for MNIST
class SimpleNet(nn.Module):
    def __init__(self, num_params=NUM_PARAMS):
        super(SimpleNet, self).__init__()
        # Adjusting the network to have NUM_PARAMS for visualization purposes
        self.fc1 = nn.Linear(28 * 28, num_params)
        self.fc2 = nn.Linear(num_params, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input tensor
        x = torch.relu(self.fc1(x))  # Apply ReLU activation
        x = self.fc2(x)  # Output layer
        return x

# GUI for parameter visualization
class ParameterPlotApp:
    def __init__(self, master):
        self.master = master
        master.title("Parameter Plot")

        self.parameters = np.random.uniform(-VALUE_RANGE, VALUE_RANGE, NUM_PARAMS)
        self.gradients = np.random.uniform(-1, 1, NUM_PARAMS)
        self.y_range_multiplier = Y_RANGE_MULTIPLIER

        # Set up matplotlib figure and axis
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Controls frame
        controls = ttk.Frame(master)
        controls.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        # Effective Learning Rate label
        self.lr_label = ttk.Label(controls, text="Effective Learning Rate: 0.00")
        self.lr_label.pack(side=tk.TOP)

        # Learning rate slider
        self.slider = ttk.Scale(controls, from_=0, to=1, orient=tk.HORIZONTAL, length=300, command=self.update_plot)
        self.slider.pack(side=tk.TOP)

        # Retain Minimum Scaling checkbox
        self.retain_min_var = tk.BooleanVar()
        ttk.Checkbutton(controls, text="Retain Minimum Scaling", variable=self.retain_min_var, command=self.update_plot).pack(side=tk.TOP)

        # Base Learning Rate input
        base_lr_frame = ttk.Frame(controls)
        base_lr_frame.pack(side=tk.TOP)
        ttk.Label(base_lr_frame, text="Base LR:").pack(side=tk.LEFT)
        self.base_lr_entry = ttk.Entry(base_lr_frame, width=10)
        self.base_lr_entry.insert(0, "1e-5")
        self.base_lr_entry.pack(side=tk.LEFT)
        ttk.Button(base_lr_frame, text="Update", command=self.update_plot).pack(side=tk.LEFT)

        # Y-Range Multiplier input
        y_range_frame = ttk.Frame(controls)
        y_range_frame.pack(side=tk.TOP)
        ttk.Label(y_range_frame, text="Y-Range Multiplier:").pack(side=tk.LEFT)
        self.y_range_entry = ttk.Entry(y_range_frame, width=10)
        self.y_range_entry.insert(0, str(Y_RANGE_MULTIPLIER))
        self.y_range_entry.pack(side=tk.LEFT)
        ttk.Button(y_range_frame, text="Update", command=self.update_y_range).pack(side=tk.LEFT)

        # Initialize plot
        self.update_plot()

    def get_learning_rate(self):
        try:
            base_lr = float(self.base_lr_entry.get())
        except ValueError:
            base_lr = 1e-5
        slider_lr = self.slider.get()
        return max(base_lr, slider_lr)

    def apply_learning_rate(self, gradients, learning_rate):
        base_lr = float(self.base_lr_entry.get())
        if not self.retain_min_var.get() or learning_rate <= base_lr:
            return learning_rate * gradients

        abs_gradients = np.abs(gradients)
        scaled_gradients = (abs_gradients - abs_gradients.min()) / (abs_gradients.max() - abs_gradients.min() + 1e-8)  # Avoid division by zero
        adjusted_lr = base_lr + (learning_rate - base_lr) * scaled_gradients
        return adjusted_lr * gradients

    def update_y_range(self):
        try:
            self.y_range_multiplier = float(self.y_range_entry.get())
            self.update_plot()
        except ValueError:
            print("Invalid Y-Range Multiplier")

    def update_plot(self, *args):
        learning_rate = self.get_learning_rate()
        self.lr_label.config(text=f"Effective Learning Rate: {learning_rate:.2e}")

        # Apply learning rate to gradients
        parameter_changes = self.apply_learning_rate(self.gradients, learning_rate)
        updated_parameters = self.parameters + parameter_changes

        self.ax.clear()
        x = np.arange(NUM_PARAMS)
        width = 0.35

        # Current parameters
        self.ax.bar(x - width/2, self.parameters, width, label='Current', color='skyblue')
        # Updated parameters
        self.ax.bar(x + width/2, updated_parameters, width, label='Updated', color='lightgreen')

        # Arrows indicating parameter changes
        for i, (current, updated) in enumerate(zip(self.parameters, updated_parameters)):
            self.ax.arrow(i, current, 0, updated - current, color='red', width=0.01,
                          head_width=0.1, head_length=0.1 * self.y_range_multiplier * VALUE_RANGE)

        self.ax.axhline(y=0, color='r', linestyle='-', linewidth=0.5)
        self.ax.set_xlabel('Parameter Index')
        self.ax.set_ylabel('Value')
        self.ax.set_title(f'Parameter Changes (LR: {learning_rate:.2e})')
        self.ax.set_ylim(-VALUE_RANGE * self.y_range_multiplier, VALUE_RANGE * self.y_range_multiplier)
        self.ax.legend()
        self.ax.grid(axis='y', linestyle='--', alpha=0.7)
        self.ax.set_xticks(x)

        self.canvas.draw()

    def update_parameters(self, parameters, gradients):
        """
        Receives updated parameters and gradients from the training loop and updates the plot.

        Args:
            parameters (np.ndarray): Current parameter values.
            gradients (np.ndarray): Current gradient values.
        """
        self.parameters = parameters
        self.gradients = gradients
        self.update_plot()

class ParameterVisualizer(threading.Thread):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.daemon = True  # Daemonize thread

    def run(self):
        self.app.master.mainloop()

# Training function for ephemeral updates
def train_ephemeral(model, device, train_loader, optimizer, epoch, log_interval=100, gui_queue=None):
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

        # Optionally, send parameter updates to GUI
        if gui_queue:
            with torch.no_grad():
                params = model.fc1.weight.data.cpu().numpy().flatten()[:NUM_PARAMS]
                grads = model.fc1.weight.grad.data.cpu().numpy().flatten()[:NUM_PARAMS]
            gui_queue.put((params, grads))

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
    plt.figure(figsize=(10, 6))
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

    optimizer_ephemeral = EphemeralSGD(model_ephemeral.parameters(), lr=learning_rate, decay=0.99, new_min=0.1, new_max=10.0)
    optimizer_adamw = optim.AdamW(model_adamw.parameters(), lr=learning_rate)
    optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=learning_rate)

    # Track losses for plotting
    ephemeral_train_losses, ephemeral_val_losses = [], []
    adamw_train_losses, adamw_val_losses = [], []
    sgd_train_losses, sgd_val_losses = [], []

    # Initialize GUI
    root = tk.Tk()
    app = ParameterPlotApp(root)

    # Function to handle incoming data from the training thread
    def handle_queue():
        try:
            while not data_queue.empty():
                params, grads = data_queue.get_nowait()
                app.update_parameters(params, grads)
        except queue.Empty:
            pass
        # Schedule the next queue check
        root.after(100, handle_queue)

    # Start handling the queue
    root.after(100, handle_queue)

    # Training loop in a separate thread
    def training_loop():
        for epoch in range(1, epochs + 1):
            print(f"--- Epoch {epoch} ---")
            
            # Train EphemeralSGD
            ephemeral_train_loss = train_ephemeral(model_ephemeral, device, train_loader, optimizer_ephemeral, epoch, gui_queue=data_queue)
            val_loss_ephemeral, val_acc_ephemeral = validate(model_ephemeral, device, val_loader)

            # Train AdamW
            adamw_train_loss = train_standard(model_adamw, device, train_loader, optimizer_adamw, epoch)
            val_loss_adamw, val_acc_adamw = validate(model_adamw, device, val_loader)

            # Train SGD
            sgd_train_loss = train_standard(model_sgd, device, train_loader, optimizer_sgd, epoch)
            val_loss_sgd, val_acc_sgd = validate(model_sgd, device, val_loader)

            # Append losses for plotting
            ephemeral_train_losses.append(ephemeral_train_loss)
            ephemeral_val_losses.append(val_loss_ephemeral)

            adamw_train_losses.append(adamw_train_loss)
            adamw_val_losses.append(val_loss_adamw)

            sgd_train_losses.append(sgd_train_loss)
            sgd_val_losses.append(val_loss_sgd)

        # After training, plot the losses
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

    # Start the training thread
    training_thread = threading.Thread(target=training_loop)
    training_thread.start()

    # Start the GUI main loop
    root.mainloop()

if __name__ == '__main__':
    main()
