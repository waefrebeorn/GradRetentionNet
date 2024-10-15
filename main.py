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


class RescaledSGD(optim.Optimizer):
    def __init__(self, params, base_lr=1e-7, peak_lr=1e-4, decay=0.99):
        """
        Initializes the RescaledSGD optimizer.

        Args:
            params (iterable): Iterable of parameters to optimize.
            base_lr (float): Base learning rate.
            peak_lr (float): Peak learning rate after scaling.
            decay (float): Decay factor for persistent gradients.
        """
        defaults = dict(base_lr=base_lr, peak_lr=peak_lr, decay=decay)
        super(RescaledSGD, self).__init__(params, defaults)

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
            base_lr = group['base_lr']
            peak_lr = group['peak_lr']
            decay = group['decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                # Initialize state
                state = self.state[p]
                if 'persistent_grad' not in state:
                    state['persistent_grad'] = torch.zeros_like(p.data)

                # Update persistent gradient with decay
                persistent_grad = state['persistent_grad']
                persistent_grad.mul_(decay).add_(p.grad.data)

                # Compute scaling factors based on min and max parameter updates
                if persistent_grad.abs().max() != 0 and persistent_grad.abs().min() != 0:
                    scaling = (persistent_grad.abs() - persistent_grad.abs().min()) / (
                        persistent_grad.abs().max() - persistent_grad.abs().min() + 1e-8
                    )
                    scaled_lr = base_lr + (peak_lr - base_lr) * scaling
                    scaled_grad = scaled_lr * persistent_grad.sign()
                else:
                    scaled_lr = base_lr if persistent_grad.abs().max() == 0 else peak_lr
                    scaled_grad = persistent_grad * scaled_lr

                # Store the effective learning rates for visualization
                if 'effective_lr' not in state:
                    state['effective_lr'] = torch.zeros_like(p.data)
                state['effective_lr'].copy_(scaled_lr)

                # Update parameters
                p.data.add_(scaled_grad, alpha=-1)

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
    def __init__(self, master, optimizers):
        self.master = master
        master.title("Optimizer Parameter Visualization")

        self.optimizers = optimizers  # List of optimizer names
        self.num_optimizers = len(optimizers)

        # Initialize parameter, gradient, and learning rate storage for each optimizer
        self.parameters = {opt: np.random.uniform(-VALUE_RANGE, VALUE_RANGE, NUM_PARAMS) for opt in optimizers}
        self.gradients = {opt: np.random.uniform(-1, 1, NUM_PARAMS) for opt in optimizers}
        self.learning_rates = {opt: np.zeros(NUM_PARAMS) for opt in optimizers}
        self.y_range_multiplier = Y_RANGE_MULTIPLIER

        # Create notebook for tabs
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Dictionaries to hold figures and axes
        self.figures = {}
        self.axes = {}
        self.canvases = {}

        # Create a tab for each optimizer
        for opt in optimizers:
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=opt)
            fig, ax = plt.subplots(figsize=(12, 6))
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            self.figures[opt] = fig
            self.axes[opt] = ax
            self.canvases[opt] = canvas

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
        base_lr_frame.pack(side=tk.TOP, pady=5)
        ttk.Label(base_lr_frame, text="Base LR:").pack(side=tk.LEFT)
        self.base_lr_entry = ttk.Entry(base_lr_frame, width=10)
        self.base_lr_entry.insert(0, "1e-7")
        self.base_lr_entry.pack(side=tk.LEFT)
        ttk.Button(base_lr_frame, text="Update", command=self.update_plot).pack(side=tk.LEFT, padx=5)

        # Y-Range Multiplier input
        y_range_frame = ttk.Frame(controls)
        y_range_frame.pack(side=tk.TOP, pady=5)
        ttk.Label(y_range_frame, text="Y-Range Multiplier:").pack(side=tk.LEFT)
        self.y_range_entry = ttk.Entry(y_range_frame, width=10)
        self.y_range_entry.insert(0, str(Y_RANGE_MULTIPLIER))
        self.y_range_entry.pack(side=tk.LEFT)
        ttk.Button(y_range_frame, text="Update", command=self.update_y_range).pack(side=tk.LEFT, padx=5)

        # Initialize plots for each optimizer
        self.update_all_plots()

    def get_learning_rate(self):
        try:
            base_lr = float(self.base_lr_entry.get())
        except ValueError:
            base_lr = 1e-7
        slider_lr = self.slider.get()
        return max(base_lr, slider_lr)

    def apply_learning_rate(self, gradients, learning_rate):
        base_lr = float(self.base_lr_entry.get())
        if not self.retain_min_var.get() or learning_rate <= base_lr:
            return learning_rate * gradients, np.full_like(gradients, learning_rate)

        abs_gradients = np.abs(gradients)
        scaled_gradients = (abs_gradients - abs_gradients.min()) / (abs_gradients.max() - abs_gradients.min() + 1e-8)  # Avoid division by zero
        adjusted_lr = base_lr + (learning_rate - base_lr) * scaled_gradients
        return adjusted_lr * gradients, adjusted_lr

    def update_y_range(self):
        try:
            self.y_range_multiplier = float(self.y_range_entry.get())
            self.update_all_plots()
        except ValueError:
            print("Invalid Y-Range Multiplier")

    def update_plot(self, *args):
        learning_rate = self.get_learning_rate()
        self.lr_label.config(text=f"Effective Learning Rate: {learning_rate:.2e}")

        # Iterate through each optimizer and update its plot
        for opt in self.optimizers:
            parameter_changes, effective_lr = self.apply_learning_rate(self.gradients[opt], learning_rate)
            updated_parameters = self.parameters[opt] + parameter_changes

            ax = self.axes[opt]
            ax.clear()
            x = np.arange(NUM_PARAMS)
            width = 0.35

            # Current parameters
            ax.bar(x - width / 2, self.parameters[opt], width, label='Current', color='skyblue')
            # Updated parameters
            ax.bar(x + width / 2, updated_parameters, width, label='Updated', color='lightgreen')
            # Plot effective learning rates as a line
            ax.plot(x, effective_lr, label='Effective LR', color='orange', marker='o', linestyle='dashed')

            # Arrows indicating parameter changes
            for i, (current, updated) in enumerate(zip(self.parameters[opt], updated_parameters)):
                ax.arrow(i, current, 0, updated - current, color='red', width=0.005,
                         head_width=0.02, head_length=0.01 * self.y_range_multiplier * VALUE_RANGE)

            ax.axhline(y=0, color='r', linestyle='-', linewidth=0.5)
            ax.set_xlabel('Parameter Index')
            ax.set_ylabel('Value / Effective LR')
            ax.set_title(f'{opt} Parameter Changes (LR: {learning_rate:.2e})')
            ax.set_ylim(-VALUE_RANGE * self.y_range_multiplier, VALUE_RANGE * self.y_range_multiplier)
            ax.legend()
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.set_xticks(x)

            self.canvases[opt].draw()

    def update_all_plots(self):
        for opt in self.optimizers:
            self.update_plot()

    def update_parameters(self, optimizer_name, parameters, gradients, effective_lr):
        """
        Receives updated parameters, gradients, and effective learning rates from the training loop.

        Args:
            optimizer_name (str): Name of the optimizer.
            parameters (np.ndarray): Current parameter values.
            gradients (np.ndarray): Current gradient values.
            effective_lr (np.ndarray): Effective learning rates applied to the parameters.
        """
        if optimizer_name in self.optimizers:
            # Ensure we only take the first NUM_PARAMS for visualization
            self.parameters[optimizer_name] = parameters[:NUM_PARAMS]
            self.gradients[optimizer_name] = gradients[:NUM_PARAMS]
            self.learning_rates[optimizer_name] = effective_lr[:NUM_PARAMS]
            self.update_plot()


# Training function for RescaledSGD
def train_rescaled_sgd(model, device, train_loader, optimizer, epoch, log_interval=100, gui_queue=None):
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

        # Perform optimizer step
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
                # Extract effective learning rates
                effective_lr = optimizer.state[model.fc1.weight]['effective_lr'].cpu().numpy().flatten()[:NUM_PARAMS]
            gui_queue.put((optimizer.__class__.__name__, params, grads, effective_lr))

    return epoch_loss / len(train_loader)


# Training function for SGD with Momentum
def train_sgd_momentum(model, device, train_loader, optimizer, epoch, log_interval=100, gui_queue=None):
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

        # Optionally, send parameter updates to GUI
        if gui_queue:
            with torch.no_grad():
                params = model.fc1.weight.data.cpu().numpy().flatten()[:NUM_PARAMS]
                grads = model.fc1.weight.grad.data.cpu().numpy().flatten()[:NUM_PARAMS]
                # Effective learning rates are fixed for SGD with Momentum (constant)
                effective_lr = np.full(NUM_PARAMS, optimizer.param_groups[0]['lr'])
            gui_queue.put((optimizer.__class__.__name__, params, grads, effective_lr))

    return epoch_loss / len(train_loader)


# Training function for AdamW
def train_adamw(model, device, train_loader, optimizer, epoch, log_interval=100, gui_queue=None):
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

        # Optionally, send parameter updates to GUI
        if gui_queue:
            with torch.no_grad():
                params = model.fc1.weight.data.cpu().numpy().flatten()[:NUM_PARAMS]
                grads = model.fc1.weight.grad.data.cpu().numpy().flatten()[:NUM_PARAMS]
                # AdamW has adaptive learning rates; approximate by using the current learning rate
                # This is a simplification; for true per-parameter learning rates, more detailed tracking is needed
                effective_lr = np.array([group['lr'] for group in optimizer.param_groups] * NUM_PARAMS)[:NUM_PARAMS]
            gui_queue.put((optimizer.__class__.__name__, params, grads, effective_lr))

    return epoch_loss / len(train_loader)


# Validation function to assess model performance
def validate(model, device, val_loader):
    model.eval()
    val_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    accuracy = 100. * correct / len(val_loader.dataset)
    print(f'Validation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(val_loader.dataset)} '
          f'({accuracy:.0f}%)\n')
    return val_loss, accuracy


# Testing function to measure model accuracy on the test set
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
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
    base_lr_rescaled_sgd = 1e-7
    peak_lr_rescaled_sgd = 1e-4
    lr_sgd = 1e-4
    momentum = 0.9
    lr_adamw = 1e-3
    weight_decay_adamw = 1e-2
    decay = 0.99
    log_interval = 100
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Data transformation and loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)

    # Split the training set into training and validation sets
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1000, shuffle=False)

    # Initialize models and optimizers
    optimizers_config = {
        'RescaledSGD': {
            'model': SimpleNet().to(device),
            'optimizer': RescaledSGD(SimpleNet().parameters(),
                                     base_lr=base_lr_rescaled_sgd,
                                     peak_lr=peak_lr_rescaled_sgd,
                                     decay=decay)
        },
        'StandardSGD': {
            'model': SimpleNet().to(device),
            'optimizer': optim.SGD(SimpleNet().parameters(),
                                   lr=lr_sgd)
        },
        'SGD_Momentum': {
            'model': SimpleNet().to(device),
            'optimizer': optim.SGD(SimpleNet().parameters(),
                                   lr=lr_sgd,
                                   momentum=momentum)
        },
        'AdamW': {
            'model': SimpleNet().to(device),
            'optimizer': optim.AdamW(SimpleNet().parameters(),
                                     lr=lr_adamw,
                                     weight_decay=weight_decay_adamw)
        }
    }

    # Track losses for plotting
    train_losses = {opt: [] for opt in optimizers_config.keys()}
    val_losses = {opt: [] for opt in optimizers_config.keys()}

    # Initialize GUI with all optimizers
    root = tk.Tk()
    app = ParameterPlotApp(root, optimizers=list(optimizers_config.keys()))

    # Function to handle incoming data from the training thread
    def handle_queue():
        try:
            while not data_queue.empty():
                item = data_queue.get_nowait()
                if item[0] == 'plot_losses':
                    _, epochs_, train_loss_data, val_loss_data, labels = item
                    # Reorganize loss data
                    train_losses_list = [train_loss_data[opt] for opt in optimizers_config.keys()]
                    val_losses_list = [val_loss_data[opt] for opt in optimizers_config.keys()]
                    plot_losses(epochs_,
                                train_losses_list,
                                val_losses_list,
                                labels,
                                'Training & Validation Loss Comparison')
                else:
                    optimizer_name, params, grads, effective_lr = item
                    app.update_parameters(optimizer_name, params, grads, effective_lr)
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

            # Train each optimizer
            for opt_name, config in optimizers_config.items():
                model = config['model']
                optimizer = config['optimizer']

                if opt_name == 'RescaledSGD':
                    train_loss = train_rescaled_sgd(model, device, train_loader, optimizer, epoch, log_interval, gui_queue=data_queue)
                elif opt_name == 'StandardSGD':
                    train_loss = train_sgd_momentum(model, device, train_loader, optimizer, epoch, log_interval, gui_queue=data_queue)
                elif opt_name == 'SGD_Momentum':
                    train_loss = train_sgd_momentum(model, device, train_loader, optimizer, epoch, log_interval, gui_queue=data_queue)
                elif opt_name == 'AdamW':
                    train_loss = train_adamw(model, device, train_loader, optimizer, epoch, log_interval, gui_queue=data_queue)
                else:
                    continue  # Unknown optimizer

                train_losses[opt_name].append(train_loss)
                val_loss, val_acc = validate(model, device, val_loader)
                val_losses[opt_name].append(val_loss)

            # After training all optimizers for this epoch, continue to next epoch

        # After training, send a message to plot the losses
        data_queue.put(('plot_losses', epochs,
                       train_losses, val_losses,
                       list(optimizers_config.keys())))

        # Test the models and output results
        for opt_name, config in optimizers_config.items():
            model = config['model']
            print(f"Testing {opt_name} Model")
            test(model, device, test_loader)

    # Start the training thread
    training_thread = threading.Thread(target=training_loop)
    training_thread.start()

    # Start the GUI main loop
    root.mainloop()


if __name__ == '__main__':
    main()
