import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import threading
import queue
from collections import defaultdict

# =======================
# Constants and Settings
# =======================
NUM_PARAMS = 10  # Number of parameters to visualize
VALUE_RANGE = 0.05  # Range for parameter visualization
Y_RANGE_MULTIPLIER = 1.0  # Multiplier for y-axis range in plots
NUM_TRIALS = 3  # Number of trials per optimizer
EPOCHS = 20  # Number of training epochs
BATCH_SIZE = 128  # Batch size for DataLoader
BASE_LR, PEAK_LR = 1e-5, 1e-3  # Learning rates for RescaledSGD
DECAY = 0.90  # Decay factor for RescaledSGD
SCALE_FACTOR = 128.0  # Scale factor for gradient scaling (if needed)

# Queue for communication between training threads and GUI
data_queue = queue.Queue()

# =======================
# Custom Optimizer
# =======================
class RescaledSGD(optim.Optimizer):
    def __init__(self, params, base_lr=1e-5, peak_lr=1e-3, decay=0.90):
        """
        Initializes the RescaledSGD optimizer.

        Args:
            params (iterable): Iterable of parameters to optimize.
            base_lr (float): Base learning rate.
            peak_lr (float): Peak learning rate after scaling.
            decay (float): Decay factor for persistent gradients.
        """
        defaults = dict(base_lr=base_lr, peak_lr=peak_lr, decay=decay, lr=base_lr)
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
                if 'effective_lr' not in state:
                    state['effective_lr'] = torch.full_like(p.data, base_lr)

                # Update persistent gradient with decay
                persistent_grad = state['persistent_grad']
                persistent_grad.mul_(decay).add_(p.grad.data)

                # Compute scaling factors based on min and max parameter updates
                grad_min, grad_max = persistent_grad.abs().min(), persistent_grad.abs().max()
                if grad_max != 0 and grad_min != 0:
                    scaling = (persistent_grad.abs() - grad_min) / (grad_max - grad_min + 1e-8)
                    scaled_lr = base_lr + (peak_lr - base_lr) * scaling
                    scaled_grad = scaled_lr * persistent_grad.sign()
                else:
                    # If gradients are too small, fallback to base learning rate
                    scaled_grad = persistent_grad * base_lr
                    scaled_lr = torch.full_like(p.data, base_lr)

                # Update the effective learning rate state
                state['effective_lr'].copy_(scaled_lr)

                # Update parameters
                p.data.add_(scaled_grad, alpha=-1)

        return loss
      
        
# =======================
# Neural Network Model
# =======================
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)  # Changed from Dropout2d
        self.dropout2 = nn.Dropout(0.5)   # Changed from Dropout2d
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# =======================
# GUI for Visualization
# =======================
class OptimizationPlotApp:
    def __init__(self, master, optimizers, epochs):
        self.master = master
        master.title("Optimizer Comparison")
        self.optimizers = optimizers
        self.epochs = epochs

        self.notebook = ttk.Notebook(master)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.figures = {}
        self.axes = {}
        self.lines = {}
        self.error_bars = {}
        self.animations = {}
        self.data = defaultdict(lambda: defaultdict(list))

        # Metrics to plot
        self.metrics = ['learning_rates', 'train_loss', 'val_loss', 'train_acc', 'val_acc']

        for name in self.metrics:
            for opt in optimizers:
                self.data[name][opt] = [[] for _ in range(NUM_TRIALS)]  # Initialize list for each trial
            self.create_plot(name)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(master, variable=self.progress_var, maximum=epochs * len(optimizers) * NUM_TRIALS)
        self.progress_bar.pack(fill=tk.X, padx=10, pady=10)

        # Status label
        self.status_var = tk.StringVar()
        self.status_label = ttk.Label(master, textvariable=self.status_var)
        self.status_label.pack()

        # Save button
        self.save_button = ttk.Button(master, text="Save Graphs", command=self.save_graphs)
        self.save_button.pack(pady=10)

    def create_plot(self, name):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=name.replace('_', ' ').title())
        fig, ax = plt.subplots(figsize=(8, 6))
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.figures[name] = fig
        self.axes[name] = ax

        x = list(range(1, self.epochs + 1))
        self.lines[name] = {opt: ax.plot(x, [0.1] * self.epochs, label=opt)[0] for opt in self.optimizers}  # Initialized with 0.1 to avoid log scale issues
        self.error_bars[name] = {opt: ax.fill_between(x, [0] * self.epochs, [0] * self.epochs, alpha=0.3) for opt in self.optimizers}

        ax.legend()
        ax.set_xlabel('Epoch')
        ax.set_ylabel(name.replace('_', ' ').title())
        ax.set_xlim(1, self.epochs)

        if 'acc' in name:
            ax.set_ylim(0, 100)
        elif 'loss' in name or 'learning_rates' in name:
            # Initially set to log scale; handle dynamically in animate
            ax.set_yscale('log')

        # Animation for dynamic updates
        self.animations[name] = FuncAnimation(fig, self.animate, fargs=(name,), interval=500, blit=True, cache_frame_data=False)

        # Hover annotation
        self.hover_annotation = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                                            bbox=dict(boxstyle="round", fc="w"),
                                            arrowprops=dict(arrowstyle="->"))
        self.hover_annotation.set_visible(False)
        fig.canvas.mpl_connect("motion_notify_event", lambda event: self.hover(event, name))

    def hover(self, event, name):
        if event.inaxes == self.axes[name]:
            for opt, line in self.lines[name].items():
                cont, ind = line.contains(event)
                if cont:
                    x_data, y_data = line.get_data()
                    if len(ind["ind"]) > 0:
                        idx = ind["ind"][0]
                        x, y = x_data[idx], y_data[idx]
                        self.hover_annotation.xy = (x, y)
                        self.hover_annotation.set_text(f"{opt}: {y:.4f}")
                        self.hover_annotation.set_visible(True)
                        self.figures[name].canvas.draw_idle()
                        return
        self.hover_annotation.set_visible(False)
        self.figures[name].canvas.draw_idle()

    def animate(self, _, name):
        updated_artists = []
        for opt in self.optimizers:
            data = self.data[name][opt]
            # Find the minimum length across trials
            min_length = min(len(trial) for trial in data)
            if min_length == 0:
                continue  # No data to plot yet

            # Compute mean and std for each epoch across trials up to min_length
            epoch_means = []
            epoch_stds = []
            for epoch_idx in range(min_length):
                epoch_values = [trial[epoch_idx] for trial in data if len(trial) > epoch_idx]
                if epoch_values:
                    epoch_means.append(np.mean(epoch_values))
                    epoch_stds.append(np.std(epoch_values))
                else:
                    epoch_means.append(0)
                    epoch_stds.append(0)

            if not epoch_means:
                continue

            self.lines[name][opt].set_ydata(epoch_means)
            self.error_bars[name][opt].remove()
            self.error_bars[name][opt] = self.axes[name].fill_between(
                range(1, len(epoch_means) + 1),
                [m - s for m, s in zip(epoch_means, epoch_stds)],
                [m + s for m, s in zip(epoch_means, epoch_stds)],
                alpha=0.3
            )
            updated_artists.extend([self.lines[name][opt], self.error_bars[name][opt]])

            # Dynamically set y-scale based on data
            if 'acc' in name:
                self.axes[name].set_ylim(0, 100)
            elif 'loss' in name or 'learning_rates' in name:
                if all(m > 0 for m in epoch_means):
                    self.axes[name].set_yscale('log')
                else:
                    self.axes[name].set_yscale('linear')

        return updated_artists

    def update_plot(self, name, new_data, trial):
        """Update the plot data with new metrics from a specific trial."""
        for opt, value in new_data.items():
            if trial < NUM_TRIALS:
                self.data[name][opt][trial].append(value)
            else:
                print(f"Received data for trial {trial}, but NUM_TRIALS is set to {NUM_TRIALS}")

    def update_progress(self, increment):
        self.progress_var.set(self.progress_var.get() + increment)

    def save_graphs(self):
        directory = filedialog.askdirectory()
        if directory:
            for name, fig in self.figures.items():
                fig.savefig(f"{directory}/{name}.png")
            messagebox.showinfo("Save Complete", "All graphs have been saved.")

# =======================
# Optimizer Configurations
# =======================
def get_rescaled_sgd_config():
    model = MNISTNet()
    optimizer = RescaledSGD(model.parameters(), base_lr=BASE_LR, peak_lr=PEAK_LR, decay=DECAY)
    # No StepLR used here; rely on dynamic adjustment
    return model, optimizer

def get_sgd_config():
    model = MNISTNet()
    optimizer = optim.SGD(model.parameters(), lr=BASE_LR)
    scheduler = StepLR(optimizer, step_size=10, gamma=(PEAK_LR / BASE_LR) ** (1 / EPOCHS))
    return model, optimizer, scheduler

def get_sgd_momentum_config():
    model = MNISTNet()
    optimizer = optim.SGD(model.parameters(), lr=BASE_LR, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=10, gamma=(PEAK_LR / BASE_LR) ** (1 / EPOCHS))
    return model, optimizer, scheduler

def get_adam_config():
    model = MNISTNet()
    optimizer = optim.Adam(model.parameters(), lr=BASE_LR)
    scheduler = StepLR(optimizer, step_size=10, gamma=(PEAK_LR / BASE_LR) ** (1 / EPOCHS))
    return model, optimizer, scheduler

optimizers_config = {
    'RescaledSGD': get_rescaled_sgd_config,
    'SGD': get_sgd_config,
    'SGD_Momentum': get_sgd_momentum_config,
    'Adam': get_adam_config
}

# =======================
# Training and Validation
# =======================
def train_with_custom_scaling(model, device, train_loader, optimizer, epoch, log_interval, gui_queue, trial):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    # Ensure 'effective_lr' is initialized for all parameters before training
    initialize_effective_lr(optimizer, BASE_LR)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Define a closure that computes the output and loss
        def closure():
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            return loss, output

        # Call the closure to get the loss and output
        loss, output = closure()
        optimizer.step(lambda: loss)

        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        # Send metrics to GUI for visualization
        if gui_queue:
            with torch.no_grad():
                params = model.fc1.weight.data.cpu().numpy().flatten()[:NUM_PARAMS]
                grads = model.fc1.weight.grad.data.cpu().numpy().flatten()[:NUM_PARAMS] if model.fc1.weight.grad is not None else np.zeros(NUM_PARAMS)

                # Safely access 'effective_lr' with a fallback value after initialization
                effective_lr = optimizer.state.get(model.fc1.weight, {}).get(
                    'effective_lr',
                    torch.full_like(model.fc1.weight.data, BASE_LR)
                ).cpu().numpy().flatten()[:NUM_PARAMS]

            gui_queue.put(('metrics', {
                'learning_rates': effective_lr.mean(),
                'train_loss': train_loss / (batch_idx + 1),
                'val_loss': 0.0,  # Placeholder; will be updated in run_experiment
                'train_acc': 100. * correct / total,
                'val_acc': 0.0  # Placeholder; will be updated in run_experiment
            }, trial))

    return train_loss / len(train_loader), 100. * correct / total

def validate(model, device, val_loader):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    accuracy = 100. * correct / len(val_loader.dataset)
    return val_loss, accuracy

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy

# =======================
# Experiment Runner
# =======================
def initialize_effective_lr(optimizer, base_lr):
    """
    Ensures that the 'effective_lr' state is initialized for all parameters
    in the given optimizer.
    """
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            if 'effective_lr' not in state:
                state['effective_lr'] = torch.full_like(p.data, base_lr)

# =======================
# Training Loop Adjustment for Optimizer Switch
# =======================
def run_experiment(config_func, train_loader, val_loader, test_loader, device, epochs, gui_queue, trial):
    # Unpack model and optimizer, optionally a scheduler if provided
    model, optimizer, *scheduler_optional = config_func()
    scheduler = scheduler_optional[0] if scheduler_optional else None

    model = model.to(device)

    # Ensure the 'effective_lr' is properly initialized on optimizer switch
    initialize_effective_lr(optimizer, BASE_LR)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_with_custom_scaling(
            model, device, train_loader, optimizer, epoch, log_interval=100, gui_queue=gui_queue, trial=trial
        )
        val_loss, val_acc = validate(model, device, val_loader)

        if scheduler:
            scheduler.step()

        gui_queue.put(('metrics', {
            'learning_rates': optimizer.param_groups[0]['lr'],
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc
        }, trial))
        gui_queue.put(('progress', 1))

    test_loss, test_acc = test(model, device, test_loader)
    return test_loss, test_acc



# =======================
# Main Function
# =======================
def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download and prepare datasets
    full_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize GUI
    root = tk.Tk()
    app = OptimizationPlotApp(root, optimizers=list(optimizers_config.keys()), epochs=EPOCHS)

    # Function to handle incoming data from the training thread
    def handle_queue():
        try:
            while not data_queue.empty():
                msg = data_queue.get_nowait()
                if len(msg) == 3:
                    msg_type, data, trial = msg
                    if msg_type == 'metrics':
                        for name, value in data.items():
                            app.update_plot(name, {current_optimizer: value}, trial)
                elif len(msg) == 2:
                    msg_type, increment = msg
                    if msg_type == 'progress':
                        app.update_progress(increment)
                else:
                    print(f"Invalid message format: {msg}")
        except queue.Empty:
            pass
        # Schedule the next queue check
        root.after(100, handle_queue)

    # Start handling the queue
    root.after(100, handle_queue)

    # Training loop in a separate thread
    def training_loop():
        results = defaultdict(list)
        for trial in range(NUM_TRIALS):
            for opt_name, config_func in optimizers_config.items():
                app.status_var.set(f"Training with {opt_name} (Trial {trial + 1}/{NUM_TRIALS})")
                global current_optimizer
                current_optimizer = opt_name

                test_loss, test_acc = run_experiment(config_func, train_loader, val_loader, test_loader, device, EPOCHS, data_queue, trial)
                results[opt_name].append((test_loss, test_acc))
                print(f"{opt_name} - Trial {trial + 1} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

        app.status_var.set("Training completed")

        # Display final results
        final_results_window = tk.Toplevel(root)
        final_results_window.title("Final Results")

        results_text = tk.Text(final_results_window, height=20, width=50)
        results_text.pack(padx=10, pady=10)

        results_text.insert(tk.END, f"Final Results (averaged over {NUM_TRIALS} trials):\n\n")
        for opt_name, trials in results.items():
            avg_loss = np.mean([t[0] for t in trials])
            avg_acc = np.mean([t[1] for t in trials])
            std_acc = np.std([t[1] for t in trials])
            results_text.insert(tk.END, f"{opt_name}:\n")
            results_text.insert(tk.END, f"  Avg Test Loss: {avg_loss:.4f}\n")
            results_text.insert(tk.END, f"  Avg Test Accuracy: {avg_acc:.2f}% ± {std_acc:.2f}%\n\n")

        results_text.config(state=tk.DISABLED)

    # Start the training thread
    training_thread = threading.Thread(target=training_loop)
    training_thread.start()

    # Start the GUI main loop
    root.mainloop()

if __name__ == '__main__':
    main()
