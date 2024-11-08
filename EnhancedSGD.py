import torch
import math
import random
import torch.optim as optim
from torch.nn import Module

class QLearningController:
    def __init__(self, param_init, learning_rate=0.1, discount=0.9, epsilon=0.5):
        self.params = param_init
        self.q_table = {}
        self.alpha = learning_rate
        self.gamma = discount
        self.epsilon = epsilon

    def get_state(self, loss, gradient_var):
        return (round(loss.item(), 2), round(gradient_var, 2))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return self.random_action()
        else:
            return self.best_action(state)

    def random_action(self):
        return {param: random.choice([-0.01, 0.0, 0.01]) for param in self.params}

    def best_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = {param: 0.0 for param in self.params}
        return max(self.q_table[state], key=self.q_table[state].get)

    def update_q_value(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in action}
        max_future_q = max(self.q_table[next_state].values(), default=0.0)
        self.q_table[state][action] += self.alpha * (reward + self.gamma * max_future_q - self.q_table[state][action])

    def adjust_params(self, action):
        for param in action:
            self.params[param] += action[param]
        return self.params

class EnhancedSGD(optim.Optimizer):
    def __init__(self, params, model=None, lr=0.01, base_scaling_factor=1.0, momentum=0.9, 
                 smoothing_factor=0.1, decay=0.99, usage_case="LLM", max_steps=100000, 
                 use_amp=False, lookahead_k=5, lookahead_alpha=0.5, apply_noise=True, 
                 adaptive_momentum=True, layer_type_scaling=None, grad_cam_scaling=False):
        
        super(EnhancedSGD, self).__init__(params, defaults=dict(
            lr=lr, base_scaling_factor=base_scaling_factor, momentum=momentum,
            smoothing_factor=smoothing_factor, decay=decay))
        
        # AMP Support
        self.use_amp = use_amp
        if self.use_amp:
            self.scaler = torch.amp.GradScaler()
        
        # Q-Learning controller for adaptive parameter adjustment
        self.q_controller = QLearningController(
            {'C': 0.5, 'r': 0.1, 's': 0.01, 'p': 0.5, 'omega': 0.01}
        )
        
        # Lookahead mechanism
        self.lookahead_k = lookahead_k
        self.lookahead_alpha = lookahead_alpha
        self.slow_weights = [p.clone().detach() for p in self.param_groups[0]['params']]
        self.lookahead_step = 0
        
        # Additional settings
        self.apply_noise = apply_noise
        self.adaptive_momentum = adaptive_momentum
        self.layer_type_scaling = layer_type_scaling
        self.grad_cam_scaling = grad_cam_scaling
        self.set_usage_defaults(usage_case)
        
        # Caching for layer types and depths
        self.param_depths = {}
        self.layer_type_groups = {}
        if model is not None:
            self._map_parameter_depths(model)
            self._create_parameter_groups_by_layer_type(model)
        
        # Internal tracking variables
        self.prev_updates = {}
        self.epoch = 1
        self.max_steps = max_steps
        self.grad_var = 0.0
        self.current_momentum = momentum
        self.prev_loss = float('inf')

    def set_usage_defaults(self, usage_case):
        if usage_case == "LLM":
            self.skip_connection_enabled = True
            self.preconditioner_enabled = True
            self.precondition_frequency = 2
            self.max_precond_dim = 4096
            self.gradient_clipping = True
            self.skip_threshold = 2
        elif usage_case == "VLM":
            self.skip_connection_enabled = True
            self.preconditioner_enabled = True
            self.precondition_frequency = 5
            self.max_precond_dim = 2048
            self.gradient_clipping = True
            self.skip_threshold = 2
        elif usage_case == "GenAI":
            self.skip_connection_enabled = True
            self.preconditioner_enabled = False
            self.precondition_frequency = 3
            self.max_precond_dim = 1024
            self.gradient_clipping = False
            self.skip_threshold = 3
        elif usage_case == "ComputerVision":
            self.skip_connection_enabled = False
            self.preconditioner_enabled = True
            self.precondition_frequency = 10
            self.max_precond_dim = 512
            self.gradient_clipping = True
            self.skip_threshold = 4
        elif usage_case == "ReinforcementLearning":
            self.skip_connection_enabled = False
            self.preconditioner_enabled = False
            self.precondition_frequency = 1
            self.max_precond_dim = 2048
            self.gradient_clipping = False
            self.skip_threshold = 4

    def _map_parameter_depths(self, model, prefix='', depth=0):
        for name, module in model.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            self._map_parameter_depths(module, full_name, depth + 1)

        for name, param in model.named_parameters():
            param_name = f"{prefix}.{name}" if prefix else name
            self.param_depths[param_name] = depth

    def _create_parameter_groups_by_layer_type(self, model):
        self.layer_type_groups = {}
        for name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                full_name = f"{name}.{param_name}" if name else param_name
                layer_type = type(module).__name__
                if layer_type not in self.layer_type_groups:
                    self.layer_type_groups[layer_type] = []
                self.layer_type_groups[layer_type].append(full_name)

    def get_layer_depth(self, param):
        param_name = getattr(param, 'name', None)
        if param_name is None:
            return 0
        return self.param_depths.get(param_name, 0)

    def adaptive_momentum(self, step):
        max_momentum = 0.95
        min_momentum = 0.7
        progress = step / self.max_steps
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        self.current_momentum = min_momentum + (max_momentum - min_momentum) * cosine_decay

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.amp.autocast("cuda") if self.use_amp else torch.no_grad():
                loss = closure()
        
        # Ensure scaling before unscale and update
        if self.use_amp and loss is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self)
        
        for group in self.param_groups:
            effective_lr = group['lr']
            for param in group['params']:
                if param.grad is None:
                    continue
                
                grad = param.grad.data
                grad = self.adaptive_clipping(grad) if self.gradient_clipping else grad
                grad = self.gradient_variance_clipping(grad)
                
                if self.preconditioner_enabled:
                    grad = self.apply_preconditioning(grad, param)
                
                layer_depth = self.get_layer_depth(param)
                if self.skip_connection_enabled and layer_depth >= self.skip_threshold:
                    grad = grad + self.prev_updates.get(param, torch.zeros_like(grad))
                
                if self.apply_noise:
                    grad = self.apply_gradient_noise(grad)
                
                update = self.current_momentum * grad + (1 - self.current_momentum) * grad
                param.data.sub_(effective_lr * update)
                self.prev_updates[param] = update
        
        self.epoch += 1
        
        # Lookahead mechanism
        self.lookahead_step += 1
        if self.lookahead_step % self.lookahead_k == 0:
            for slow, param in zip(self.slow_weights, self.param_groups[0]['params']):
                slow.data.add_(self.lookahead_alpha * (param.data - slow.data))
                param.data.copy_(slow.data)
        
        if self.use_amp and loss is not None:
            self.scaler.step(self)
            self.scaler.update()
        
        return loss

    def adaptive_clipping(self, grad):
        self.grad_var = 0.9 * self.grad_var + 0.1 * grad.var()
        clip_value = math.sqrt(self.grad_var) * 2
        return torch.clamp(grad, -clip_value, clip_value)

    def gradient_variance_clipping(self, grad, threshold=1.0):
        grad_std = grad.std()
        grad_mean = grad.mean()
        clipped_grad = torch.clamp(grad, grad_mean - threshold * grad_std, grad_mean + threshold * grad_std)
        return clipped_grad

    def apply_gradient_noise(self, grad, std_dev=0.01):
        noise = torch.randn_like(grad) * std_dev
        return grad + noise

    def apply_preconditioning(self, grad, param):
        state = self.state[param]
        if 'preconditioner' not in state:
            self.init_preconditioner(grad, state)

        precond_grad = grad
        for preconditioner in state['preconditioner']:
            if preconditioner is not None:
                precond_grad = torch.matmul(precond_grad, preconditioner)
        return precond_grad

    def init_preconditioner(self, grad, state):
        state['preconditioner'] = []
        for dim in grad.shape:
            if dim <= self.max_precond_dim:
                state['preconditioner'].append(torch.eye(dim, device=grad.device))
            else:
                state['preconditioner'].append(None)
        state['step'] = 0

# Training loop example with AMP support
def train(model, optimizer, train_loader, criterion, num_epochs=5):
    model.train()
    train_losses = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            
            # Handle AMP context if enabled
            if hasattr(optimizer, 'use_amp') and optimizer.use_amp:
                with torch.amp.autocast("cuda"):
                    output = model(data)
                    loss = criterion(output, target)
                optimizer.scaler.scale(loss).backward()
            else:
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
            
            optimizer.step()
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")
    return train_losses
