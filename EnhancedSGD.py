import torch
import math
import random
import torch.optim as optim
from scipy.stats import norm

class QLearningController:
    def __init__(self, param_init, learning_rate=0.1, discount=0.9, epsilon=0.5):
        self.params = param_init
        self.q_table = {}
        self.alpha = learning_rate
        self.gamma = discount
        self.epsilon = epsilon

    def get_state(self, loss, gradient_var):
        # Define a state based on rounded loss and gradient variance
        return (round(loss.item(), 2), round(gradient_var, 2))

    def choose_action(self, state):
        # Use epsilon-greedy strategy to choose action
        if random.uniform(0, 1) < self.epsilon:
            return self.random_action()
        else:
            return self.best_action(state)

    def random_action(self):
        # Random exploration in action space
        return {param: random.choice([-0.01, 0.0, 0.01]) for param in self.params}

    def best_action(self, state):
        # Choose best action for given state based on Q-values
        if state not in self.q_table:
            self.q_table[state] = {param: 0.0 for param in self.params}
        return max(self.q_table[state], key=self.q_table[state].get)

    def update_q_value(self, state, action, reward, next_state):
        # Update Q-table with reward and discounted future Q-value
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in action}
        max_future_q = max(self.q_table[next_state].values(), default=0.0)
        self.q_table[state][action] += self.alpha * (reward + self.gamma * max_future_q - self.q_table[state][action])

    def adjust_params(self, action):
        # Adjust parameters according to chosen action
        for param in action:
            self.params[param] += action[param]
        return self.params


class EnhancedSGD(optim.Optimizer):
    def __init__(self, params, model=None, lr=0.01, base_scaling_factor=1.0, momentum=0.9,
                 smoothing_factor=0.1, decay=0.99, usage_case="LLM", max_steps=100000,
                 use_amp=False, lookahead_k=5, lookahead_alpha=0.5, apply_noise=True,
                 adaptive_momentum=True, layer_type_scaling=None, grad_cam_scaling=False):
        
        defaults = dict(
            lr=lr,
            base_scaling_factor=base_scaling_factor,
            momentum=momentum,
            smoothing_factor=smoothing_factor,
            decay=decay,
            weight_decay=0.01  # Default weight decay if not explicitly provided
        )
        super(EnhancedSGD, self).__init__(params, defaults)
        
        # Initialize optimizer state
        self.state['step_count'] = 0
        self.state['grad_history'] = []
        self.state['lr_history'] = []
        self.grad_var = 0.0
        self.prev_grad_var = 0.0
        self.current_momentum = momentum

        # Q-Learning controller for adaptive parameter adjustment
        self.q_controller = QLearningController(
            {'lr_scale': 1.0, 'momentum_scale': 1.0, 'grad_scale': 1.0}
        )

        # AMP, Lookahead, and Noise settings
        self.use_amp = use_amp
        if self.use_amp:
            self.scaler = torch.amp.GradScaler()

        self.lookahead_k = lookahead_k
        self.lookahead_alpha = lookahead_alpha
        self.slow_weights = [p.clone().detach() for p in self.param_groups[0]['params']]
        for sw in self.slow_weights:
            sw.requires_grad = False
        self.lookahead_step = 0

        # Adaptive, noise, and scaling settings
        self.apply_noise = apply_noise
        self.adaptive_momentum = adaptive_momentum
        self.layer_type_scaling = layer_type_scaling
        self.grad_cam_scaling = grad_cam_scaling
        self.set_usage_defaults(usage_case)

        # Layer and type mapping for adaptive scaling
        self.param_depths = {}
        self.layer_type_groups = {}
        if model is not None:
            self._map_parameter_depths(model)
            self._create_parameter_groups_by_layer_type(model)

        # Internal tracking variables
        self.prev_updates = {}
        self.epoch = 1
        self.max_steps = max_steps
        self.prev_loss = float('inf')

    def set_usage_defaults(self, usage_case):
        # Define defaults for various use cases
        if usage_case == "LLM":
            self.gradient_clipping = True
            self.max_precond_dim = 4096
        elif usage_case == "VLM":
            self.gradient_clipping = True
            self.max_precond_dim = 2048
        elif usage_case == "GenAI":
            self.gradient_clipping = False
            self.max_precond_dim = 1024
        elif usage_case == "ComputerVision":
            self.gradient_clipping = True
            self.max_precond_dim = 512
        elif usage_case == "ReinforcementLearning":
            self.gradient_clipping = False
            self.max_precond_dim = 2048

    def _map_parameter_depths(self, model, prefix='', depth=0):
        # Recursively map parameter depths
        for name, module in model.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            self._map_parameter_depths(module, full_name, depth + 1)

        for name, param in model.named_parameters():
            param_name = f"{prefix}.{name}" if prefix else name
            self.param_depths[param_name] = depth

    def _create_parameter_groups_by_layer_type(self, model):
        # Group parameters by layer type
        self.layer_type_groups = {}
        for name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                full_name = f"{name}.{param_name}" if name else param_name
                layer_type = type(module).__name__
                if layer_type not in self.layer_type_groups:
                    self.layer_type_groups[layer_type] = []
                self.layer_type_groups[layer_type].append(full_name)

    def get_layer_depth(self, param):
        # Retrieve depth for given parameter
        param_name = getattr(param, 'name', None)
        if param_name is None:
            return 0
        return self.param_depths.get(param_name, 0)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.amp.autocast("cuda") if self.use_amp else torch.no_grad():
                loss = closure()

        self.state['step_count'] += 1
        effective_lr = self.param_groups[0]['lr']

        # Gradient collection for variance calculation
        total_grad_norm = 0.0
        grad_list = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                grad_list.append(grad.view(-1))
                total_grad_norm += grad.norm(2).item()

        # Calculate and apply gradient variance
        if grad_list:
            all_grads = torch.cat(grad_list)
            self.prev_grad_var = self.grad_var
            self.grad_var = all_grads.var().item()

            # Store in history (limited size)
            self.state['grad_history'].append(self.grad_var)
            if len(self.state['grad_history']) > 100:
                self.state['grad_history'].pop(0)

        # Q-Learning state and action
        if loss is not None:
            current_state = self.q_controller.get_state(loss, self.grad_var)
            action = self.q_controller.choose_action(current_state)

            # Apply Q-learning adjustments
            base_lr = self.param_groups[0]['lr']
            lr_adjustment = action['lr_scale']
            momentum_adjustment = action['momentum_scale']

            # Adaptive learning rate based on gradient variance
            grad_var_change = self.grad_var - self.prev_grad_var
            lr_scale = 1.0
            if grad_var_change > 0:
                # Increase in variance - reduce learning rate
                lr_scale = max(0.5, 1.0 - grad_var_change)
            else:
                # Stable/decreasing variance - increase learning rate
                lr_scale = min(1.5, 1.0 + abs(grad_var_change))

            # Apply adjustments
            effective_lr = base_lr * lr_adjustment * lr_scale
            self.current_momentum = self.defaults['momentum'] * momentum_adjustment

            # Store history
            self.state['lr_history'].append(effective_lr)
            if len(self.state['lr_history']) > 100:
                self.state['lr_history'].pop(0)

        # Apply updates with adjusted parameters
        for group in self.param_groups:
            group['lr'] = effective_lr
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data

                # Adaptive gradient clipping
                if self.gradient_clipping:
                    grad = self.adaptive_clipping(grad)

                # Initialize momentum buffer
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p.data)

                momentum_buffer = state['momentum_buffer']
                # Update with momentum
                momentum_buffer.mul_(self.current_momentum).add_(grad)
                p.data.add_(momentum_buffer, alpha=-group['lr'])

        # Update Q-learning values
        if loss is not None:
            next_state = self.q_controller.get_state(loss, self.grad_var)
            reward = -abs(grad_var_change)  # Stability in gradient variance
            self.q_controller.update_q_value(current_state, action, reward, next_state)

        return loss

    def adaptive_clipping(self, grad):
        # Adaptive clipping based on variance
        if self.grad_var > 0:
            clip_value = math.sqrt(self.grad_var) * 2
            return torch.clamp(grad, -clip_value, clip_value)
        return grad

    def bayesian_initialize_params(self):
        for param, init_val in self.q_controller.params.items():
            self.q_controller.params[param] = norm.rvs(loc=init_val, scale=0.05)
