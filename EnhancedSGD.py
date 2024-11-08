# EnhancedSGD.py

import torch
import math
import random
import torch.optim as optim
from scipy.stats import norm

class QLearningController:
    """
    Q-Learning Controller to adaptively adjust optimizer parameters based on training dynamics.
    """
    def __init__(self, param_init, learning_rate=0.1, discount=0.9, epsilon=0.5):
        """
        Initializes the Q-Learning controller.

        Args:
            param_init (dict): Initial parameter scales (e.g., {'lr_scale': 1.0, 'momentum_scale': 1.0, 'grad_scale': 1.0}).
            learning_rate (float): Learning rate for Q-learning updates.
            discount (float): Discount factor for future rewards.
            epsilon (float): Exploration rate for epsilon-greedy strategy.
        """
        # Ensure all initial parameters are floats
        self.params = {k: float(v) for k, v in param_init.items()}
        self.q_table = {}
        self.alpha = float(learning_rate)
        self.gamma = float(discount)
        self.epsilon = float(epsilon)

    def get_state(self, loss, gradient_var, epoch, layer_depth):
        """
        Defines the state based on rounded loss, gradient variance, epoch, and layer depth.

        Args:
            loss (torch.Tensor): Current loss value.
            gradient_var (float): Current gradient variance.
            epoch (int): Current epoch number.
            layer_depth (int): Depth of the layer in the model.

        Returns:
            tuple: Rounded loss, gradient variance, epoch, and layer depth as the state.
        """
        return (
            round(float(loss.item()), 2),
            round(float(gradient_var), 2),
            int(epoch),
            int(layer_depth)
        )

    def choose_action(self, state):
        """
        Chooses an action based on the epsilon-greedy strategy.

        Args:
            state (tuple): Current state.

        Returns:
            dict: Chosen action as parameter adjustments.
        """
        if random.uniform(0, 1) < self.epsilon:
            return self.random_action()
        else:
            return self.best_action(state)

    def random_action(self):
        """
        Generates a random action for exploration.

        Returns:
            dict: Random adjustments for each parameter.
        """
        return {param: random.choice([-0.01, 0.0, 0.01]) for param in self.params}

    def best_action(self, state):
        """
        Chooses the best action based on the current Q-table.

        Args:
            state (tuple): Current state.

        Returns:
            dict: Best action for the given state.
        """
        if state not in self.q_table:
            self.q_table[state] = {param: 0.0 for param in self.params}
        best_param = max(self.q_table[state], key=self.q_table[state].get)
        return {best_param: float(self.q_table[state][best_param])}

    def update_q_value(self, state, action, reward, next_state):
        """
        Updates the Q-table with the received reward and the maximum future Q-value.

        Args:
            state (tuple): Previous state.
            action (dict): Action taken.
            reward (float): Reward received.
            next_state (tuple): New state after taking the action.
        """
        reward = float(reward)
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.params}
        max_future_q = max(self.q_table.get(next_state, {}).values(), default=0.0)
        for param, value in action.items():
            self.q_table[state][param] += self.alpha * (reward + self.gamma * max_future_q - self.q_table[state][param])

    def adjust_params(self, action):
        """
        Adjusts the optimizer's parameters based on the chosen action.

        Args:
            action (dict): Parameter adjustments.

        Returns:
            dict: Updated parameters.
        """
        for param in action:
            self.params[param] += float(action[param])
        return self.params


class EnhancedSGD(optim.Optimizer):
    """
    Enhanced Stochastic Gradient Descent optimizer with adaptive learning rate and momentum adjustments via Q-Learning.
    """
    def __init__(self, params, model=None, lr=0.01, base_scaling_factor=1.0, momentum=0.9,
                 smoothing_factor=0.1, decay=0.99, usage_case="LLM", max_steps=100000,
                 use_amp=False, lookahead_k=5, lookahead_alpha=0.5, apply_noise=True,
                 adaptive_momentum=True, layer_type_scaling=None, grad_cam_scaling=False,
                 gradient_centering=True, noise_scale=1e-4):
        """
        Initializes the EnhancedSGD optimizer.

        Args:
            params (iterable): Iterable of parameters to optimize.
            model (nn.Module, optional): The model being optimized.
            lr (float, optional): Learning rate.
            base_scaling_factor (float, optional): Base scaling factor for learning rate.
            momentum (float, optional): Momentum factor.
            smoothing_factor (float, optional): Smoothing factor for moving averages.
            decay (float, optional): Decay rate for moving averages.
            usage_case (str, optional): Use case scenario to set defaults.
            max_steps (int, optional): Maximum number of optimization steps.
            use_amp (bool, optional): Use Automatic Mixed Precision.
            lookahead_k (int, optional): Number of steps for lookahead.
            lookahead_alpha (float, optional): Alpha parameter for lookahead.
            apply_noise (bool, optional): Apply noise to gradients.
            adaptive_momentum (bool, optional): Use adaptive momentum scaling.
            layer_type_scaling (dict, optional): Scaling factors per layer type.
            grad_cam_scaling (bool, optional): Enable Grad-CAM scaling.
            gradient_centering (bool, optional): Centralize gradients.
            noise_scale (float, optional): Scale of noise to inject.
        """
        defaults = dict(
            lr=float(lr),
            base_scaling_factor=float(base_scaling_factor),
            momentum=float(momentum),
            smoothing_factor=float(smoothing_factor),
            decay=float(decay),
            weight_decay=0.01  # Default weight decay if not explicitly provided
        )
        super(EnhancedSGD, self).__init__(params, defaults)
        
        # Initialize optimizer state
        self.state['step_count'] = 0
        self.state['grad_history'] = []
        self.state['lr_history'] = []
        self.grad_var = 0.0
        self.prev_grad_var = 0.0
        self.current_momentum = float(momentum)
        self.smoothing_factor = float(smoothing_factor)
        self.gradient_centering = bool(gradient_centering)
        self.noise_scale = float(noise_scale)

        # Q-Learning controller for adaptive parameter adjustment
        self.q_controller = QLearningController(
            {'lr_scale': 1.0, 'momentum_scale': 1.0, 'grad_scale': 1.0}
        )

        # AMP, Lookahead, and Noise settings
        self.use_amp = bool(use_amp)
        if self.use_amp:
            # Updated GradScaler initialization to comply with latest PyTorch API
            self.scaler = torch.amp.GradScaler()

        self.lookahead_k = int(lookahead_k)
        self.lookahead_alpha = float(lookahead_alpha)
        self.slow_weights = [p.clone().detach() for p in self.param_groups[0]['params']]
        for sw in self.slow_weights:
            sw.requires_grad = False
        self.lookahead_step = 0

        # Adaptive, noise, and scaling settings
        self.apply_noise = bool(apply_noise)
        self.adaptive_momentum = bool(adaptive_momentum)
        self.layer_type_scaling = layer_type_scaling
        self.grad_cam_scaling = bool(grad_cam_scaling)
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
        self.max_steps = int(max_steps)
        self.prev_loss = float('inf')

    def set_usage_defaults(self, usage_case):
        """
        Sets default parameters based on the usage case.

        Args:
            usage_case (str): The usage scenario to set defaults for.
        """
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
        else:
            # Default settings
            self.gradient_clipping = True
            self.max_precond_dim = 1024

    def _map_parameter_depths(self, model, prefix='', depth=0):
        """
        Recursively maps parameter depths for adaptive scaling.

        Args:
            model (nn.Module): The model to map parameters for.
            prefix (str, optional): Prefix for parameter names.
            depth (int, optional): Current depth in the module hierarchy.
        """
        for name, module in model.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            self._map_parameter_depths(module, full_name, depth + 1)

        for name, param in model.named_parameters():
            param_name = f"{prefix}.{name}" if prefix else name
            self.param_depths[param_name] = depth

    def _create_parameter_groups_by_layer_type(self, model):
        """
        Groups parameters by their layer types for adaptive scaling.

        Args:
            model (nn.Module): The model to group parameters for.
        """
        self.layer_type_groups = {}
        for name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                full_name = f"{name}.{param_name}" if name else param_name
                layer_type = type(module).__name__
                if layer_type not in self.layer_type_groups:
                    self.layer_type_groups[layer_type] = []
                self.layer_type_groups[layer_type].append(full_name)

    def get_layer_depth(self, param):
        """
        Retrieves the depth of a given parameter.

        Args:
            param (torch.nn.Parameter): The parameter to retrieve depth for.

        Returns:
            int: Depth level of the parameter.
        """
        param_name = getattr(param, 'name', None)
        if param_name is None:
            return 0
        return self.param_depths.get(param_name, 0)

    def step(self, closure=None):
        """
        Performs a single optimization step.
    
        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
    
        Returns:
            torch.Tensor or None: The loss if closure is provided, else None.
        """
        loss = None
        if closure is not None:
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    loss = closure()
            else:
                with torch.no_grad():
                    loss = closure()
    
        self.state['step_count'] += 1
        effective_lr = float(self.param_groups[0]['lr'])
    
        # Gradient collection for variance calculation
        grad_list = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                grad_list.append(grad.view(-1))
    
        # Calculate gradient variance with exponential moving average
        if grad_list:
            all_grads = torch.cat(grad_list)
            current_grad_var = float(all_grads.var().item())
            self.grad_var = self.smoothing_factor * current_grad_var + (1 - self.smoothing_factor) * self.grad_var
    
            # Store gradient variance in history (limit to 100 entries)
            self.state['grad_history'].append(self.grad_var)
            if len(self.state['grad_history']) > 100:
                self.state['grad_history'].pop(0)
    
        # Apply Q-Learning adjustments if loss is calculated
        if loss is not None:
            try:
                epoch = int(self.epoch)
                layer_depth = 1  # Placeholder; adjust as necessary for actual layer depth
                current_state = self.q_controller.get_state(loss, self.grad_var, epoch, layer_depth)
                action = self.q_controller.choose_action(current_state)
    
                # Ensure adjustments are floats
                base_lr = float(self.param_groups[0]['lr'])
                lr_adjustment = float(action.get('lr_scale', 1.0))
                momentum_adjustment = float(action.get('momentum_scale', 1.0))
    
                grad_var_change = float(self.grad_var - self.prev_grad_var)
                lr_scale = 1.0
                if grad_var_change > 0:
                    lr_scale = max(0.5, 1.0 - grad_var_change)
                else:
                    lr_scale = min(1.5, 1.0 + abs(grad_var_change))
    
                effective_lr = base_lr * lr_adjustment * lr_scale
                if self.adaptive_momentum:
                    self.current_momentum = float(self.defaults['momentum'] * momentum_adjustment)
    
                print(f"Effective LR: {effective_lr}, Grad Var Change: {grad_var_change}, Momentum Adjustment: {momentum_adjustment}")
    
            except Exception as e:
                print(f"Error in Q-Learning adjustment: {e}")
                raise
    
        # Apply updates with adjusted parameters
        grad_means = []
        for group in self.param_groups:
            group['lr'] = effective_lr
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
    
                if self.gradient_centering:
                    grad = grad - grad.mean()
    
                if self.gradient_clipping:
                    grad = self.adaptive_clipping(grad)
    
                if self.apply_noise:
                    noise = torch.randn_like(grad) * self.noise_scale
                    grad = grad + noise
    
                state = self.state.setdefault(p, {})
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p.data)
    
                momentum_buffer = state['momentum_buffer']
                momentum_buffer.mul_(self.current_momentum).add_(grad)
                p.data.add_(momentum_buffer, alpha=-group['lr'])
    
                grad_means.append(float(grad.mean()))  # Collect for summary
    
        # Summarized batch stats
        if len(grad_means) > 0:
            avg_grad_mean = sum(grad_means) / len(grad_means)
            #print(f"Batch Summary - Avg Grad Mean: {avg_grad_mean:.6e}, LR: {effective_lr:.4e}, Momentum Buffer Type: {type(momentum_buffer)}")
    
        if loss is not None:
            try:
                next_state = self.q_controller.get_state(loss, self.grad_var, epoch, layer_depth)
                reward = -abs(grad_var_change)
                self.q_controller.update_q_value(current_state, action, reward, next_state)
            except Exception as e:
                print(f"Error in Q-learning update: {e}")
                raise
    
        return loss if isinstance(loss, torch.Tensor) else None
    
    def adaptive_clipping(self, grad):
        """
        Applies adaptive gradient clipping based on gradient variance.

        Args:
            grad (torch.Tensor): The gradient tensor.

        Returns:
            torch.Tensor: Clipped gradient tensor.
        """
        if self.grad_var > 0:
            clip_value = math.sqrt(self.grad_var) * 2
            return torch.clamp(grad, -clip_value, clip_value)
        return grad

    def bayesian_initialize_params(self):
        """
        Initializes parameters using a Bayesian approach with normal distributions.
        """
        for param, init_val in self.q_controller.params.items():
            self.q_controller.params[param] = float(norm.rvs(loc=init_val, scale=0.05))

    def load_state_dict(self, state_dict):
        """
        Loads the state dictionary and ensures any new keys introduced in EnhancedSGD are handled.

        Args:
            state_dict (dict): State dictionary.
        """
        super().load_state_dict(state_dict)
        # Ensure any new keys introduced in EnhancedSGD are handled
        for group in self.param_groups:
            for p in group['params']:
                self.state.setdefault(p, {})
