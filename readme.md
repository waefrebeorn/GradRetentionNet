# GradRetentionNet: Dynamic Gradient Retention and Optimization Framework for Enhanced Model Convergence

GradRetentionNet is a cutting-edge framework designed to explore advanced optimization strategies in deep learning, integrating a unique approach to gradient retention and optimizer control through **EnhancedSGD**. Leveraging Q-learning-based adaptive adjustments, gradient variance tracking, and Bayesian parameter initialization, EnhancedSGD facilitates faster and more stable convergence across diverse neural network models. This framework is suited for researchers interested in advanced model training analysis, providing detailed logging, memory efficiency, and comparative insights across multiple optimizers and datasets.

## Table of Contents
1. [Introduction](#introduction)
2. [EnhancedSGD: Reinforcement-Learning-Based Optimizer](#enhancedsgd-reinforcement-learning-based-optimizer)
3. [Core Equations](#core-equations)
4. [Features](#features)
5. [Supported Datasets and Models](#supported-datasets-and-models)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Experiments and Results](#experiments-and-results)
9. [Acknowledgements](#acknowledgements)

---

## Introduction

Optimization is central to machine learning, impacting training speed, convergence stability, and model performance. GradRetentionNet addresses limitations in traditional optimizers by introducing **EnhancedSGD**—a novel optimizer that combines elements of **Q-learning** and **stochastic gradient descent (SGD)** to improve adaptability. Using dynamic adjustments based on gradient variance and learning rate scaling, EnhancedSGD is designed to adapt to diverse tasks, especially in noisy or complex data environments. This framework supports popular datasets and models for image classification, sentiment analysis, and segmentation, allowing for comprehensive testing across both vision and NLP domains.

---

## EnhancedSGD: Reinforcement-Learning-Based Optimizer

EnhancedSGD is the foundation of GradRetentionNet’s optimization approach, aiming to stabilize and accelerate convergence through adaptive learning strategies:

1. **Q-Learning-Based Adjustments**: EnhancedSGD integrates a **Q-Learning Controller** that adaptively adjusts learning rate (`lr_scale`), momentum (`momentum_scale`), and gradient scaling (`grad_scale`) based on training state variables such as **loss** and **gradient variance**. This controller operates through epsilon-greedy action selection, optimizing for actions that maximize stability and performance:
   \[
   Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
   \]
   where \( Q(s, a) \) is the expected reward for taking action \( a \) in state \( s \), \( \alpha \) is the learning rate, and \( \gamma \) is the discount factor.

2. **Gradient Variance Tracking**: By calculating the **gradient variance** with an exponential moving average, EnhancedSGD can assess model stability and adjust the learning rate accordingly. This helps mitigate issues where gradients become unstable, leading to improved convergence rates:
   \[
   \sigma^2_{\text{grad}} \leftarrow \beta \sigma^2_{\text{grad}} + (1 - \beta) \text{Var}(g)
   \]
   where \( \sigma^2_{\text{grad}} \) is the smoothed variance, \( \beta \) is the smoothing factor, and \( \text{Var}(g) \) represents the variance of gradients.

3. **Adaptive Clipping and Noise Injection**: EnhancedSGD incorporates **adaptive gradient clipping** based on gradient variance and **Bayesian noise injection** to prevent overfitting and improve generalization, especially in complex datasets.

4. **Bayesian Parameter Initialization**: To improve exploration during training, parameters are initialized using a normal distribution based on initial values:
   \[
   \theta \sim \mathcal{N}(\mu, \sigma^2)
   \]
   where \( \mu \) is the initial parameter value, and \( \sigma \) controls variability, helping avoid local minima in the loss landscape.

---

## Core Equations

**1. Q-Learning Update Rule**  
In EnhancedSGD, the Q-Learning Controller uses an update rule to optimize parameter adjustments:
\[
Q(s, a) = Q(s, a) + \alpha \left( r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right)
\]
where:
- \( Q(s, a) \) is the quality of action \( a \) in state \( s \),
- \( \alpha \) is the learning rate of the Q-learning agent,
- \( \gamma \) is the discount factor for future rewards,
- \( r \) is the reward obtained after taking action \( a \),
- \( s' \) is the next state, and \( a' \) is the optimal action in \( s' \).

**2. Gradient Variance-Based Learning Rate Adjustment**
To scale the learning rate based on gradient variance, EnhancedSGD calculates:
\[
\text{effective\_lr} = \text{lr} \times \left(1 \pm \frac{\Delta \sigma^2_{\text{grad}}}{\sigma^2_{\text{grad}}}\right)
\]
where \( \sigma^2_{\text{grad}} \) is the smoothed gradient variance.

**3. Bayesian Noise Injection**  
Bayesian initialization helps explore parameter space:
\[
\theta \sim \mathcal{N}(\mu, \sigma^2)
\]
where each parameter \( \theta \) is initialized with a variance based on the Bayesian prior.

---

## Features

1. **Flexible Dataset Loading**: Supports both Hugging Face datasets and CSV-based loading, ensuring broad applicability across different research settings.
2. **Memory and Gradient Tracking**: Real-time tracking of VRAM/RAM usage, gradient mean, and gradient variance per batch and epoch.
3. **Grad-CAM Visualization**: Visualizations for segmentation tasks to highlight the regions influencing model decisions.
4. **Multi-Optimizer Support**: Compare EnhancedSGD with standard optimizers (SGD, Adam, RMSprop) across various datasets.
5. **Comprehensive Logging and Analytics**: Extensive logging options to track test accuracy, memory changes, epoch times, and gradient variance over time.

---

## Supported Datasets and Models

GradRetentionNet accommodates a wide range of tasks, allowing for extensive analysis across different data domains.

### Datasets
- **MNIST** (Image Classification)
- **CIFAR-10** (Image Classification)
- **IMDB** (Sentiment Analysis)
- **AG_NEWS** (Topic Classification)
- **Pascal VOC** (Image Segmentation)

### Models
- **SimpleCNN** (for MNIST, CIFAR-10)
- **BERT-based TextClassifier** (for IMDB, AG_NEWS)
- **SimpleUNet** (for Pascal VOC segmentation)

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/waefrebeorn/GradRetentionNet.git
   cd GradRetentionNet
   ```

2. **Set Up Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Datasets**:
   Place preprocessed CSV files in the `data/` directory for IMDB and AG_NEWS.

---

## Usage

1. **Run Main Script**:
   ```bash
   python main.py
   ```
   Select datasets and optimizers through command prompts or specify `all` to run all configurations.

2. **Result Logs and Visualizations**:
   Results are saved to `results/`, providing detailed analytics for each experiment run. This includes metrics for test accuracy, training loss, memory usage, and time per epoch.

---

## Experiments and Results

The primary experimental focus in GradRetentionNet is on:
- **Training Efficiency**: Testing optimizer performance across datasets.
- **Memory Usage**: Monitoring VRAM/RAM during model training.
- **Adaptive Learning Dynamics**: Evaluating the impact of EnhancedSGD’s dynamic learning rate and gradient variance tracking on convergence stability.
- **Visual Explanations**: Grad-CAM results highlight regions of focus in segmentation tasks.

EnhancedSGD has shown improvements in convergence speed and memory stability, especially in complex datasets like Pascal VOC and AG_NEWS, due to its unique handling of gradient variance.

---

## Acknowledgements

Special thanks to **Hugging Face** for their datasets library, which enabled seamless integration of NLP datasets like IMDB and AG_NEWS. **PyTorch** provided the foundation for model implementation, while **SciPy** supported Bayesian sampling for optimizer initialization. Our approach was also inspired by reinforcement learning techniques in optimization research, making EnhancedSGD an example of applying **Q-learning** in practical, scalable scenarios.

---

GradRetentionNet is designed as a robust platform for experimentation and research in optimization. The project showcases innovative strategies to control gradient variance and retain stability during training, making it suitable for researchers aiming to improve model convergence under challenging conditions.

MIT LICENSE