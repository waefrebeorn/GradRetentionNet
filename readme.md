# RescaledSGD vs. StandardSGD: MNIST Training with Real-Time Visualization

![Project Banner](https://github.com/yourusername/your-repo/blob/main/banner.png)

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Visualization](#visualization)
- [Credits](#credits)
- [License](#license)

## Introduction

Welcome to the **RescaledSGD vs. StandardSGD** project! This project demonstrates the implementation and comparison of a custom optimizer, **RescaledSGD**, against the standard **SGD** optimizer in training a simple neural network on the MNIST dataset. Additionally, it provides a real-time graphical user interface (GUI) to visualize parameter updates, gradients, and effective learning rates during training.

## Features

- **Custom Optimizer (`RescaledSGD`)**: Dynamically adjusts learning rates based on gradient magnitudes, aiming for more effective and stable training.
- **Standard Optimizer (`SGD`)**: Serves as a baseline for comparison with a fixed learning rate.
- **Real-Time Visualization**: Utilizes Tkinter and Matplotlib to display parameter values, gradients, and effective learning rates in real-time.
- **Multi-Threaded Training**: Ensures the GUI remains responsive by running the training process in a separate thread.
- **User Controls**: Interactive sliders and input fields allow users to adjust learning rates and visualization parameters on the fly.

## Installation

### Prerequisites

Ensure you have Python 3.6 or later installed. It's recommended to use a virtual environment to manage dependencies.

### Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```

2. **Create and Activate a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   *If you don't have a `requirements.txt`, you can install the necessary packages individually:*
   ```bash
   pip install torch torchvision matplotlib
   ```

   > **Note:** `tkinter` is usually included with Python installations. If it's missing, refer to your operating system's instructions to install it.

## Usage

1. **Activate the Virtual Environment**
   ```bash
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Run the Script**
   ```bash
   python main.py
   ```

   This will launch the GUI and start the training process in the background. You will see real-time updates of parameter changes, gradients, and effective learning rates for both `RescaledSGD` and `StandardSGD` optimizers.

## Visualization

The GUI provides a comprehensive view of the training dynamics:

- **Tabs for Each Optimizer**: Switch between `RescaledSGD` and `StandardSGD` to observe their behaviors.
- **Parameter Bars**: Blue bars represent current parameter values, while green bars show updated values after applying gradients.
- **Effective Learning Rates**: Dashed orange lines indicate the learning rates applied to each parameter.
- **Arrows**: Red arrows illustrate the direction and magnitude of parameter updates.
- **Controls**:
  - **Learning Rate Slider**: Adjust the effective learning rate manually.
  - **Retain Minimum Scaling**: Toggle whether to retain minimum scaling in learning rate adjustments.
  - **Base Learning Rate Input**: Set the base learning rate for `RescaledSGD`.
  - **Y-Range Multiplier Input**: Modify the Y-axis scaling for better visualization.

After training completes, a separate plot displays the training and validation losses for both optimizers across all epochs, allowing for an in-depth comparison of their performances.

## Credits

- **WuBu WaefreBeorn**: Lead Developer, Implemented and Bug-Fixed the Project.
- **Kalomaze**: Conceptual Contributor, Provided General Ideas and Guidance.
- **OpenAI ChatGPT**: Assisted in Code Generation and Implementation Strategies.

A special thanks to both WuBu WaefreBeorn and Kalomaze for their collaboration and dedication to making this project successful.

## License

This project is licensed under the [MIT License](LICENSE).

---

*Feel free to contribute, report issues, or suggest improvements!*

# Acknowledgements

- **OpenAI** for providing the powerful language model, ChatGPT, which assisted in generating and refining the code.
- The **PyTorch** and **Matplotlib** communities for their excellent libraries and documentation.

---

