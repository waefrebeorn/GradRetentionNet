# GradRetentionNet


## Overview

**GradRetentionNet** is a pioneering project that introduces the **RescaledSGD** optimizer, inspired by the **Retention Net** concept. This custom optimizer dynamically adjusts learning rates based on gradient magnitudes, enhancing the training efficiency and performance of neural networks. By retaining and scaling gradients, **RescaledSGD** offers a more adaptive optimization strategy compared to the traditional **Stochastic Gradient Descent (SGD)**.

The project includes a user-friendly graphical interface built with Tkinter, allowing real-time visualization of parameter updates, gradients, and effective learning rates during the training process. This visual feedback aids in understanding the optimization dynamics and the impact of the RescaledSGD optimizer.

## Features

- **RescaledSGD Optimizer**: Implements a gradient retention and dynamic learning rate scaling mechanism.
- **Real-Time Visualization**: Interactive GUI to monitor parameter changes, gradients, and effective learning rates.
- **Comparison with Standard SGD**: Demonstrates the advantages of RescaledSGD over traditional SGD on the MNIST dataset.
- **User Controls**: Adjustable learning rate settings and scaling parameters to experiment with optimization behavior.
- **Threaded Training**: Ensures a responsive GUI by running the training process in a separate thread.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- **Python 3.6 or higher**
- **pip** (Python package installer)

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/waefrebeorn/GradRetentionNet.git
   cd GradRetentionNet
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   *If `requirements.txt` is not provided, install the necessary packages manually:*

   ```bash
   pip install torch torchvision matplotlib
   ```

   *Note: `tkinter` is typically included with standard Python installations. If not, install it based on your operating system.*

### Running the Application

Execute the main script to start training and launch the GUI:

```bash
python main.py
```

Upon running, a GUI window will appear, displaying real-time graphs of parameter updates, gradients, and effective learning rates for both **RescaledSGD** and **StandardSGD** optimizers.

## Usage

The GUI provides several interactive controls to customize the training visualization:

- **Learning Rate Slider**: Adjusts the effective learning rate manually.
- **Retain Minimum Scaling**: Toggles whether to retain the minimum scaling factor in learning rate adjustments.
- **Base Learning Rate Input**: Sets the base learning rate (`base_lr`) for **RescaledSGD**.
- **Y-Range Multiplier Input**: Modifies the Y-axis scaling for better visualization.

These controls allow users to experiment with different optimization settings and observe their effects in real-time.

## RescaledSGD Optimizer

The **RescaledSGD** optimizer is the core innovation of this project, embodying the **Retention Net** idea. Here's how it works:

1. **Gradient Retention**: Instead of discarding gradients after each update, **RescaledSGD** retains a persistent gradient that decays over time. This retention helps in smoothing out gradient fluctuations and maintaining a memory of past gradients.

2. **Dynamic Learning Rate Scaling**:
   - **Base Learning Rate (`base_lr`)**: The minimum learning rate applied to parameters with the smallest gradients.
   - **Peak Learning Rate (`peak_lr`)**: The maximum learning rate applied to parameters with the largest gradients.
   - **Scaling Mechanism**: For each parameter, the effective learning rate is scaled between `base_lr` and `peak_lr` based on the relative magnitude of its retained gradient. This ensures that parameters with larger gradients receive larger updates, facilitating faster convergence.

3. **Decay Factor (`decay`)**: Controls the rate at which the retained gradients decay over time, influencing the optimizer's sensitivity to recent gradient information.

**Benefits of RescaledSGD**:

- **Adaptive Updates**: Tailors the learning rate for each parameter individually, enhancing optimization efficiency.
- **Improved Convergence**: Facilitates faster and more stable convergence by balancing aggressive and conservative updates based on gradient magnitudes.
- **Enhanced Generalization**: Demonstrated better performance on validation and test datasets compared to standard SGD, indicating improved generalization capabilities.

## Results

In experiments conducted on the MNIST dataset, **RescaledSGD** significantly outperformed **StandardSGD**:

- **Validation Accuracy**: Achieved up to 92% accuracy with **RescaledSGD**, compared to 53% with **StandardSGD**.
- **Test Accuracy**: Reached 92% on the test set using **RescaledSGD**, whereas **StandardSGD** only achieved 54%.
- **Loss Metrics**: Consistently lower validation and test loss values with **RescaledSGD**, indicating better model performance and generalization.

These results underscore the effectiveness of the gradient retention and dynamic learning rate scaling mechanisms in **RescaledSGD**.

## Repository

Access the project repository here: [GradRetentionNet](https://github.com/waefrebeorn/GradRetentionNet)

## Credits

- **WuBu WaefreBeorn**: Project lead, implementation, and bug fixes.
- **Kalomaze**: Conceptual ideas and guidance for prompting the creation of the RescaledSGD optimizer.
- **OpenAI's ChatGPT**: Assisted in code generation and optimization discussions.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements, bug fixes, or suggestions.

1. **Fork the Repository**
2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add Your Feature"
   ```

4. **Push to the Branch**

   ```bash
   git push origin feature/YourFeature
   ```

5. **Open a Pull Request**

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- **PyTorch**: For providing a robust deep learning framework.
- **Tkinter & Matplotlib**: For enabling effective data visualization.
- **MNIST Dataset**: For serving as a benchmark dataset for evaluation.

