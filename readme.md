# GradRetentionNet: Exploring Persistent Gradient Descent for Global Optimization

## Overview
**GradRetentionNet** is a research project focused on exploring the concept of *Persistent Gradient Descent (PGD)* and comparing its performance with standard optimizers like `AdamW` and custom variants such as `AdEMA`. The primary goal is to implement, analyze, and understand the effectiveness of *retaining gradients across iterations* and its impact on global convergence in neural networks.

The project implements and tests Kalomaze's idea of *never fully clearing gradients*, allowing for a form of momentum that adapts more globally rather than being biased by recent gradient information.

## Project Structure
```
GradRetentionNet/
├── main.py               # Main training and testing script with comparison and graphing
├── setup.bat             # Environment setup script
├── run.bat               # Run script to activate environment and execute the project
├── requirements.txt      # Dependencies file
├── .gitignore            # Git ignore file to exclude unnecessary files
├── README.md             # Project documentation and guidelines
└── data/                 # Folder for datasets (excluded in .gitignore)
```

## Optimizers Implemented
### 1. PersistentSGD
The `PersistentSGD` optimizer maintains a state of *persistent gradients* across iterations. Instead of zeroing gradients after each step, it accumulates them using a decaying multiplier. This technique allows the optimizer to adapt more globally, potentially leading to better convergence in complex loss landscapes.

### 2. AdEMA
`AdEMA` is a custom variant of Adam that uses Exponential Moving Average (EMA) to track gradients with an adjustable decay. This helps reduce oscillations and improve generalization.

### 3. AdamW
`AdamW` serves as a baseline adaptive optimizer with weight decay and momentum, focusing on local gradient information.

## Installation
1. **Clone the Repository**
   ```bash
   git clone https://github.com/waefrebeorn/GradRetentionNet.git
   cd GradRetentionNet
   ```

2. **Set Up the Environment**
   Run the `setup.bat` script to create a virtual environment, install dependencies, and download the MNIST dataset.
   ```bash
   setup.bat
   ```

3. **Running the Project**
   Use the `run.bat` script to run the `main.py` file.
   ```bash
   run.bat
   ```

## Usage
1. **Modify Parameters**:
   You can customize learning rates, decay factors, and model configurations directly in the `main.py` script.
   
2. **Adding New Optimizers**:
   To add new optimizers, implement a new optimizer class in `main.py` following the pattern of `PersistentSGD` and `AdEMA`.

3. **View Results**:
   The script generates training loss curves comparing different optimizers and outputs test accuracy for each configuration.

## Results
The project includes comparison graphs between `PersistentSGD`, `AdamW`, and `AdEMA`, showing the impact of retaining gradient history on loss convergence. Below is a sample graph generated by the project (if applicable):

![Training Loss Comparison](images/loss_comparison.png)

## Research Background
The concept of **Persistent Gradient Descent** (PGD) was inspired by discussions in machine learning forums, where researchers hypothesized that retaining gradient memory could lead to better global convergence. This project implements the idea in a concrete manner and compares it against traditional optimizers to test its effectiveness in practice.

### Key Concepts:
1. **Persistent Gradient Accumulation**: Keeping a fraction of the past gradients to avoid overfitting to recent information.
2. **Global Adaptation**: Using a decaying memory of gradients to adjust parameters more globally.
3. **Comparison to AdamW**: Highlighting differences between localized and global adaptations in complex loss surfaces.

## Future Work
1. **Dynamic Decay Scheduling**: Implementing learnable decay factors to adjust memory retention dynamically during training.
2. **Extended Optimizer Configurations**: Testing a wider range of configurations on deeper architectures.
3. **Real-World Applications**: Applying PersistentSGD and its variants to real-world datasets like CIFAR-10 and ImageNet.

## Contributing
If you'd like to contribute, feel free to open a pull request. Contributions are welcome for new optimizers, visualization tools, or bug fixes.

## License
This project is licensed under the MIT License.

## Contact
For further discussions, feel free to reach out through GitHub at [waefrebeorn](https://github.com/waefrebeorn).


