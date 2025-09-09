# neural-network

# Neural Network Implementation from Scratch

A comprehensive neural network implementation built from scratch in Python for MNIST digit classification, exploring the mathematical foundations of deep learning through linear algebra and calculus.

## Academic Project Overview

This project was developed as part of a Linear Algebra/Multivariable Calculus course to explore the practical applications of mathematical concepts in artificial intelligence. The implementation focuses on understanding the underlying mathematics of neural networks, including matrix operations, partial derivatives, and gradient descent optimization.

## Features

- **From-Scratch Implementation**: Built entirely using NumPy for matrix operations
- **Modular Architecture**: Object-oriented design with separate layer classes
- **Mathematical Foundation**: Detailed implementation of forward and backward propagation
- **MNIST Classification**: Digit recognition on the classic 28×28 pixel dataset
- **Custom Layer Types**: Dense, ReLU activation, and Softmax output layers
- **Gradient Descent**: Implementation of backpropagation with configurable learning rates

## Results

- **Training Accuracy**: 92.6%
- **Test Accuracy**: 91.4%
- **Dataset**: MNIST (28×28 grayscale digit images)
- **Architecture**: 784 → 10 → 10 → 10 (with ReLU and Softmax activations)

## Architecture

### Network Structure
```
Input Layer (784) → Dense Layer (10) → ReLU → Dense Layer (10) → Softmax → Output (10)
```

- **Input**: 784 features (28×28 pixel values)
- **Hidden Layer**: 10 neurons with ReLU activation
- **Output**: 10 classes (digits 0-9) with Softmax probabilities

### Mathematical Foundation

The implementation includes detailed mathematical derivations for:
- **Forward Propagation**: Linear combinations with weights and biases
- **Backward Propagation**: Gradient computation using chain rule
- **Weight Updates**: Gradient descent optimization
- **Error Gradients**: Partial derivatives for each layer type

## Installation

### Prerequisites
- Python 3.7+
- NumPy for matrix operations

### Setup
```bash
# Clone the repository
git clone https://github.com/mlicamele/neural-network.git
cd neural-network

# Install dependencies
pip install numpy
```

## 🚀 Quick Start

```python
from neural_network import Network, Dense, Relu, Softmax
import numpy as np

# Create the network architecture
net = Network([
    Dense(784, 10),    # Input to hidden layer
    Relu(),            # ReLU activation
    Dense(10, 10),     # Hidden layer
    Softmax()          # Output layer with softmax
], 
learning_rate=0.5,
epochs=1000)

# Train the network
net.train(X_train, y_train)

# Make predictions
predictions = net.predict(X_test)
```

## Layer Classes

### Dense Layer
Implements fully connected layers with learnable weights and biases:
```python
class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(output_size, input_size) - 0.5
        self.biases = np.random.rand(output_size, 1) - 0.5
```

### Activation Layers
- **ReLU**: Rectified Linear Unit activation function
- **Softmax**: Probability distribution for multi-class classification

### Mathematical Operations
- **Forward Pass**: `output = weights @ input + biases`
- **Backward Pass**: Gradient computation using chain rule
- **Weight Updates**: `weights -= learning_rate * gradient`

## Mathematical Implementation

### Forward Propagation
For each layer, the output is computed as:
```
Y = W·X + B
```
Where:
- `W` is the weight matrix
- `X` is the input vector
- `B` is the bias vector

### Backward Propagation
Gradients are computed using partial derivatives:
- **Input gradient**: `∂E/∂X = W^T · ∂E/∂Y`
- **Weight gradient**: `∂E/∂W = ∂E/∂Y · X^T`
- **Bias gradient**: `∂E/∂B = ∂E/∂Y`

## 📁 Project Structure

```
neural-network/
│
├── network.py              # Main Network class
├── test_network.py         # Training network and predicting on MNIST dataset
├── gradient_descent.ipynb  # Non-object oriented approach to get familiar with the basics of the computation
├── data
|   ├── test.csv            # Test dataset
|   └── train.csv           # Train dataset
└── README.md
```

## Key Learning Outcomes

This project demonstrates:
- **Matrix Operations**: Extensive use of NumPy for linear algebra
- **Calculus Applications**: Partial derivatives in backpropagation
- **Optimization**: Gradient descent implementation
- **Object-Oriented Design**: Modular layer architecture
- **Machine Learning Fundamentals**: Training, validation, and testing

## Training Process

1. **Data Preprocessing**: MNIST images flattened to 784-dimensional vectors
2. **One-Hot Encoding**: Target labels converted to probability distributions
3. **Forward Propagation**: Data flows through network layers
4. **Loss Calculation**: Error computed between predictions and targets
5. **Backward Propagation**: Gradients computed and weights updated
6. **Iteration**: Process repeated for specified epochs

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 0.5 | Step size for gradient descent |
| Epochs | 1000 | Number of training iterations |
| Hidden Units | 10 | Neurons in hidden layer |
| Batch Processing | Full dataset | All samples processed simultaneously |

## Educational Features

- **Mathematical Derivations**: Complete derivations included in documentation
- **Step-by-Step Implementation**: Each mathematical operation explained
- **Visualization Ready**: Easy to add plotting for loss curves and accuracy
- **Extensible Design**: Simple to add new layer types or activation functions

## Academic References

This implementation is based on fundamental neural network principles detailed in:
- Higham, C. F., & Higham, D. J. (2019). Deep learning: An introduction for applied mathematicians
- Fan, J., Ma, C., & Zhong, Y. (2021). A selective overview of deep learning
- Zhang, C.-H. (2007). Continuous generalized gradient descent

---

*This project demonstrates the beautiful intersection of mathematics and artificial intelligence, showing how linear algebra and calculus power modern machine learning.*
