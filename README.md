# Neural Network - Digit Recognizer

A simple 2-layer neural network implementation for MNIST digit recognition built from scratch using NumPy.

## Project Overview

This project implements a digit recognition system using a neural network with the following architecture:
- **Input Layer**: 784 neurons (28×28 pixel images)
- **Hidden Layer**: 10 neurons with ReLU activation
- **Output Layer**: 10 neurons with Softmax activation (for 10 digit classes)

The implementation includes forward propagation, backpropagation, and gradient descent optimization.

## Features

- ✅ Custom neural network implementation (no ML frameworks used)
- ✅ Forward and backward propagation
- ✅ ReLU and Softmax activation functions
- ✅ Gradient descent optimization
- ✅ Training and evaluation on MNIST dataset
- ✅ Sample visualization and prediction
- ✅ Object-oriented design with reusable classes

## Project Structure

```
Neural_Network/
├── Neural_Network.ipynb          # Original Jupyter notebook
├── neural_network.py             # Main Python module with classes
├── digit-recognizer/             # MNIST dataset folder
│   ├── train.csv                # Training dataset
│   ├── test.csv                 # Test dataset
│   └── sample_submission.csv     # Sample submission format
├── requirements.txt              # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## Installation

### Prerequisites
- Python 3.7+
- pip

### Setup

1. Clone or download the project:
```bash
cd Neural_Network
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Using the Python Module

```python
from neural_network import DigitRecognizer

# Initialize the recognizer
recognizer = DigitRecognizer(data_path='digit-recognizer/train.csv')

# Load and split data
recognizer.load_and_split_data(test_size=1000)

# Train the model
recognizer.train(learning_rate=0.1, num_iterations=1000)

# Evaluate on test set
recognizer.evaluate()

# Visualize some predictions
recognizer.visualize_samples(num_samples=5)
```

### Running the Script

```bash
python neural_network.py
```

This will:
1. Load the MNIST training data
2. Split it into train/test sets
3. Train the neural network
4. Evaluate accuracy on the test set
5. Visualize sample predictions

## Class Documentation

### NeuralNetwork

Core neural network implementation with the following methods:

- `initialize_weights_and_bias()`: Initialize network parameters
- `forward_propagation(x)`: Perform forward pass through the network
- `backward_propagation(z1, a1, z2, a2, x, y, m)`: Calculate gradients
- `update_parameters(dw1, db1, dw2, db2, learning_rate)`: Update weights using gradients
- `train(x, y, learning_rate, num_iterations)`: Train the network
- `predict(x)`: Make predictions on new data
- `evaluate_sample(x, y, position)`: Visualize and evaluate a sample

### DigitRecognizer

High-level pipeline for digit recognition:

- `load_and_split_data(test_size)`: Load CSV data and split into train/test
- `train(learning_rate, num_iterations)`: Train the model
- `evaluate()`: Calculate test set accuracy
- `visualize_samples(num_samples)`: Display sample predictions

## Model Architecture

```
Input (784)
    ↓
Dense Layer (10) + Bias
    ↓
ReLU Activation
    ↓
Dense Layer (10) + Bias
    ↓
Softmax Activation
    ↓
Output (10 classes)
```

## Training Details

- **Loss Function**: Cross-entropy (via Softmax)
- **Optimization**: Gradient descent
- **Activation Functions**: ReLU (hidden), Softmax (output)
- **Default Learning Rate**: 0.1
- **Default Iterations**: 1000

## Dependencies

- `pandas` - Data loading and manipulation
- `numpy` - Numerical computations
- `matplotlib` - Visualization

See `requirements.txt` for specific versions.

## Dataset

This project uses the MNIST dataset in CSV format. The dataset files should be placed in the `digit-recognizer/` folder:

- `train.csv` - Training data (784 pixel values + 1 label per row)
- `test.csv` - Test data

## Example Output

```
Data loaded successfully!
Training set size: 41000
Test set size: 1000
Starting training...
Accuracy after iteration 0: 8.29%
Accuracy after iteration 100: 60.22%
Accuracy after iteration 200: 72.41%
...
Accuracy after iteration 1000: 87.56%

Test Set Accuracy: 86.30%
```

## Notes

- This is a simplified educational implementation
- For production use, consider using established frameworks like TensorFlow or PyTorch
- Model performance can be improved by:
  - Adding more layers or neurons
  - Using better activation functions (e.g., ELU, SELU)
  - Implementing dropout or batch normalization
  - Tuning hyperparameters
  - Training for more iterations

## Future Enhancements

- [ ] Add model persistence (save/load trained weights)
- [ ] Implement mini-batch gradient descent
- [ ] Add more activation functions
- [ ] Implement different optimization algorithms (Adam, RMSprop)
- [ ] Add cross-validation
- [ ] Create a web interface for predictions

## License

This project is open source and available for educational purposes.

## Author

Created as a neural network learning project.

Implementation of [Video by Samson Zhang](https://www.youtube.com/watch?v=w8yWXqWQYmU)