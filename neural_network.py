import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


class NeuralNetwork:
    """
    A simple 2-layer neural network for digit recognition.
    Architecture: Input(784) -> Hidden(10) -> Output(10)
    """
    
    def __init__(self):
        """Initialize the neural network weights and biases."""
        self.weight1 = None
        self.bias1 = None
        self.weight2 = None
        self.bias2 = None
    
    def initialize_weights_and_bias(self):
        """Initialize weights and biases with random values."""
        self.weight1 = np.random.rand(10, 784) - 0.5
        self.bias1 = np.random.rand(10, 1) - 0.5
        self.weight2 = np.random.rand(10, 10) - 0.5
        self.bias2 = np.random.rand(10, 1) - 0.5
    
    @staticmethod
    def relu(z):
        """ReLU activation function."""
        return np.maximum(0, z)
    
    @staticmethod
    def relu_derivative(z):
        """Derivative of ReLU activation function."""
        return z > 0
    
    @staticmethod
    def softmax(z):
        """Softmax activation function."""
        return np.exp(z) / sum(np.exp(z))
    
    @staticmethod
    def one_hot_encoding(y):
        """Convert labels to one-hot encoded format."""
        encoded_y = np.zeros((y.size, y.max() + 1))
        encoded_y[np.arange(y.size), y] = 1
        encoded_y = encoded_y.T
        return encoded_y
    
    def forward_propagation(self, x):
        """
        Perform forward propagation through the network.
        
        Args:
            x: Input data
            
        Returns:
            Tuple of (z1, a1, z2, a2)
        """
        z1 = np.dot(self.weight1, x) + self.bias1
        a1 = self.relu(z1)
        z2 = np.dot(self.weight2, a1) + self.bias2
        a2 = self.softmax(z2)
        
        return z1, a1, z2, a2
    
    def backward_propagation(self, z1, a1, z2, a2, x, y, m):
        """
        Perform backward propagation to calculate gradients.
        
        Args:
            z1, a1, z2, a2: Outputs from forward propagation
            x: Input data
            y: Labels
            m: Number of samples
            
        Returns:
            Gradients (dw1, db1, dw2, db2)
        """
        encoded_y = self.one_hot_encoding(y)
        
        dz2 = a2 - encoded_y
        dw2 = (1 / m) * np.dot(dz2, a1.T)
        db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
        
        dz1 = np.dot(self.weight2.T, dz2) * self.relu_derivative(z1)
        dw1 = (1 / m) * np.dot(dz1, x.T)
        db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)
        
        return dw1, db1, dw2, db2
    
    def update_parameters(self, dw1, db1, dw2, db2, learning_rate):
        """
        Update weights and biases using gradient descent.
        
        Args:
            dw1, db1, dw2, db2: Gradients
            learning_rate: Learning rate for gradient descent
        """
        self.weight1 = self.weight1 - learning_rate * dw1
        self.bias1 = self.bias1 - learning_rate * db1
        self.weight2 = self.weight2 - learning_rate * dw2
        self.bias2 = self.bias2 - learning_rate * db2
    
    @staticmethod
    def get_predictions(a2):
        """Get predicted class from output layer."""
        return np.argmax(a2, 0)
    
    @staticmethod
    def calculate_accuracy(predictions, y):
        """Calculate accuracy of predictions."""
        return np.sum(predictions == y) / y.size
    
    def train(self, x, y, learning_rate=0.1, num_iterations=1000):
        """
        Train the neural network using gradient descent.
        
        Args:
            x: Training input data
            y: Training labels
            learning_rate: Learning rate for gradient descent
            num_iterations: Number of training iterations
        """
        m = y.size
        self.initialize_weights_and_bias()
        
        for i in range(num_iterations):
            z1, a1, z2, a2 = self.forward_propagation(x)
            dw1, db1, dw2, db2 = self.backward_propagation(z1, a1, z2, a2, x, y, m)
            self.update_parameters(dw1, db1, dw2, db2, learning_rate)
            
            if i % 100 == 0:
                predictions = self.get_predictions(a2)
                accuracy = self.calculate_accuracy(predictions, y)
                print(f"Accuracy after iteration {i}: {accuracy * 100:.2f}%")
    
    def predict(self, x):
        """
        Make predictions on input data.
        
        Args:
            x: Input data
            
        Returns:
            Predicted classes
        """
        z1, a1, z2, a2 = self.forward_propagation(x)
        predictions = self.get_predictions(a2)
        return predictions
    
    def evaluate_sample(self, x, y, position):
        """
        Evaluate and visualize a single sample.
        
        Args:
            x: Input data
            y: Labels
            position: Index of the sample to evaluate
        """
        image = x[:, position, None]
        prediction = self.predict(image)
        label = y[position]
        
        print(f"Prediction: {prediction[0]}, Label: {label}")
        
        image = image.reshape((28, 28)) * 255.0
        plt.gray()
        plt.imshow(image)
        plt.show()


class DigitRecognizer:
    """Digit recognizer pipeline using the MNIST dataset."""
    
    def __init__(self, data_path='digit-recognizer/train.csv'):
        """
        Initialize the digit recognizer.
        
        Args:
            data_path: Path to the training CSV file
        """
        self.data_path = data_path
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.model = NeuralNetwork()
    
    def load_and_split_data(self, test_size=1000):
        """
        Load data from CSV and split into train and test sets.
        
        Args:
            test_size: Number of samples for test set
        """
        df = pd.read_csv(self.data_path)
        data = np.array(df)
        m, n = data.shape
        
        np.random.shuffle(data)
        
        # Test data
        data_test = data[0:test_size].T
        self.x_test = data_test[1:n] / 255.0
        self.y_test = data_test[0]
        
        # Training data
        data_train = data[test_size:m].T
        self.x_train = data_train[1:n] / 255.0
        self.y_train = data_train[0]
        
        print(f"Data loaded successfully!")
        print(f"Training set size: {self.x_train.shape[1]}")
        print(f"Test set size: {self.x_test.shape[1]}")
    
    def train(self, learning_rate=0.1, num_iterations=1000):
        """
        Train the neural network.
        
        Args:
            learning_rate: Learning rate for gradient descent
            num_iterations: Number of training iterations
        """
        if self.x_train is None:
            raise ValueError("Data not loaded. Call load_and_split_data() first.")
        
        print("Starting training...")
        self.model.train(self.x_train, self.y_train, learning_rate, num_iterations)
    
    def evaluate(self):
        """Evaluate the model on the test set."""
        if self.x_test is None:
            raise ValueError("Data not loaded. Call load_and_split_data() first.")
        
        test_predictions = self.model.predict(self.x_test)
        accuracy = self.model.calculate_accuracy(test_predictions, self.y_test)
        print(f"\nTest Set Accuracy: {accuracy * 100:.2f}%")
        return accuracy
    
    def visualize_samples(self, num_samples=5):
        """
        Visualize and evaluate sample predictions.
        
        Args:
            num_samples: Number of samples to visualize
        """
        for i in range(num_samples):
            self.model.evaluate_sample(self.x_train, self.y_train, i)


def main():
    """Main function to demonstrate the neural network."""
    # Initialize the digit recognizer
    recognizer = DigitRecognizer(data_path='digit-recognizer/train.csv')
    
    # Load and split data
    recognizer.load_and_split_data(test_size=1000)
    
    # Train the model
    recognizer.train(learning_rate=0.1, num_iterations=1000)
    
    # Evaluate on test set
    recognizer.evaluate()
    
    # Visualize some predictions
    print("\nVisualizing sample predictions:")
    recognizer.visualize_samples(num_samples=5)


if __name__ == "__main__":
    main()
