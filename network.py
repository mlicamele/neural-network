import numpy as np

class Layer:

    def __init__(self):

        self.input: np.ndarray = None
        self.output: np.ndarray = None
    
    def forward(self, input):

        pass

    def backward(self, output_gradient, learning_rate):

        pass

class Dense(Layer):

    def __init__(self, input_size, output_size):

        self.weights: np.ndarray = np.random.rand(output_size, input_size) - 0.5
        self.biases: np.ndarray = np.random.rand(output_size, 1) - 0.5
    
    def forward(self, input):
        
        self.input = input
        self.output = self.weights.dot(self.input) + self.biases

        return self.output
    
    def backward(self, output_gradient: np.ndarray, learning_rate: float):

        error_gradient = self.weights.T.dot(output_gradient)
        self.weights -= learning_rate * output_gradient.dot(self.input.T)
        self.biases -= learning_rate * np.sum(output_gradient)

        return error_gradient

class Activation(Layer):

    def __init__(self, activation, activation_prime):

        self.activation: function = activation
        self.activation_prime: function = activation_prime
    
    def forward(self, input):

        self.input = input
        self.output = self.activation(self.input)

        return self.output

    def backward(self, output_gradient, learning_weight):

        error_gradient = output_gradient * self.activation_prime(self.input)

        return error_gradient

class Tanh(Activation):

    def __init__(self):

        tanh: function = TanH
        tanh_prime: function = deriv_TanH
        super().__init__(tanh, tanh_prime)

def TanH(x):
    
    return np.tanh(x)

def deriv_TanH(x):

    return 1 - np.tanh(x) ** 2

class Relu(Activation):

    def __init__(self):

        relu: function = ReLU
        relu_prime: function = deriv_ReLU
        super().__init__(relu, relu_prime)

def ReLU(x):

    return np.maximum(0, x)

def deriv_ReLU(x):

    return x > 0

class Softmax(Activation):

    def __init__(self):

        softmax: function = SoftMax
        softmax_prime: function = deriv_SoftMax
        super().__init__(softmax, softmax_prime)

def SoftMax(x):

    return (np.exp(x) / sum(np.exp(x)))

def deriv_SoftMax(x):

    return 1

def error(y_true, output):

    one_hot_y = one_hot(y_true)

    error =  (output - one_hot_y) / len(y_true)

    return error

def mse(y_true, y_pred):

    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):

    result = y_true - y_pred

    return result

class Network:

    def __init__(self, layers: list[Layer], learning_rate: float, epochs: int):
        
        self.layers = layers
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def train(self, X: np.ndarray, Y: np.ndarray):

        for epoch in range(self.epochs + 1):

            output = X

            for layer in self.layers:
                
                output = layer.forward(output)

            error_gradient = error(Y, output)
            # error_gradient = mse_prime(y, output)

            for layer in reversed(self.layers):

                error_gradient = layer.backward(error_gradient, self.learning_rate)

            if epoch % 50 == 0:

                preds = get_preds(output)

                print(f"Epoch {epoch}")
                print(f"Accuracy: {np.sum(preds == Y) / Y.size}")
                print(f"Preds:\t {preds[:20]}")
                print(f"Real:\t {Y[:20]}")
                print()
    
    def predict(self, X):

        output = X

        for layer in self.layers:
                        
            output = layer.forward(output)
        
        preds = get_preds(output)
        
        return preds

def one_hot(Y: np.ndarray):

    one_hot_Y = np.zeros((Y.size, Y.max() + 1))

    one_hot_Y[np.arange(Y.size), Y] = 1
    
    # Transpose matrix
    one_hot_Y = one_hot_Y.T

    return one_hot_Y

def get_preds(y: np.ndarray):

    preds = np.argmax(y, 0)

    return preds

def get_accuracy(y_true: np.ndarray, y_pred: np.ndarray):
    
    accuracy = np.sum(y_true == y_pred) / y_pred.size

    return accuracy