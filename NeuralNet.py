import numpy as np
from enum import Enum

class __TrainingMethod(Enum):
    GD = "gd"
    SGD = "sgd"
    BGD = "bgd"

class NeuralNet:
    def __init__(self, layers, init_method="he", training_method=None):
        self.__layers = layers
        self.__available_training_methods = [method.value for method in __TrainingMethod]
        self.__current_training_method = __TrainingMethod.GD
        if init_method == "he":
            self.__W = [np.random.randn(layers[i], layers[i + 1]) * np.sqrt(2 / layers[i]) for i in range(len(layers) - 1)]
            self.__B = [np.zeros((1, layers[i + 1])) for i in range(len(layers) - 1)]
            self.__losses = []
        else:
            raise ValueError("Unsupported initialization method")
        
        if training_method is not None:
            self.set_training_method(training_method)
    
    def __ReLU(self, x):
        return np.maximum(0, x)

    def __softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # stability trick
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def __hidden_layer(self, X, W, b):
        Z = np.dot(X, W) + b
        A = self.__ReLU(Z)
        return A, Z

    def __output_layer(self, X, W, b):
        return self.__softmax(np.dot(X, W) + b)

    def __forward_pass(self, X, W, B):
        Zs = []
        As = [X]
        for i in range(len(W) - 1):
            A, Z = self.__hidden_layer(As[-1], W[i], B[i])
            As.append(A)
            Zs.append(Z)
        out = self.__output_layer(As[-1], W[-1], B[-1])
        return Zs, As, out

    def __one_hot(self, Y, num_classes):
        one_hot_Y = np.eye(num_classes)[Y.astype(int)]
        return one_hot_Y

    def __back_prop(self, Zs, As, output, W, X, Y):
        # Number of layers (including output layer)
        L = len(W)

        # One-hot encode labels
        num_classes = output.shape[1]
        one_hot_Y = self.__one_hot(Y, num_classes)

        # Gradients containers
        dW = [None] * L
        dB = [None] * L

        # Gradient for output layer (__softmax + cross-entropy)
        dZ = (output - one_hot_Y) / X.shape[0]
        dW[L - 1] = np.dot(As[-1].T, dZ)
        dB[L - 1] = np.sum(dZ, axis=0, keepdims=True)

        # Backpropagate through hidden layers
        for i in reversed(range(L - 1)):
            dA = np.dot(dZ, W[i + 1].T)
            dZ = dA * (Zs[i] > 0)  # __ReLU derivative
            dW[i] = np.dot(As[i].T, dZ)
            dB[i] = np.sum(dZ, axis=0, keepdims=True)

        return dW, dB

    def __compute_loss(self, output, Y):
        m = Y.size
        # output shape: (m, 10)
        # Y: (m,) integer labels
        prob = np.clip(output, 1e-8, 1 - 1e-8)
        log_likelihood = -np.log(prob[range(m), Y])
        loss = np.sum(log_likelihood) / m
        return loss


    def __update_params(self, W, B, dW, dB, alpha):
        for i in range(len(W)):
            W[i] -= alpha * dW[i]
            B[i] -= alpha * dB[i]
        return W, B
    
    def __get_predictions(self, output):
        return np.argmax(output, axis=1)

    def __get_accuracy(self, predictions, Y):
        return np.sum(predictions == Y) / Y.size

    def __gradient_descent(self, X, Y, lr, epochs, log_interval=10, suppress_output=False):
        W = [w.copy() for w in self.__W]
        B = [b.copy() for b in self.__B]

        for i in range(epochs):
            Zs, As, output = self.__forward_pass(X, W, B)
            dW, dB = self.__back_prop(Zs, As, output, W, X, Y)
            W, B = self.__update_params(W, B,dW, dB, lr)
            if i % log_interval == 0:
                loss = self.__compute_loss(output, Y)
                predictions = self.__get_predictions(output)
                acc = self.__get_accuracy(predictions, Y)
                if not suppress_output:
                    print(f"Iteration {i}, Accuracy: {acc}, Loss: {loss}")
                self.__losses.append(loss)

        self.__W = W
        self.__B = B

    def __stochastic_gradient_descent(self, X, Y, lr, epochs=1, log_interval=None, suppress_output=False):
        W = [w.copy() for w in self.__W]
        B = [b.copy() for b in self.__B]
        log_interval = log_interval or max(1, (len(X) // 50))
        for epoch in range(epochs):
            indices = np.random.permutation(len(X))
            for i, idx in enumerate(indices):
                X_sample = X[idx].reshape(1,-1)
                Y_sample = np.array([Y[idx]])
                Zs, As, output = self.__forward_pass(X_sample, W, B)
                dW, dB = self.__back_prop(Zs, As, output, W, X_sample, Y_sample)
                W, B = self.__update_params(W, B, dW, dB, lr)
                if i % log_interval == 0:
                    loss = self.__compute_loss(output, Y_sample)
                    if not suppress_output:
                        print(f"Epoch {epoch}, Iteration {i}, Loss: {loss}")
                    self.__losses.append(loss)
        self.__W = W
        self.__B = B

    def __batch_gradient_descent(self, X, Y, lr, epochs=1, log_interval=100 , batch_size=32, suppress_output=False):
        if batch_size >= len(X):
            print("Warning: full-batch selected; consider using GD instead.")

        W = [w.copy() for w in self.__W]
        B = [b.copy() for b in self.__B]

        for epoch in range(epochs):
            indices = np.random.permutation(len(X))
            X = X[indices]
            Y = Y[indices]
            for step in range(0, len(X), batch_size):
                X_batch = X[step : step + batch_size]
                Y_batch = Y[step : step + batch_size]
                Zs, As, output = self.__forward_pass(X_batch, W, B)
                dW, dB = self.__back_prop(Zs, As, output, W, X_batch, Y_batch)
                W, B = self.__update_params(W, B, dW, dB, lr)

                if (step // batch_size) % log_interval == 0:
                    loss = self.__compute_loss(output, Y_batch)
                    if not suppress_output:
                        print(f"Epoch {epoch}, batch {step // batch_size}, Loss: {loss}")
                    self.__losses.append(loss)
    
        self.__W = W
        self.__B = B

    def predict(self, X):
        _, _, output = self.__forward_pass(X, self.__W, self.__B)
        predictions = self.__get_predictions(output)
        return predictions
    
    def predict_proba(self, X):
        _, _, output = self.__forward_pass(X, self.__W, self.__B)
        return output

    def evaluate(self, X, Y):
        predictions = self.predict(X)
        accuracy = self.__get_accuracy(predictions, Y)
        print(f"Test Accuracy: {accuracy}")

    def test_predictions(self, index, X, Y):
        prediction = self.predict(X[index].reshape(1, -1))
        print(f"Model Prediction: {prediction[0]}")
        print(f"True Label: {Y[index]}")

    def get_losses(self):
        return self.__losses

    def avalible_training_methods(self):
        print(f"Avalible Methods: {self.__available_training_methods}")

    def reset(self):
        old_method = self.__current_training_method
        self.__init__(layers=self.__layers)
        self.__current_training_method = old_method
    
    def reset_losses(self):
        self.__losses = []

    def set_training_method(self, method):
        method = method.strip().lower()
        if method in ["gradient_descent", "gd"]:
            self.__current_training_method = __TrainingMethod.GD
        elif method in ["stochastic_gradient_descent", "sgd"]:
            self.__current_training_method = __TrainingMethod.SGD
        elif method in ["batch_gradient_descent", "bgd"]:
            self.__current_training_method = __TrainingMethod.BGD
        else:
            raise ValueError("Unsupported training method")

    def train(self, X, Y, lr, epochs, training_method=None, log_interval=None, batch_size=None, suppress_output=False, reset_all=False, reset_losses=False):
        if training_method is None:
            if self.__current_training_method is None:
                raise ValueError("Training method not set. Please set it using set_training_method or provide it in train method.")
        else:
            self.set_training_method(training_method)

        if reset_all:
            self.reset()

        if reset_losses:
            self.reset_losses()

        if self.__current_training_method == __TrainingMethod.GD:
            if log_interval is None:
                log_interval = 10
            self.__gradient_descent(X=X, Y=Y, lr=lr, epochs=epochs, log_interval=log_interval, suppress_output=suppress_output)
        elif self.__current_training_method == __TrainingMethod.SGD:
            if log_interval is None:
                log_interval = 100
            self.__stochastic_gradient_descent(X=X, Y=Y, lr=lr, epochs=epochs, log_interval=log_interval, suppress_output=suppress_output)
        elif self.__current_training_method == __TrainingMethod.BGD:
            if batch_size is None:
                batch_size = 32
            if log_interval is None:
                log_interval = 100
            self.__batch_gradient_descent(X=X, Y=Y, lr=lr, epochs=epochs, log_interval=log_interval, batch_size=batch_size, suppress_output=suppress_output)
        else:
            print(f"Avalible Methods: {self.__available_training_methods}")

        return list(self.__losses)