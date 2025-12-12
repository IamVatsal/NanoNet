import numpy as np
from enum import Enum

class __TrainingMethod(Enum):
    GD = "gd"
    SGD = "sgd"
    BGD = "bgd"

class __Optimizer(Enum):
    SGD = "sgd"
    ADAM = "adam"
    MOMENTUM = "momentum"
    RMSPROP = "rmsprop"
    NESTEROV = "nesterov"

class NeuralNet:
    def __init__(self, layers, init_method="he", training_method=None):
        self.__layers = layers
        self.__available_training_methods = [method.value for method in __TrainingMethod]
        self.__current_training_method = __TrainingMethod.GD
        self.__available_optimizers = [opt.value for opt in __Optimizer]
        self.__current_optimizer = __Optimizer.SGD
        if init_method == "he":
            self.__W = [np.random.randn(layers[i], layers[i + 1]) * np.sqrt(2 / layers[i]) for i in range(len(layers) - 1)]
            self.__B = [np.zeros((1, layers[i + 1])) for i in range(len(layers) - 1)]
            self.__losses = []
        else:
            raise ValueError("Unsupported initialization method")
        
        # Adam optimizer parameters (not implemented yet)
        self.__Vdw = [np.zeros_like(w) for w in self.__W]
        self.__Sdw = [np.zeros_like(w) for w in self.__W]
        self.__Vdb = [np.zeros_like(b) for b in self.__B]
        self.__Sdb = [np.zeros_like(b) for b in self.__B]
        self.__t = 0  # timestep for Adam

        # RMSprop and Momentum parameters
        self.__Rdw = [np.zeros_like(w) for w in self.__W]
        self.__Rdb = [np.zeros_like(b) for b in self.__B]
        self.__Mdw = [np.zeros_like(w) for w in self.__W]
        self.__Mdb = [np.zeros_like(b) for b in self.__B]

        if training_method is not None:
            self.set_training_method(training_method)
    
    def __ReLU(self, x):
        return np.maximum(0, x)
    
    def __ReLU_derivative(self, x):
        return (x > 0).astype(float)

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
            dZ = dA * self.__ReLU_derivative(Zs[i])  # __ReLU derivative
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
        if self.__current_optimizer == __Optimizer.SGD:
            return self.__sgd_update(W, B, dW, dB, alpha)
        elif self.__current_optimizer == __Optimizer.ADAM:
            return self.__adam_update(W, B, dW, dB, alpha)
        elif self.__current_optimizer == __Optimizer.MOMENTUM:
            return self.__momentum_update(W, B, dW, dB, alpha)
        elif self.__current_optimizer == __Optimizer.RMSPROP:
            return self.__RMSprop_update(W, B, dW, dB, alpha)
        elif self.__current_optimizer == __Optimizer.NESTEROV:
            return self.__nesterov_update(W, B, dW, dB, alpha)
        else:
            raise ValueError("Unsupported optimizer")
    
    def __sgd_update(self, W, B, dW, dB, lr):
        for i in range(len(W)):
            W[i] -= lr * dW[i]
            B[i] -= lr * dB[i]
        return W, B

    def __adam_update(self, W, B, dW, dB, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.__t += 1
        t = self.__t

        for i in range(len(W)):
            # Update biased first moment estimate
            self.__Vdw[i] = beta1 * self.__Vdw[i] + (1 - beta1) * dW[i]
            self.__Vdb[i] = beta1 * self.__Vdb[i] + (1 - beta1) * dB[i]

            # Update biased second raw moment estimate
            self.__Sdw[i] = beta2 * self.__Sdw[i] + (1 - beta2) * (dW[i] ** 2)
            self.__Sdb[i] = beta2 * self.__Sdb[i] + (1 - beta2) * (dB[i] ** 2)

            # Compute bias-corrected first moment estimate
            Vdw_corrected = self.__Vdw[i] / (1 - beta1 ** t)
            Vdb_corrected = self.__Vdb[i] / (1 - beta1 ** t)

            # Compute bias-corrected second raw moment estimate
            Sdw_corrected = self.__Sdw[i] / (1 - beta2 ** t)
            Sdb_corrected = self.__Sdb[i] / (1 - beta2 ** t)

            # Update parameters
            W[i] -= lr * Vdw_corrected / (np.sqrt(Sdw_corrected) + eps)
            B[i] -= lr * Vdb_corrected / (np.sqrt(Sdb_corrected) + eps)

        return W, B
    
    def __momentum_update(self, W, B, dW, dB, lr, beta=0.9):
        for i in range(len(W)):
            # Update velocity
            self.__Mdw[i] = beta * self.__Mdw[i] + dW[i]
            self.__Mdb[i] = beta * self.__Mdb[i] + dB[i]

            # Update parameters
            W[i] -= lr * self.__Mdw[i]
            B[i] -= lr * self.__Mdb[i]

        return W, B
    
    def __nesterov_update(self, W, B, dW, dB, lr, beta=0.9):
        for i in range(len(W)):
            # Lookahead step
            V_prev_dw = self.__Mdw[i].copy()
            V_prev_db = self.__Mdb[i].copy()

            # Update velocity
            self.__Mdw[i] = beta * self.__Mdw[i] + dW[i]
            self.__Mdb[i] = beta * self.__Mdb[i] + dB[i]

            # Update parameters
            W[i] -= lr * (dW[i] + beta * V_prev_dw)
            B[i] -= lr * (dB[i] + beta * V_prev_db)

        return W, B
    

    def __RMSprop_update(self, W, B, dW, dB, lr, beta=0.9, eps=1e-8):
        for i in range(len(W)):
            # Update squared gradients
            self.__Rdw[i] = beta * self.__Rdw[i] + (1 - beta) * (dW[i] ** 2)
            self.__Rdb[i] = beta * self.__Rdb[i] + (1 - beta) * (dB[i] ** 2)
            # Update parameters
            W[i] -= lr * dW[i] / (np.sqrt(self.__Rdw[i]) + eps)
            B[i] -= lr * dB[i] / (np.sqrt(self.__Rdb[i]) + eps)

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
        if log_interval is None:
            log_interval = max(1, len(X) // 20)

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
            print("Using full-batch training.")
            batch_size = len(X)


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

    def avalible_optimizers(self):
        print(f"Avalible Optimizers: {self.__available_optimizers}")

    def reset(self):
        old_method = self.__current_training_method
        old_optimizer = self.__current_optimizer
        self.__init__(layers=self.__layers)
        self.__current_training_method = old_method
        self.__current_optimizer = old_optimizer
    
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
        
    def set_optimizer(self, optimizer):
        optimizer = optimizer.strip().lower()
        if optimizer == "sgd":
            self.__current_optimizer = __Optimizer.SGD
        elif optimizer == "adam":
            self.__current_optimizer = __Optimizer.ADAM
        elif optimizer == "momentum":
            self.__current_optimizer = __Optimizer.MOMENTUM
        elif optimizer == "rmsprop":
            self.__current_optimizer = __Optimizer.RMSPROP
        elif optimizer == "nesterov":
            self.__current_optimizer = __Optimizer.NESTEROV
        else:
            raise ValueError("Unsupported optimizer")

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