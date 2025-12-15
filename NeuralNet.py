import numpy as np
from enum import Enum
from LRScheduler import LRScheduler

class _TrainingMethod(Enum):
    GD = "gd"
    SGD = "sgd"
    BGD = "bgd"

class _Optimizer(Enum):
    SGD = "sgd"
    ADAM = "adam"
    MOMENTUM = "momentum"
    RMSPROP = "rmsprop"
    NESTEROV = "nesterov"

class _Regularization(Enum):
    NONE = "none"
    L1 = "l1"
    L2 = "l2"

class _GradientClipper(Enum):
    NONE = "none"
    NORM = "norm"
    VALUE = "value"

class NeuralNet:
    def __init__(self, layers, init_method="he", training_method=None, optimizer="sgd", regularization=None, reg_lambda=0.0, lr_scheduler=None, base_lr=0.001, gradient_clipper=None, clip_value=1.0):
        self.__layers = layers
        self.__available_training_methods = [method.value for method in _TrainingMethod]
        self.__current_training_method = _TrainingMethod.GD
        self.__available_optimizers = [opt.value for opt in _Optimizer]
        self.__current_optimizer = _Optimizer.SGD
        self.__available_regularizations = [reg.value for reg in _Regularization]
        self.__current_regularization = _Regularization.NONE
        self.__available_gradient_clippers = [clipper.value for clipper in _GradientClipper]
        self.__current_gradient_clipper = _GradientClipper.NONE
        
        # Regularization lambda
        self.__regularization_lambda = reg_lambda

        # Initialize learning rate scheduler
        self.lr_scheduler = LRScheduler(base_lr=base_lr)

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
        if regularization is not None:
            self.__current_regularization = _Regularization(regularization)
        if optimizer is not None:
            self.set_optimizer(optimizer)
        if lr_scheduler is not None:
            self.lr_scheduler.set_decay_type(lr_scheduler)
        if gradient_clipper is not None:
            self.set_gradient_clipper(gradient_clipper, clip_value)
    
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
    
    def __compute_weights(self, A, dz, W):
        m = A.shape[0]
        basegrad = np.dot(A, dz) / m

        if self.__current_regularization == _Regularization.NONE:
            return basegrad
        elif self.__current_regularization == _Regularization.L2:
            return basegrad + (self.__regularization_lambda / m) * W
        elif self.__current_regularization == _Regularization.L1:
            return basegrad + (self.__regularization_lambda / m) * np.sign(W + 1e-8)
        
    def __clip_gradients(self, dW, dB):
        if self.__current_gradient_clipper == _GradientClipper.NONE:
            return dW, dB
        
        elif self.__current_gradient_clipper == _GradientClipper.VALUE:
            for i in range(len(dW)):
                dW[i] = np.clip(dW[i], -self.__gradient_clip_value, self.__gradient_clip_value)
                dB[i] = np.clip(dB[i], -self.__gradient_clip_value, self.__gradient_clip_value)
            return dW, dB
        
        elif self.__current_gradient_clipper == _GradientClipper.NORM:
            total_norm = 0.0
            for i in range(len(dW)):
                total_norm += np.sum(dW[i] ** 2) 
                total_norm += np.sum(dB[i] ** 2)

            total_norm = np.sqrt(total_norm)

            if total_norm > self.__gradient_clip_value:
                scale = self.__gradient_clip_value / (total_norm + 1e-6)
                for i in range(len(dW)):
                    dW[i] *= scale
                    dB[i] *= scale
            return dW, dB


    def __back_prop(self, Zs, As, output, W, X, Y):
        # Number of layers (including output layer)
        L = len(W)
        m = X.shape[0]

        # One-hot encode labels
        num_classes = output.shape[1]
        one_hot_Y = self.__one_hot(Y, num_classes)

        # Gradients containers
        dW = [None] * L
        dB = [None] * L

        # Gradient for output layer (__softmax + cross-entropy)
        dZ = (output - one_hot_Y) / m
        # Compute gradients for output layer
        dW[L - 1] = self.__compute_weights(As[-1].T, dZ, W[L - 1])
        dB[L - 1] = np.sum(dZ, axis=0, keepdims=True)

        # Backpropagate through hidden layers
        for i in reversed(range(L - 1)):
            dA = np.dot(dZ, W[i + 1].T)
            dZ = dA * self.__ReLU_derivative(Zs[i])  # __ReLU derivative
            dW[i] = self.__compute_weights(As[i].T, dZ, W[i])
            dB[i] = np.sum(dZ, axis=0, keepdims=True)

        return dW, dB

    def __compute_loss(self, output, Y):
        m = Y.size
        # output shape: (m, 10)
        # Y: (m,) integer labels
        prob = np.clip(output, 1e-8, 1 - 1e-8)
        log_likelihood = -np.log(prob[range(m), Y])
        loss = np.sum(log_likelihood) / m
        if self.__current_regularization == _Regularization.L2:
            l2_sum = sum([np.sum(np.square(w)) for w in self.__W])
            loss += (self.__regularization_lambda / (2 * m)) * l2_sum
        elif self.__current_regularization == _Regularization.L1:
            l1_sum = sum([np.sum(np.abs(w)) for w in self.__W])
            loss += (self.__regularization_lambda / m) * l1_sum
        return loss

    def __update_params(self, W, B, dW, dB, alpha, X=None, Y=None):
        if self.__current_optimizer == _Optimizer.SGD:
            return self.__sgd_update(W, B, dW, dB, alpha)
        elif self.__current_optimizer == _Optimizer.ADAM:
            return self.__adam_update(W, B, dW, dB, alpha)
        elif self.__current_optimizer == _Optimizer.MOMENTUM:
            return self.__momentum_update(W, B, dW, dB, alpha)
        elif self.__current_optimizer == _Optimizer.RMSPROP:
            return self.__RMSprop_update(W, B, dW, dB, alpha)
        elif self.__current_optimizer == _Optimizer.NESTEROV:
            return self.__nesterov_update(W, B, dW, dB, alpha, X, Y)
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
    
    def __nesterov_update(self, W, B, dW, dB, lr, X, Y, beta=0.9):
        W_lookahead = [W[i] - beta * self.__Mdw[i] for i in range(len(W))]
        B_lookahead = [B[i] - beta * self.__Mdb[i] for i in range(len(B))]

        Zs, As, output = self.__forward_pass(X, W_lookahead, B_lookahead)
        dW_lookahead, dB_lookahead = self.__back_prop(Zs, As, output, W_lookahead, X, Y)

        for i in range(len(W)):
            # Update velocity
            self.__Mdw[i] = beta * self.__Mdw[i] + dW_lookahead[i]
            self.__Mdb[i] = beta * self.__Mdb[i] + dB_lookahead[i]

            # Update parameters
            W[i] -= lr * self.__Mdw[i]
            B[i] -= lr * self.__Mdb[i]

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

    def __gradient_descent(self, X, Y, epochs=1, log_interval=10, suppress_output=False):
        W = [w.copy() for w in self.__W]
        B = [b.copy() for b in self.__B]

        for i in range(epochs):
            self.lr_scheduler.epoch_step()
            Zs, As, output = self.__forward_pass(X, W, B)
            dW, dB = self.__back_prop(Zs, As, output, W, X, Y)
            dW, dB = self.__clip_gradients(dW, dB)
            lr = self.lr_scheduler.lr
            W, B = self.__update_params(W, B,dW, dB, lr, X, Y)
            if i % log_interval == 0:
                loss = self.__compute_loss(output, Y)
                predictions = self.__get_predictions(output)
                acc = self.__get_accuracy(predictions, Y)
                if not suppress_output:
                    print(f"Iteration {i}, Accuracy: {acc}, Loss: {loss}")
                self.__losses.append(loss)

        self.__W = W
        self.__B = B

    def __stochastic_gradient_descent(self, X, Y, epochs=1, log_interval=None, suppress_output=False):
        W = [w.copy() for w in self.__W]
        B = [b.copy() for b in self.__B]
        if log_interval is None:
            log_interval = max(1, len(X) // 20)

        for epoch in range(epochs):
            self.lr_scheduler.epoch_step()
            indices = np.random.permutation(len(X))
            for i, idx in enumerate(indices):
                self.lr_scheduler.step()
                X_sample = X[idx].reshape(1,-1)
                Y_sample = np.array([Y[idx]])
                Zs, As, output = self.__forward_pass(X_sample, W, B)
                dW, dB = self.__back_prop(Zs, As, output, W, X_sample, Y_sample)
                dW, dB = self.__clip_gradients(dW, dB)
                lr = self.lr_scheduler.lr
                W, B = self.__update_params(W, B, dW, dB, lr, X_sample, Y_sample)
                if i % log_interval == 0:
                    loss = self.__compute_loss(output, Y_sample)
                    if not suppress_output:
                        print(f"Epoch {epoch}, Iteration {i}, Loss: {loss}")
                    self.__losses.append(loss)
        self.__W = W
        self.__B = B

    def __batch_gradient_descent(self, X, Y, epochs=1, log_interval=100 , batch_size=32, suppress_output=False):
        if batch_size >= len(X):
            print("Using full-batch training.")
            batch_size = len(X)


        W = [w.copy() for w in self.__W]
        B = [b.copy() for b in self.__B]

        for epoch in range(epochs):
            self.lr_scheduler.epoch_step()
            indices = np.random.permutation(len(X))
            X = X[indices]
            Y = Y[indices]
            for step in range(0, len(X), batch_size):
                self.lr_scheduler.step()
                X_batch = X[step : step + batch_size]
                Y_batch = Y[step : step + batch_size]
                Zs, As, output = self.__forward_pass(X_batch, W, B)
                dW, dB = self.__back_prop(Zs, As, output, W, X_batch, Y_batch)
                dW, dB = self.__clip_gradients(dW, dB)
                lr = self.lr_scheduler.lr
                W, B = self.__update_params(W, B, dW, dB, lr, X_batch, Y_batch)

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

    def available_training_methods(self):
        print(f"Available Methods: {self.__available_training_methods}")

    def available_optimizers(self):
        print(f"Available Optimizers: {self.__available_optimizers}")

    def available_decay_types(self):
        self.lr_scheduler.available_decay_types()

    def available_regularizations(self):
        print(f"Available Regularizations: {self.__available_regularizations}")

    def available_gradient_clippers(self):
        print(f"Available Gradient Clippers: {self.__available_gradient_clippers}")

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
            self.__current_training_method = _TrainingMethod.GD
        elif method in ["stochastic_gradient_descent", "sgd"]:
            self.__current_training_method = _TrainingMethod.SGD
        elif method in ["batch_gradient_descent", "bgd"]:
            self.__current_training_method = _TrainingMethod.BGD
        else:
            raise ValueError("Unsupported training method")
        
    def set_optimizer(self, optimizer):
        optimizer = optimizer.strip().lower()
        if optimizer == "sgd":
            self.__current_optimizer = _Optimizer.SGD
        elif optimizer == "adam":
            self.__current_optimizer = _Optimizer.ADAM
        elif optimizer == "momentum":
            self.__current_optimizer = _Optimizer.MOMENTUM
        elif optimizer == "rmsprop":
            self.__current_optimizer = _Optimizer.RMSPROP
        elif optimizer == "nesterov":
            self.__current_optimizer = _Optimizer.NESTEROV
        else:
            raise ValueError("Unsupported optimizer")
        
    def set_decay_type(self, decay_type, **kwargs):
        self.lr_scheduler.set_decay_type(decay_type, **kwargs)

    def set_regularization(self, regularization, reg_lambda=0.0):
        regularization = regularization.strip().lower()
        if regularization == "none":
            self.__current_regularization = _Regularization.NONE
        elif regularization == "l1":
            self.__current_regularization = _Regularization.L1
        elif regularization == "l2":
            self.__current_regularization = _Regularization.L2
        else:
            raise ValueError("Unsupported regularization type")
        self.__regularization_lambda = reg_lambda

    def set_gradient_clipper(self, gradient_clipper, clip_value=1.0):
        gradient_clipper = gradient_clipper.strip().lower()

        if gradient_clipper == "none":
            self.__current_gradient_clipper = _GradientClipper.NONE
            self.__gradient_clip_value = None
            return

        if clip_value is None or clip_value <= 0:
            raise ValueError("Clip value must be a positive number")
        
        elif gradient_clipper == "norm":
            self.__current_gradient_clipper = _GradientClipper.NORM
        elif gradient_clipper == "value":
            self.__current_gradient_clipper = _GradientClipper.VALUE
        else:
            raise ValueError("Unsupported gradient clipper type")
        
        self.__gradient_clip_value = clip_value

    def get_base_lr(self):
        return self.lr_scheduler.get_base_lr()
    
    def set_base_lr(self, base_lr):
        self.lr_scheduler.set_base_lr(base_lr)

    def train(self, X, Y, epochs, lr=None, training_method=None, log_interval=None, batch_size=None, suppress_output=False, reset_all=False, reset_losses=False, lr_override=True):
        if training_method is None:
            if self.__current_training_method is None:
                raise ValueError("Training method not set. Please set it using set_training_method or provide it in train method.")
        else:
            self.set_training_method(training_method)

        if reset_all:
            self.reset()

        if reset_losses:
            self.reset_losses()

        if lr_override:
            if lr is not None:
                self.lr_scheduler.set_base_lr(lr)

        if self.__current_training_method == _TrainingMethod.GD:
            if log_interval is None:
                log_interval = 10
            self.__gradient_descent(X=X, Y=Y, epochs=epochs, log_interval=log_interval, suppress_output=suppress_output)
        elif self.__current_training_method == _TrainingMethod.SGD:
            if log_interval is None:
                log_interval = 100
            self.__stochastic_gradient_descent(X=X, Y=Y, epochs=epochs, log_interval=log_interval, suppress_output=suppress_output)
        elif self.__current_training_method == _TrainingMethod.BGD:
            if batch_size is None:
                batch_size = 32
            if log_interval is None:
                log_interval = 100
            self.__batch_gradient_descent(X=X, Y=Y, epochs=epochs, log_interval=log_interval, batch_size=batch_size, suppress_output=suppress_output)
        else:
            print(f"Avalible Methods: {self.__available_training_methods}")

        return list(self.__losses)
    
    def plot_losses(self):
        import matplotlib.pyplot as plt
        plt.plot(self.__losses)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Loss over Iterations")
        plt.show()