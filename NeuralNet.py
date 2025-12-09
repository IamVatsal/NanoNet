import numpy as np

class NeuralNet:
    def __init__(self, layers, method="he"):
        np.random.seed(1)
        if method == "he":
            self.__W = [np.random.randn(layers[i], layers[i + 1]) * np.sqrt(2 / layers[i]) for i in range(len(layers) - 1)]
            self.__B = [np.zeros((1, layers[i + 1])) for i in range(len(layers) - 1)]
            self.__losses = []
            self.__traing_methods = ["gradient_descent"]
    
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
        # print("X.shape:", X.shape)
        for i in range(len(W) - 1):
            # print("Layer: ", i, " W[i] shape:", w[i].shape)
            A, Z = self.__hidden_layer(As[-1], W[i], B[i])
            As.append(A)
            Zs.append(Z)
        out = self.__output_layer(As[-1], W[-1], B[-1])
        return Zs, As, out

    def __one_hot(self, Y, num_classes):
        one_hot_Y = np.eye(num_classes)[Y.astype(int)]
        return one_hot_Y

    def __back_prop(self, Zs, As, output, W, X, Y):
        # Number of training examples
        m = X.shape[0]

        # Number of layers (including output layer)
        L = len(W)

        # One-hot encode labels
        num_classes = output.shape[1]
        one_hot_Y = self.__one_hot(Y, num_classes)

        # Gradients containers
        dW = [None] * L
        dB = [None] * L

        # Gradient for output layer (__softmax + cross-entropy)
        dZ = (output - one_hot_Y) / m
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
        log_likelihood = -np.log(output[range(m), Y] + 1e-8)
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

    def __gradient_descent(self, X, Y, lr, epochs):
        W = self.__W
        B = self.__B
        for i in range(epochs + 1):
            Zs, As, output = self.__forward_pass(X, W, B)
            dW, dB = self.__back_prop(Zs, As, output, W, X, Y)
            W, B = self.__update_params(W, B,dW, dB, lr)
            if i % 10 == 0:
                loss = self.__compute_loss(output, Y)
                predictions = self.__get_predictions(output)
                acc = self.__get_accuracy(predictions, Y)
                print(f"Iteration {i}, Accuracy: {acc}, Loss: {loss}")
                self.__losses.append(loss)

        self.__W = W
        self.__B = B
    
    def predict(self, X):
        _, _, output = self.__forward_pass(X, self.__W, self.__B)
        predictions = self.__get_predictions(output)
        return predictions

    def evaluate(self, X, Y):
        predictions = self.predict(X)
        accuracy = self.__get_accuracy(predictions, Y)
        print(f"Test Accuracy: {accuracy}")

    def test_predictions(self, index, X, Y):
        prediction = self.predict(X[index].reshape(1, -1))
        print(f"Model Prediction: {prediction[0]}")
        print(f"True Label: {Y[index]}")
        # current_image = X[index].reshape(28, 28)
        # plt.imshow(current_image, cmap='gray')
        # plt.show()

    def get_losses(self):
        return self.__losses

    def avalible_training_methods(self):
        print(f"Avalible Methods: {self.__traing_methods}")

    def train(self, X, Y, lr, epochs, traing_method="gradient_descent"):
        if traing_method == "gradient_descent":
            self.__gradient_descent(X, Y, lr, epochs)
        else:
            print(f"Avalible Methods: {self.__traing_methods}")