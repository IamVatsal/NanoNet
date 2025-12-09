import numpy as np
import pandas as pd

from NeuralNet import NeuralNet

data = pd.read_csv("./data/train.csv")

print(data.head())
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_test = data_dev[0]
X_test = data_dev[1:n] / 255.0
X_test = X_test.T

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n] / 255.0
X_train = X_train.T

# layers[0] is input layer and layers[-1] is output layer
layers = [784, 128, 64, 10]

model = NeuralNet(layers=layers)

# model.avalible_training_methods()

# It is Useing Gradient Descent by Default
model.train(X_train, Y_train, lr=0.05, epochs=500)

# Evaluate the model on test data
model.evaluate(X_test, Y_test)

# Test some predictions
model.test_predictions(10, X_test, Y_test)
model.test_predictions(25, X_test, Y_test)

# Get training losses
# losses = model.get_losses()
# print("Training Losses:", losses)