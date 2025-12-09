# NanoNet – Neural Network From Scratch (NumPy Only)

This project implements a fully functional **multi-layer neural network from scratch** using only **NumPy**.  
No TensorFlow, no PyTorch — just pure mathematics and linear algebra.

The network is trained on the **MNIST handwritten digits dataset** ([data/train.csv](data/train.csv) format) and supports:

- He initialization  
- ReLU activation  
- Softmax output  
- Cross-entropy loss  
- Full forward + backward propagation  
- Gradient descent optimization  
- Accuracy + loss tracking  
- A clean [`NeuralNet`](NeuralNet.py) class with `.train()`, `.predict()`, `.evaluate()`  

---

## 🚀 Features

- **Modular architecture** – define arbitrary layers like `[784, 128, 64, 10]`
- **Pure NumPy implementation** – complete understanding of the internals
- **Training loop with loss + accuracy logging**
- **Easy prediction + evaluation**
- **Object-oriented design** ([NeuralNet.py](NeuralNet.py))

---

## 📁 Project Structure

```
/
├── NeuralNet.py        # Neural network implementation (class)
├── train.py            # Script to load MNIST, train, and evaluate
└── data/
    └── train.csv       # MNIST dataset (Kaggle digit recognizer format)
```

---

## 🧠 How It Works

1. **Weights initialized using He initialization**
2. **Forward pass**
   - Hidden layers → ReLU  
   - Output layer → Softmax  
3. **Loss computed using cross-entropy**
4. **Backpropagation**
   - Softmax gradient  
   - ReLU derivative  
   - Layer-by-layer weight updates  
5. **Gradient descent training**

This mimics the internal logic of modern deep learning libraries — but written manually.

---

## 🏁 Usage

### **Train the model**

Run:

```bash
python3 train.py
```

Your training output will log:

```
Iteration 0   Accuracy: 0.07   Loss: 2.38
Iteration 10  Accuracy: 0.45   Loss: 1.95
...
```

---

## 🧪 Evaluate

Inside [train.py](train.py):

```python
model.evaluate(X_test, Y_test)
```

Example output:

```
Test Accuracy: 0.92
```

---

## 🔍 Sample Prediction

```python
model.test_predictions(10, X_test, Y_test)
```

Output:

```
Model Prediction: 5
True Label: 5
```

---

## 📚 Requirements

- Python 3.9+
- NumPy
- Pandas
- Matplotlib (optional, for visualization)

Install:

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy pandas
```

---

## 🌟 Future Improvements

- Add mini-batch gradient descent  
- Add Adam optimizer  
- Add regularization (L2 / dropout)  
- Add plotting utilities for loss + accuracy  
- Save/load model weights  
- Implement momentum-based training  

---

## 📜 License

MIT License – free to use, modify, and learn from.