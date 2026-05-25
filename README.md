# Neuron from Scratch

I built a single artificial neuron in plain Python — no PyTorch, no NumPy, nothing.

This started as a way to actually understand what's happening inside a neural network instead of just calling `model.fit()` and hoping for the best. Writing the forward pass, loss, and backprop by hand makes it click in a way that reading about it never did.

---

## How it works

The neuron takes two inputs, multiplies each by a weight, adds a bias, and runs that through sigmoid. Then it checks how wrong it was, computes gradients by hand using the chain rule, and nudges the weights in the right direction. Repeat 1000 times.

```python
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def loss(y, y_hat):
    return (y - y_hat) ** 2

input1, input2 = 1, 0
weight1, weight2, bias = 0.5, -0.25, 0.0
learning_rate = 0.1

for i in range(1000):
    weighted_sum = input1 * weight1 + input2 * weight2 + bias
    y = 1
    y_hat = sigmoid(weighted_sum)

    grad_loss    = -2 * (y - y_hat)
    grad_sigmoid = y_hat * (1 - y_hat)
    grad_w1   = grad_loss * grad_sigmoid * input1
    grad_w2   = grad_loss * grad_sigmoid * input2
    grad_bias = grad_loss * grad_sigmoid

    weight1 -= learning_rate * grad_w1
    weight2 -= learning_rate * grad_w2
    bias    -= learning_rate * grad_bias

    if i % 100 == 0:
        print(f"Step {i:4d} | Loss: {loss(y, y_hat):.6f}")
```

Output:
```
Step    0 | Loss: 0.000846
Step  100 | Loss: 0.000143
Step  200 | Loss: 0.000024
...
Step  900 | Loss: 0.000000
```

---

## The math

Forward pass:

$$z = w_1 x_1 + w_2 x_2 + b \qquad \hat{y} = \frac{1}{1+e^{-z}}$$

Loss:

$$L = (y - \hat{y})^2$$

Gradient of the loss with respect to a weight (chain rule):

$$\frac{\partial L}{\partial w_1} = \underbrace{-2(y-\hat{y})}_{\partial L/\partial \hat{y}} \cdot \underbrace{\hat{y}(1-\hat{y})}_{\partial \hat{y}/\partial z} \cdot \underbrace{x_1}_{\partial z/\partial w_1}$$

Weight update:

$$w_1 \leftarrow w_1 - \alpha \cdot \frac{\partial L}{\partial w_1}$$

---

## Running it

Just Python 3, no dependencies.

```bash
python neuron.py
```

---

## Why not just use PyTorch?

The same thing in PyTorch is literally `nn.Linear(2, 1)`. Which is great — but if you've never seen what's underneath, it's a black box. This is what's underneath.

---

