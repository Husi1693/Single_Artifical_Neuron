import math

def sigmoid (x):
    return (1/(1+math.exp(-x)))

def loss(y, y_hat):
    return (y - y_hat)**2


learning_rate = 0.1

input1 = 1
input2 = 0

weight1 = 0.5
weight2 = -0.25


bias = 3

for i in range(1000):
    weighted_sum = input1 * weight1 + input2 * weight2 + bias
    y = 1
    y_hat = sigmoid(weighted_sum)

    grad_loss = -2*(y - y_hat)
    grad_sigmoid = y_hat*(1-y_hat)
    grad_w1 = grad_loss * grad_sigmoid *input1
    grad_w2 = grad_loss * grad_sigmoid *input2
    grad_bias = grad_loss * grad_sigmoid * 1

    weight1 = weight1-learning_rate*grad_w1
    weight2 = weight2-learning_rate*grad_w2

    bias = bias - learning_rate*grad_bias

    if i % 100 == 0:
        print(i, loss(y, y_hat))

