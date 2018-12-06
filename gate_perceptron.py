import numpy as np
import math
from numpy import random, dot, array
import matplotlib.pyplot as plt


# This function assigns the x and y value of an array.
# Then, it computes the ideal output based on the x and y values.
def correction(p):
    x = p[0]
    y = p[1]
    if x == 1 and y == 1:
        return 1
    else:
        return 0

# This is the activation function. 
def sigmoid(a):
    return 1 / (1 + math.exp(-a))


# On this block, the weights are being randomly initialized.
w = random.rand(2)
print(" Weights: " + str(w))
iterations = 0
learning_rate = 0.1
bias = 0

# This is the training data:
data = [array([0,0]), array([0,1]), array([1,0]), array([1,1])]
answers = []

for i in range(len(data)):
    answers.append(correction(data[i]))



while iterations < 8000:
    
    k = random.randint(len(data))
    x = data[k]
    
    w_sum = np.sum(w*x)                   # Take the weighted sum of the input vector and weights vector
    w_sum_bias = w_sum + bias
    desired_output = answers[k]
    output = sigmoid(w_sum_bias)

    error = desired_output   -    output
    loss = ((error) ** 2)                 # Loss function
    

    dloss_dw = 2 * (error) * -x           # derivative of loss function with respect to the weights
    
    w = w - learning_rate * dloss_dw      # updating weights
    bias = bias + learning_rate * error   # updating bias
    
    iterations = iterations + 1


for i in range(len(data)):
    x = data[i]
    final_w_sum = dot(w,x) + bias
    final_output = sigmoid(final_w_sum)
    print("({}, {}) -> {} = {}".format(x[0], x[1], correction(data[i]), final_output))

    
print(" dloss_dw: " + str(dloss_dw))
print(" Error at iteration " + str(iterations) + " :  " + str(error))
print(" Updated Weights: " + str(w))
print(f'bias {bias}')