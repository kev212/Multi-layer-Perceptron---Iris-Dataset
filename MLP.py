import csv
import random
import math
import matplotlib.pyplot as plt

#load dataset
with open('iris.csv') as file:
    csvreader = csv.reader(file)
    dataset = list(csvreader)

# Change string value to numeric
for row in dataset:
    row[4] = ["setosa", "versicolor", "virginica"].index(row[4])
    row[:4] = [float(row[j]) for j in range(len(row)-1)]
    
# Split x and y (feature and target)
random.shuffle(dataset)
datatrain = dataset[:int(len(dataset) * 0.8)]
datatest = dataset[int(len(dataset) * 0.8):]
train_x = [data[:4] for data in datatrain]
train_y = [data[4] for data in datatrain]
test_x = [data[:4] for data in datatest]
test_y = [data[4] for data in datatest]

"""
Multilayer perceptron model, with one hidden layer.
input layer : 4 neuron, represents the feature of Iris
hidden layer : 3 neuron, activation using sigmoid
output layer : 3 neuron, represents the class of Iris
optimizer = gradient descent
loss function = Square ROot Error
"""

def matrix_mul_bias(A, B, bias): # Matrix multiplication (for Testing)
    C = [[0 for i in range(len(B[0]))] for i in range(len(A))]    
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                C[i][j] += A[i][k] * B[k][j]
            C[i][j] += bias[j]
    return C

def vec_mat_bias(A, B, bias): # Vector (A) x matrix (B) multiplication
    C = [0 for i in range(len(B[0]))]
    for j in range(len(B[0])):
        for k in range(len(B)):
            C[j] += A[k] * B[k][j]
            C[j] += bias[j]
    return C

def mat_vec(A, B): # Matrix (A) x vector (B) multipilicatoin (for backprop)
    C = [0 for i in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B)):
            C[i] += A[i][j] * B[j]
    return C

def sigmoid(A, deriv=False):
    if deriv: # derivation of sigmoid (for backprop)
        for i in range(len(A)):
            A[i] = A[i] * (1 - A[i])
    else:
        for i in range(len(A)):
            A[i] = 1 / (1 + math.exp(-A[i]))
    return A

# Define parameter
alfa = 0.005
epoch = 400
neuron = [4, 3, 3] # number of neuron each layer
error_train = []
error_test = []
accuracy_train = []
accuracy_test = []

# Initiate weight and bias with 0 value
weight = [[0 for j in range(neuron[1])] for i in range(neuron[0])]
weight_2 = [[0 for j in range(neuron[2])] for i in range(neuron[1])]
bias = [0 for i in range(neuron[0])]
bias_2 = [0 for i in range(neuron[1])]

# Initiate weight with random between -1.0 ... 1.0
for i in range(neuron[0]):
    for j in range(neuron[1]):
        weight[i][j] = 2 * random.random() - 1

for i in range(neuron[1]):
    for j in range(neuron[2]):
        weight_2[i][j] = 2 * random.random() - 1
        
for e in range(epoch):
    cost_total_train = 0
    cost_total_test = 0
    for idx, x in enumerate(train_x): # Update for each data; SGD
        
        # Forward propagation
        h_1 = vec_mat_bias(x, weight, bias)
        X_1 = sigmoid(h_1)
        h_2 = vec_mat_bias(X_1, weight_2, bias_2)
        X_2 = sigmoid(h_2)
        
        # Convert to One-hot target
        target = [0, 0, 0]
        target[int(train_y[idx])] = 1

        # Cost function, Square Root Eror
        eror = 0
        for i in range(3):
            eror +=  0.5 * (target[i] - X_2[i]) ** 2 
        cost_total_train += eror
        
        
        # Backward propagation
        # Update weight_2 and bias_2 (layer 2)
        delta_2 = []
        for j in range(neuron[2]):
            delta_2.append(-1 * (target[j]-X_2[j]) * X_2[j] * (1-X_2[j]))

        for i in range(neuron[1]):
            for j in range(neuron[2]):
                weight_2[i][j] -= alfa * (delta_2[j] * X_1[i])
                bias_2[j] -= alfa * delta_2[j]
        
        # Update weight and bias (layer 1)
        delta_1 = mat_vec(weight_2, delta_2)
        for j in range(neuron[1]):
            delta_1[j] = delta_1[j] * (X_1[j] * (1-X_1[j]))
        
        for i in range(neuron[0]):
            for j in range(neuron[1]):
                weight[i][j] -=  alfa * (delta_1[j] * x[i])
                bias[j] -= alfa * delta_1[j]
    
    #menghitung eror pada data test / validasi
    for idx, x in enumerate(test_x):
        # Forward propagation
        h_1 = vec_mat_bias(x, weight, bias)
        X_1 = sigmoid(h_1)
        h_2 = vec_mat_bias(X_1, weight_2, bias_2)
        X_2 = sigmoid(h_2)
        
        # Convert to One-hot target
        target = [0, 0, 0]
        target[int(test_y[idx])] = 1

        # Cost function, Square Root Eror
        eror = 0
        for i in range(3):
            eror +=  0.5 * (target[i] - X_2[i]) ** 2 
        cost_total_test += eror               
                
    cost_total_train /= len(train_x)
    error_train.append(cost_total_train)
    cost_total_test /= len(test_x)
    error_test.append(cost_total_test)
    
    #menghitung akurasi data train
    res = matrix_mul_bias(train_x, weight, bias)
    res_2 = matrix_mul_bias(res, weight_2, bias)
    # Get prediction
    preds = []
    for r in res_2:
        preds.append(max(enumerate(r), key=lambda x:x[1])[0])
    acc = 0.0
    for i in range(len(preds)):
        if preds[i] == int(train_y[i]):
            acc += 1        
    acc = acc / len(preds) * 100
    accuracy_train.append(acc)
    
    #menghitung akurasi data test / validasi
    res = matrix_mul_bias(test_x, weight, bias)
    res_2 = matrix_mul_bias(res, weight_2, bias)
    # Get prediction
    preds = []
    for r in res_2:
        preds.append(max(enumerate(r), key=lambda x:x[1])[0])
    acc = 0.0
    for i in range(len(preds)):
        if preds[i] == int(test_y[i]):
            acc += 1        
    acc = acc / len(preds) * 100
    accuracy_test.append(acc)      

# Print prediction
print("Actual : ")
print(test_y)
print("Prediction : ")
print (preds)

# Calculate accuration
acc = 0.0
for i in range(len(preds)):
    if preds[i] == int(test_y[i]):
        acc += 1
print("Accuracy : ")        
print (acc / len(preds) * 100, "%")

#Plot Total Error vs Epoch
plt.title('Grafik Error pada tiap epoch')
plt.plot(range(epoch),error_train)
plt.plot(range(epoch),error_test, color='red')
plt.legend(['Train', 'Test'])
plt.ylabel('Total Error')
plt.xlabel('Epoch')
plt.show()

#Plot Akurasi vs Epoch
plt.title('Grafik Akurasi pada tiap epoch')
plt.plot(range(epoch),accuracy_train)
plt.plot(range(epoch),accuracy_test, color='red')
plt.legend(['Train', 'Test'])
plt.ylabel('Accuracy %')
plt.xlabel('Epoch')
plt.show()
