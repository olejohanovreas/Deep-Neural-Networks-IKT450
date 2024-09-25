import torch
import torch.nn as nn
import torch.optim as optim


#crim	tax	rm	age	ptratio	medv
training_dataset = """
0.00632	296	6.575	65.2	15.3	24
0.02731	242	6.421	78.9	17.8	21.6
0.03237	222	6.998	45.8	18.7	33.4
0.06905	222	7.147	54.2	18.7	36.2
0.08829	311	6.012	66.6	15.2	22.9
0.22489	311	6.377	94.3	15.2	15
0.11747	311	6.009	82.9	15.2	18.9
0.09378	311	5.889	39	15.2	21.7
0.62976	307	5.949	61.8	21	20.4
"""


training_dataset = [[float(f) for f in i.split("\t")] for i in  training_dataset.strip().split("\n")]

#Making into a binary classifier
training_dataset = [row[:-1]+[0 if row[-1]<25 else 1] for row in training_dataset]
training_dataset =training_dataset*1000

#print(training_dataset)
X = torch.Tensor([i[0:5] for i in training_dataset])
Y = torch.Tensor([i[5] for i in training_dataset])








class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(5,2)
        self.fc2 = nn.Linear(2,1)

    def forward(self,x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

model = Net()
print(model)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

allloss = []

for epoch in range(500):
    print(epoch)
    outputs = model(X)
    loss = criterion(outputs,Y)
    loss.backward()
    optimizer.step()
    allloss.append(loss.item())
import pdb;pdb.set_trace()


import matplotlib.pyplot as plt
plt.plot(allloss)
plt.show()

print(list(model.parameters()))

import sys
sys.exit(0)

#From scratch
weights = [-0.1, 0.20, -0.23, -0.1, 0.20, -0.23, -0.1, 0.20, -0.23]

import math
def sigmoid(z):
    if(z<-100):
        return 0
    if(z>100):
        return 1
    return 1.0/math.exp(-z)

def firstLayer(row,weights):
    activation_1 = weights[0]*1
    activation_1 += weights[1]*row[0]
    activation_1 += weights[2]*row[1]

    activation_2 = weights[3]*1
    activation_2 += weights[4]*row[2]
    activation_2 += weights[5]*row[3]
    return sigmoid(activation_1),sigmoid(activation_2)

def secondLayer(row,weights):
    activation_3 = weights[6]
    activation_3 += weights[7]*row[0]
    activation_3 += weights[8]*row[1]
    return sigmoid(activation_3)

def predict(row,weights):
    input_layer = row
    first_layer = firstLayer(input_layer,weights)
    second_layer = secondLayer(first_layer,weights)
    return second_layer,first_layer


for d in training_dataset:
    print(predict(d,weights)[0],d[-1])   #Prints y_hat and y


def train_weights(train,learningrate,epochs):
    for epoch in range(epochs):
        sum_error = 0.0
        for row in train:
            prediction,first_layer = predict(row,weights)
            error = row[-1]-prediction
            sum_error += error
            #First layer
            weights[0] = weights[0] + learningrate*error*1
            weights[3] = weights[3] + learningrate*error

            weights[1] = weights[1] + learningrate*error*row[0]
            weights[2] = weights[2] + learningrate*error*row[1]
            weights[4] = weights[4] + learningrate*error*row[2]
            weights[5] = weights[5] + learningrate*error*row[3]

            #Second layer
            weights[6] = weights[6] + learningrate*error
            weights[7] = weights[7] + learningrate*error*first_layer[0]
            weights[8] = weights[8] + learningrate*error*first_layer[1]
        if((epoch%100==0) or (last_error != sum_error)):
            print("Epoch "+str(epoch) + " Learning rate " + str(learningrate) + " Error " + str(sum_error))
        last_error = sum_error
    return weights

learningrate = 0.0001#0.00001
epochs = 1000
train_weights = train_weights(training_dataset,learningrate,epochs)
print(train_weights)
